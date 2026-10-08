// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::aggregates::group_values::multi_group_by::GroupColumn;
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBufferBuilder, DictionaryArray, Int64Array,
    PrimitiveArray,
};
use arrow::compute::take;
use arrow::datatypes::{ArrowDictionaryKeyType, ArrowNativeType, DataType, Field};
use datafusion_common::hash_utils::RandomState;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{Result, exec_err};
use datafusion_execution::memory_pool::proxy::HashTableAllocExt;
use hashbrown::hash_table::HashTable;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use crate::aggregates::AGGREGATION_HASH_SEED;

/// [`GroupColumn`] for dictionary-encoded columns with key type `K`.
///
/// `inner` holds one slot per distinct value seen across all batches.
/// `group_to_inner[group_idx]` maps each group to its slot in `inner`,
/// so groups with the same value share a slot rather than duplicating data.
pub struct DictionaryGroupValuesColumn<K: ArrowDictionaryKeyType + Send + Sync> {
    /// Deduplicated store of distinct values.
    inner: Box<dyn GroupColumn>,
    /// Unary null array (length 1) reused for every null appended to `inner`.
    null_array: ArrayRef,
    /// Maps each group index to its slot in `inner`.
    group_to_inner: Vec<usize>,
    /// Lookup table mapping `(value_hash, inner_slot)` for each non-null distinct value.
    value_dedup: HashTable<(u64, usize)>,
    /// Tracked allocation size of `value_dedup` for memory accounting via `size()`.
    value_dedup_size: usize,
    /// Slot in `inner` for the null group; `None` until the first null is seen.
    null_inner_slot: Option<usize>,
    /// Hash seed — must match `create_hashes` so hashes are consistent across calls.
    random_state: RandomState,
    /// Maps `val_idx → inner_slot` for the values array in `cached_values`.
    ///
    /// Valid only while `cached_values` is `Some` and matches the incoming
    /// values array; `usize::MAX` means "not yet resolved". Entries are only
    /// ever filled in, never invalidated, because `inner` slots are stable
    /// under append. `take_n` remaps `inner` and therefore drops the cache.
    val_to_inner: Vec<usize>,
    /// Hashes of the values array in `cached_values`, one per `val_idx`.
    val_hashes: Vec<u64>,
    /// The values array that `val_hashes` and `val_to_inner` were built for.
    /// When an incoming `dict.values()` is `Arc::ptr_eq` to this, both caches
    /// are reused instead of being rebuilt. See [`Self::sync_value_cache`].
    cached_values: Option<ArrayRef>,
    _phantom: PhantomData<K>,
}

impl<K: ArrowDictionaryKeyType + Send + Sync> DictionaryGroupValuesColumn<K> {
    pub fn new(inner: Box<dyn GroupColumn>, field: &Field) -> Self {
        let null_array = arrow::array::new_null_array(field.data_type(), 1);
        Self {
            inner,
            null_array,
            group_to_inner: Vec::new(),
            value_dedup: HashTable::new(),
            value_dedup_size: 0,
            null_inner_slot: None,
            random_state: AGGREGATION_HASH_SEED,
            val_to_inner: Vec::default(),
            val_hashes: Vec::default(),
            cached_values: None,
            _phantom: PhantomData,
        }
    }

    /// Build a `DictionaryArray` from `values` (all inner slots) and the
    /// per-group slot mapping.  The null inner slot, if any, is excluded from
    /// the values array and its groups emit a null key — so it never consumes
    /// a key index regardless of where it sits in `inner`.
    fn into_dict(
        values: ArrayRef,
        group_to_inner: &[usize],
        null_inner_slot: Option<usize>,
    ) -> ArrayRef {
        let Some(null_slot) = null_inner_slot else {
            // Fast path: no null group — raw slot indices are valid keys.
            let keys: PrimitiveArray<K> = group_to_inner
                .iter()
                .map(|&slot| Some(K::Native::usize_as(slot)))
                .collect();
            return Arc::new(DictionaryArray::<K>::new(keys, values));
        };

        // Build a compact remap: each non-null slot gets a contiguous key
        // starting from 0; the null slot is skipped entirely.
        let n = values.len();
        let mut remap = vec![0usize; n];
        let mut next = 0usize;
        for (i, mapped) in remap.iter_mut().enumerate() {
            if i != null_slot {
                *mapped = next;
                next += 1;
            }
        }

        let keys: PrimitiveArray<K> = group_to_inner
            .iter()
            .map(|&slot| {
                if slot == null_slot {
                    None
                } else {
                    Some(K::Native::usize_as(remap[slot]))
                }
            })
            .collect();

        // Compact values array: drop the null slot so key indices stay tight.
        let compact_indices: Int64Array = (0..n)
            .filter(|&i| i != null_slot)
            .map(|i| i as i64)
            .collect();
        let compact =
            take(&*values, &compact_indices, None).expect("compact values in into_dict");
        Arc::new(DictionaryArray::<K>::new(keys, compact))
    }

    // https://github.com/apache/datafusion/issues/23127
    // Null groups emit a null key (None), not a slot index, so the null inner
    // slot never consumes a key index regardless of its position in inner.
    fn check_key_overflow(&self) -> Result<()> {
        let non_null_count = self.inner.len() - self.null_inner_slot.is_some() as usize;
        if !Self::key_type_fits(non_null_count) {
            return exec_err!(
                "Dictionary key type {:?} cannot represent {} distinct values",
                K::DATA_TYPE,
                non_null_count
            );
        }
        Ok(())
    }

    fn key_type_fits(num_values: usize) -> bool {
        let max: usize = match K::DATA_TYPE {
            DataType::Int8 => i8::MAX as usize,
            DataType::Int16 => i16::MAX as usize,
            DataType::Int32 => i32::MAX as usize,
            DataType::Int64 => i64::MAX as usize,
            DataType::UInt8 => u8::MAX as usize,
            DataType::UInt16 => u16::MAX as usize,
            DataType::UInt32 => u32::MAX as usize,
            DataType::UInt64 => usize::MAX,
            _ => return false,
        };
        num_values == 0 || num_values - 1 <= max
    }

    fn hash_values(&mut self, values: &ArrayRef) {
        self.cached_values = None;
        self.val_hashes.clear();
        self.val_hashes.resize(values.len(), 0);
        create_hashes(
            std::slice::from_ref(values),
            &self.random_state,
            &mut self.val_hashes,
        )
        .unwrap();
    }

    /// Makes `val_hashes` and `val_to_inner` correspond to `dict_values`,
    /// reusing them when they were already built for this same values array.
    ///
    /// `take`, `filter` and repartitioning clone the values `Arc` and rewrite
    /// only the keys, so batches in a partition usually share one values
    /// array; without this the per-batch cost is O(dictionary cardinality)
    /// rather than O(rows). `take_n` remaps `inner` and clears
    /// `cached_values`, forcing a rebuild.
    fn sync_value_cache(&mut self, dict_values: &ArrayRef) {
        if self
            .cached_values
            .as_ref()
            .is_some_and(|c| Arc::ptr_eq(c, dict_values))
        {
            return;
        }
        self.hash_values(dict_values);
        self.val_to_inner.clear();
        self.val_to_inner.resize(dict_values.len(), usize::MAX);
        self.cached_values = Some(Arc::clone(dict_values));
    }

    fn find_or_insert_value(
        &mut self,
        dict_values: &ArrayRef,
        val_idx: usize,
        hash: u64,
    ) -> Result<usize> {
        let inner = &*self.inner;
        let existing = self
            .value_dedup
            .find(hash, |&(entry_hash, slot)| {
                entry_hash == hash && inner.equal_to(slot, dict_values, val_idx)
            })
            .map(|&(_, slot)| slot);

        match existing {
            Some(slot) => Ok(slot),
            None => {
                let slot = self.inner.len();
                self.inner.append_val(dict_values, val_idx)?;
                self.value_dedup.insert_accounted(
                    (hash, slot),
                    |&(entry_hash, _)| entry_hash,
                    &mut self.value_dedup_size,
                );
                Ok(slot)
            }
        }
    }

    /// Compares the value at `val_idx` against `inner[lhs_slot]` directly,
    /// memorizing `lhs_slot` in `val_to_inner` when they are equal.
    ///
    /// Each distinct value occupies exactly one slot in `inner`, so an equal
    /// comparison proves `lhs_slot` is the slot for `val_idx` without probing
    /// `value_dedup` or hashing. An unequal result is deliberately not
    /// memoized: it says which slot the value is *not* in, not which one it
    /// is. That is what keeps `val_to_inner` fill-only and never stale.
    ///
    /// Callers must have run [`Self::sync_value_cache`] for `dict_values`.
    ///
    /// Only called on a `val_to_inner` miss. It takes the fields it needs
    /// rather than `&mut self`, so the caller's hot loop can keep its slices
    /// of `val_to_inner` and `group_to_inner` in registers across the call.
    #[cold]
    #[inline(never)]
    fn equal_to_uncached(
        val_to_inner: &mut [usize],
        inner: &dyn GroupColumn,
        null_array: &ArrayRef,
        dict_values: &ArrayRef,
        val_idx: usize,
        lhs_slot: usize,
    ) -> bool {
        let equal = if dict_values.is_null(val_idx) {
            inner.equal_to(lhs_slot, null_array, 0)
        } else {
            inner.equal_to(lhs_slot, dict_values, val_idx)
        };

        if equal {
            val_to_inner[val_idx] = lhs_slot;
        }
        equal
    }

    fn find_or_insert_null(&mut self) -> Result<usize> {
        if let Some(slot) = self.null_inner_slot {
            return Ok(slot);
        }
        let slot = self.inner.len();
        self.inner.append_val(&self.null_array, 0)?;
        self.null_inner_slot = Some(slot);
        Ok(slot)
    }

    /// Per-row fallback for `vectorized_equal_to` used when the values array is
    /// not cached and the number of rows to check is smaller than the
    /// dictionary cardinality, making the O(D) hashing in `sync_value_cache`
    /// more expensive than direct value comparison.
    ///
    /// `#[cold]` + `#[inline(never)]` keeps this code out of the hot
    /// `val_to_inner` loops in `vectorized_equal_to` so LLVM can pipeline them.
    #[cold]
    #[inline(never)]
    fn equal_to_per_row(
        &self,
        lhs_rows: &[usize],
        dict_values: &ArrayRef,
        dict: &DictionaryArray<K>,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        let group_to_inner = self.group_to_inner.as_slice();
        for (idx, (&lhs_row, &rhs_row)) in
            lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
        {
            if !equal_to_results.get_bit(idx) {
                continue;
            }
            let lhs_slot = group_to_inner[lhs_row];
            let equal = match dict.key(rhs_row) {
                None => self.inner.equal_to(lhs_slot, &self.null_array, 0),
                Some(val_idx) if dict_values.is_null(val_idx) => {
                    self.inner.equal_to(lhs_slot, &self.null_array, 0)
                }
                Some(val_idx) => self.inner.equal_to(lhs_slot, dict_values, val_idx),
            };
            if !equal {
                equal_to_results.set_bit(idx, false);
            }
        }
    }
}

impl<K: ArrowDictionaryKeyType + Send + Sync> GroupColumn
    for DictionaryGroupValuesColumn<K>
{
    fn equal_to(&self, lhs_row: usize, array: &ArrayRef, rhs_row: usize) -> bool {
        let lhs_slot = self.group_to_inner[lhs_row];
        let dict = array.as_dictionary::<K>();
        match dict.key(rhs_row) {
            None => self.inner.equal_to(lhs_slot, &self.null_array, 0),
            Some(val_idx) if dict.values().is_null(val_idx) => {
                self.inner.equal_to(lhs_slot, &self.null_array, 0)
            }
            Some(val_idx) => self.inner.equal_to(lhs_slot, dict.values(), val_idx),
        }
    }

    fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        let dict = array.as_dictionary::<K>();
        let inner_slot = match dict.key(row) {
            None => self.find_or_insert_null()?,
            Some(val_idx) if dict.values().is_null(val_idx) => {
                self.find_or_insert_null()?
            }
            Some(val_idx) => {
                let dict_values = dict.values();
                self.sync_value_cache(dict_values);
                self.find_or_insert_value(dict_values, val_idx, self.val_hashes[val_idx])?
            }
        };
        self.group_to_inner.push(inner_slot);
        self.check_key_overflow()
    }

    fn vectorized_equal_to(
        &mut self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        let dict = array.as_dictionary::<K>();
        let dict_keys = dict.keys();
        let dict_values = dict.values();

        let cached = self
            .cached_values
            .as_ref()
            .is_some_and(|c| Arc::ptr_eq(c, dict_values));
        if !cached && rhs_rows.len() < dict_values.len() {
            self.equal_to_per_row(
                lhs_rows,
                dict_values,
                dict,
                rhs_rows,
                equal_to_results,
            );
            return;
        }
        self.sync_value_cache(dict_values);

        let raw_keys = dict_keys.values();
        let group_to_inner = self.group_to_inner.as_slice();
        let val_to_inner = self.val_to_inner.as_mut_slice();
        let inner = &*self.inner;
        let null_array = &self.null_array;
        let null_inner_slot = self.null_inner_slot;
        if dict_keys.null_count() == 0 {
            // No null keys : skip the get_bit guard: we only ever write false,
            // so overwriting an already-false bit is a no-op.
            for (idx, (&lhs_row, &rhs_row)) in
                lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
            {
                let val_idx = raw_keys[rhs_row].as_usize();
                let lhs_slot = group_to_inner[lhs_row];
                let equal = match val_to_inner[val_idx] {
                    usize::MAX => Self::equal_to_uncached(
                        val_to_inner,
                        inner,
                        null_array,
                        dict_values,
                        val_idx,
                        lhs_slot,
                    ),
                    slot => slot == lhs_slot,
                };
                if !equal {
                    equal_to_results.set_bit(idx, false);
                }
            }
        } else {
            let null_buf = dict_keys.nulls().unwrap();
            for (idx, (&lhs_row, &rhs_row)) in
                lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
            {
                if equal_to_results.get_bit(idx) {
                    // A null key is not a position in the values array, so it
                    // resolves straight to the null slot instead of going
                    // through `val_to_inner`.
                    let lhs_slot = group_to_inner[lhs_row];
                    let equal = if null_buf.is_null(rhs_row) {
                        null_inner_slot == Some(lhs_slot)
                    } else {
                        let val_idx = raw_keys[rhs_row].as_usize();
                        match val_to_inner[val_idx] {
                            usize::MAX => Self::equal_to_uncached(
                                val_to_inner,
                                inner,
                                null_array,
                                dict_values,
                                val_idx,
                                lhs_slot,
                            ),
                            slot => slot == lhs_slot,
                        }
                    };
                    if !equal {
                        equal_to_results.set_bit(idx, false);
                    }
                }
            }
        }
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        let dict = array.as_dictionary::<K>();
        let dict_keys = dict.keys();
        let dict_values = dict.values();

        // Reuses `val_hashes`/`val_to_inner` when this batch carries the same
        // values array as the last one, which is the common case downstream of
        // a repartition or filter. Previously this unconditionally re-hashed
        // and re-initialised both buffers, costing O(D) per batch.
        self.sync_value_cache(dict_values);

        self.group_to_inner.try_reserve(rows.len()).map_err(|e| {
            datafusion_common::DataFusionError::from(
                Box::new(e) as Box<dyn std::error::Error + Send + Sync>
            )
        })?;

        if dict_keys.null_count() == 0 {
            let raw_keys = dict_keys.values();
            for &row in rows {
                let val_idx = raw_keys[row].as_usize();
                if self.val_to_inner[val_idx] == usize::MAX {
                    // A non-null key can still point to a null value in the values array.
                    self.val_to_inner[val_idx] = if dict_values.is_null(val_idx) {
                        self.find_or_insert_null()?
                    } else {
                        self.find_or_insert_value(
                            dict_values,
                            val_idx,
                            self.val_hashes[val_idx],
                        )?
                    };
                }
                self.group_to_inner.push(self.val_to_inner[val_idx]);
            }
        } else {
            let raw_keys = dict_keys.values();
            let null_buf = dict_keys.nulls().unwrap();
            for &row in rows {
                let slot = if null_buf.is_null(row) {
                    self.find_or_insert_null()?
                } else {
                    let val_idx = raw_keys[row].as_usize();
                    if self.val_to_inner[val_idx] == usize::MAX {
                        self.val_to_inner[val_idx] = if dict_values.is_null(val_idx) {
                            self.find_or_insert_null()?
                        } else {
                            self.find_or_insert_value(
                                dict_values,
                                val_idx,
                                self.val_hashes[val_idx],
                            )?
                        };
                    }
                    self.val_to_inner[val_idx]
                };
                self.group_to_inner.push(slot);
            }
        }

        self.check_key_overflow()
    }

    fn len(&self) -> usize {
        self.group_to_inner.len()
    }

    fn size(&self) -> usize {
        self.inner.size()
            + self.value_dedup_size
            + self.group_to_inner.capacity() * size_of::<usize>()
            + self.val_to_inner.capacity() * size_of::<usize>()
            + self.val_hashes.capacity() * size_of::<u64>()
            + self
                .cached_values
                .as_ref()
                .map_or(0, |values| values.get_array_memory_size())
            + self.null_array.get_array_memory_size()
            + size_of::<Self>()
    }

    fn build(self: Box<Self>) -> ArrayRef {
        let null_inner_slot = self.null_inner_slot;
        let values = self.inner.build();
        Self::into_dict(values, &self.group_to_inner, null_inner_slot)
    }

    fn take_n(&mut self, n: usize) -> ArrayRef {
        let old_inner_len = self.inner.len();
        let all_inner_values = self.inner.take_n(old_inner_len);

        let mut emit_old_to_new = vec![usize::MAX; old_inner_len];
        let mut emit_new_to_old: Vec<usize> = Vec::new();
        for &old in &self.group_to_inner[..n] {
            // Null groups emit a null key (None) and need no slot in the
            // values array, so excluding them keeps key indices tight and
            // prevents overflow at key-type capacity.
            if all_inner_values.is_null(old) {
                continue;
            }
            if emit_old_to_new[old] == usize::MAX {
                emit_old_to_new[old] = emit_new_to_old.len();
                emit_new_to_old.push(old);
            }
        }
        let emit_indices =
            Int64Array::from_iter(emit_new_to_old.iter().map(|&i| i as i64));
        let compact_emit_values =
            take(&*all_inner_values, &emit_indices, None).expect("take emit values");
        let emitted_keys: PrimitiveArray<K> = self.group_to_inner[..n]
            .iter()
            .map(|&old| {
                if all_inner_values.is_null(old) {
                    None
                } else {
                    Some(K::Native::usize_as(emit_old_to_new[old]))
                }
            })
            .collect();
        let emitted: ArrayRef =
            Arc::new(DictionaryArray::<K>::new(emitted_keys, compact_emit_values));

        let remaining = self.group_to_inner[n..].to_vec();
        let mut old_to_new = vec![usize::MAX; old_inner_len];
        let mut new_to_old = Vec::new();
        for &old in &remaining {
            if old_to_new[old] == usize::MAX {
                old_to_new[old] = new_to_old.len();
                new_to_old.push(old);
            }
        }

        self.value_dedup = HashTable::new();
        self.value_dedup_size = 0;
        self.null_inner_slot = None;
        self.hash_values(&all_inner_values);

        for (new_slot, &old_slot) in new_to_old.iter().enumerate() {
            if all_inner_values.is_null(old_slot) {
                self.inner
                    .append_val(&self.null_array, 0)
                    .expect("append null failed in take_n");
                self.null_inner_slot = Some(new_slot);
            } else {
                self.inner
                    .append_val(&all_inner_values, old_slot)
                    .expect("append value failed in take_n");
                self.value_dedup.insert_accounted(
                    (self.val_hashes[old_slot], new_slot),
                    |&(entry_hash, _)| entry_hash,
                    &mut self.value_dedup_size,
                );
            }
        }

        self.group_to_inner = remaining.iter().map(|&old| old_to_new[old]).collect();
        self.check_key_overflow().expect("key overflow in take_n");

        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::group_values::multi_group_by::bytes::ByteGroupValueBuilder;
    use arrow::array::{
        Array, ArrayRef, BooleanBufferBuilder, DictionaryArray, Int32Array, StringArray,
    };
    use arrow::compute::cast;
    use arrow::datatypes::{DataType, Int8Type, Int32Type};
    use datafusion_physical_expr::binary_map::OutputType;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn col() -> DictionaryGroupValuesColumn<Int32Type> {
        let field = Field::new("", DataType::Utf8, true);
        DictionaryGroupValuesColumn::new(
            Box::new(ByteGroupValueBuilder::<i32>::new(OutputType::Utf8)),
            &field,
        )
    }

    struct CountingGroupColumn {
        inner: ByteGroupValueBuilder<i32>,
        comparisons: Arc<AtomicUsize>,
    }

    impl GroupColumn for CountingGroupColumn {
        fn equal_to(&self, lhs_row: usize, array: &ArrayRef, rhs_row: usize) -> bool {
            self.comparisons.fetch_add(1, Ordering::Relaxed);
            self.inner.equal_to(lhs_row, array, rhs_row)
        }

        fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
            self.inner.append_val(array, row)
        }

        fn vectorized_equal_to(
            &mut self,
            lhs_rows: &[usize],
            array: &ArrayRef,
            rhs_rows: &[usize],
            equal_to_results: &mut BooleanBufferBuilder,
        ) {
            self.inner
                .vectorized_equal_to(lhs_rows, array, rhs_rows, equal_to_results)
        }

        fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
            self.inner.vectorized_append(array, rows)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn size(&self) -> usize {
            self.inner.size()
        }

        fn build(self: Box<Self>) -> ArrayRef {
            Box::new(self.inner).build()
        }

        fn take_n(&mut self, n: usize) -> ArrayRef {
            self.inner.take_n(n)
        }
    }

    fn counting_col(
        comparisons: Arc<AtomicUsize>,
    ) -> DictionaryGroupValuesColumn<Int32Type> {
        let field = Field::new("", DataType::Utf8, true);
        DictionaryGroupValuesColumn::new(
            Box::new(CountingGroupColumn {
                inner: ByteGroupValueBuilder::<i32>::new(OutputType::Utf8),
                comparisons,
            }),
            &field,
        )
    }

    fn int8_col() -> DictionaryGroupValuesColumn<Int8Type> {
        let field = Field::new("", DataType::Utf8, true);
        DictionaryGroupValuesColumn::new(
            Box::new(ByteGroupValueBuilder::<i32>::new(OutputType::Utf8)),
            &field,
        )
    }

    fn i32_dict(keys: &[Option<i32>], values: &[Option<&str>]) -> ArrayRef {
        Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(keys.to_vec()),
            Arc::new(StringArray::from(values.to_vec())),
        ))
    }

    fn i8_dict(keys: &[Option<i8>], values: &[Option<&str>]) -> ArrayRef {
        use arrow::array::Int8Array;
        Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(keys.to_vec()),
            Arc::new(StringArray::from(values.to_vec())),
        ))
    }

    fn equal_results(values: &[bool]) -> BooleanBufferBuilder {
        let mut results = BooleanBufferBuilder::new(values.len());
        for &value in values {
            results.append(value);
        }
        results
    }

    fn plain_values(arr: &ArrayRef) -> Vec<Option<String>> {
        let plain = cast(arr.as_ref(), &DataType::Utf8).unwrap();
        plain
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|value| value.map(ToOwned::to_owned))
            .collect()
    }

    fn all_true(len: usize) -> BooleanBufferBuilder {
        let mut buf = BooleanBufferBuilder::new(len);
        buf.append_n(len, true);
        buf
    }

    fn bool_vec(buf: &BooleanBufferBuilder) -> Vec<bool> {
        (0..buf.len()).map(|i| buf.get_bit(i)).collect()
    }

    fn dict_with_values(keys: &[Option<i32>], values: &ArrayRef) -> ArrayRef {
        Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(keys.to_vec()),
            Arc::clone(values),
        ))
    }

    #[test]
    fn cached_values_are_retained_and_accounted_until_take_n() {
        fn append_with_unreferenced_value(
            value: &str,
        ) -> (
            DictionaryGroupValuesColumn<Int32Type>,
            std::sync::Weak<dyn Array>,
            usize,
        ) {
            let values: ArrayRef =
                Arc::new(StringArray::from(vec![Some("a"), Some(value)]));
            let weak_values = Arc::downgrade(&values);
            let values_size = values.get_array_memory_size();
            let input = dict_with_values(&[Some(0)], &values);
            let mut column = col();
            column.vectorized_append(&input, &[0]).unwrap();
            drop(input);
            drop(values);
            (column, weak_values, values_size)
        }

        let (short, short_weak, short_array_size) = append_with_unreferenced_value("b");
        let (mut long, long_weak, long_array_size) =
            append_with_unreferenced_value(&"x".repeat(4096));

        assert!(short_weak.upgrade().is_some());
        assert!(long_weak.upgrade().is_some());
        assert_eq!(
            long.size() - short.size(),
            long_array_size - short_array_size,
            "the retained values-array size must be included in builder accounting"
        );

        let size_before_take = long.size();
        let _emitted = long.take_n(1);
        assert!(long_weak.upgrade().is_none());
        assert!(long.size() < size_before_take);
        assert_eq!(short.inner.len(), 1);
    }

    #[test]
    fn dictionary_values_are_deduplicated_and_rebuilt() {
        let mut group_column = col();
        let input = i32_dict(
            &[Some(0), Some(1), Some(0), Some(1), Some(0)],
            &[Some("a"), Some("b")],
        );
        group_column
            .vectorized_append(&input, &[0, 1, 2, 3, 4])
            .unwrap();
        let output = Box::new(group_column).build();
        assert_eq!(output.data_type(), input.data_type());
        assert_eq!(output.as_dictionary::<Int32Type>().values().len(), 2);
        assert_eq!(
            plain_values(&output),
            vec![
                Some("a".into()),
                Some("b".into()),
                Some("a".into()),
                Some("b".into()),
                Some("a".into())
            ]
        );
    }

    #[test]
    fn null_keys_and_null_values_are_the_same_group() {
        let mut group_column = col();
        let input = i32_dict(&[None, Some(0), Some(1)], &[None, Some("b")]);
        group_column.vectorized_append(&input, &[0, 1, 2]).unwrap();
        assert!(group_column.equal_to(0, &input, 1));
        assert!(!group_column.equal_to(0, &input, 2));
        let output = Box::new(group_column).build();
        assert_eq!(output.as_dictionary::<Int32Type>().values().len(), 1);
        assert_eq!(plain_values(&output), vec![None, None, Some("b".into())]);
    }

    #[test]
    fn dictionary_encodings_can_change_between_batches() {
        let mut group_column = col();
        group_column
            .vectorized_append(&i32_dict(&[Some(0)], &[Some("a"), Some("b")]), &[0])
            .unwrap();
        let other_encoding = i32_dict(&[Some(1)], &[Some("z"), Some("a")]);
        let mut equals = equal_results(&[true]);
        group_column.vectorized_equal_to(&[0], &other_encoding, &[0], &mut equals);
        assert!(equals.get_bit(0));
    }

    #[test]
    fn value_cache_avoids_repeated_per_row_value_comparisons() {
        let comparisons = Arc::new(AtomicUsize::new(0));
        let mut group_column = counting_col(Arc::clone(&comparisons));
        let values: Vec<String> = (0..16).map(|i| format!("value-{i}")).collect();
        let refs: Vec<Option<&str>> =
            values.iter().map(|value| Some(value.as_str())).collect();
        let keys: Vec<Option<i32>> = (0..16).map(Some).collect();
        let input = i32_dict(&keys, &refs);
        group_column
            .vectorized_append(&input, &(0..16).collect::<Vec<_>>())
            .unwrap();

        // `vectorized_append` already resolved every value of this values
        // array, so checking it again must not compare any value a second
        // time.
        comparisons.store(0, Ordering::Relaxed);
        let mut sparse = equal_results(&[true]);
        group_column.vectorized_equal_to(&[0], &input, &[0], &mut sparse);
        assert!(sparse.get_bit(0));
        assert_eq!(comparisons.load(Ordering::Relaxed), 0);

        let dense_keys: Vec<Option<i32>> = (0..256).map(|i| Some(i % 16)).collect();
        let dense = i32_dict(&dense_keys, &refs);
        let lhs_rows: Vec<usize> = (0..256).map(|i| i % 16).collect();
        let rhs_rows: Vec<usize> = (0..256).collect();

        // A values array the cache has not seen is compared once per distinct
        // `val_idx`, not once per row: 16 comparisons for 256 rows.
        comparisons.store(0, Ordering::Relaxed);
        let mut dense_results = equal_results(&[true; 256]);
        group_column.vectorized_equal_to(
            &lhs_rows,
            &dense,
            &rhs_rows,
            &mut dense_results,
        );
        assert!((0..256).all(|i| dense_results.get_bit(i)));
        assert_eq!(comparisons.load(Ordering::Relaxed), 16);

        // Repeating the check against the now cached values array performs no
        // further value comparisons.
        comparisons.store(0, Ordering::Relaxed);
        let mut repeated_results = equal_results(&[true; 256]);
        group_column.vectorized_equal_to(
            &lhs_rows,
            &dense,
            &rhs_rows,
            &mut repeated_results,
        );
        assert!((0..256).all(|i| repeated_results.get_bit(i)));
        assert_eq!(comparisons.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn vectorized_equal_to_keeps_existing_false_bits() {
        let mut group_column = col();
        let input = i32_dict(&[Some(0), Some(1)], &[Some("a"), Some("b")]);
        group_column.vectorized_append(&input, &[0, 1]).unwrap();

        // The sparse branch must not turn a false comparison bit back on.
        let sparse = i32_dict(&[Some(0), Some(2)], &[Some("a"), Some("b"), Some("c")]);
        let mut equals = equal_results(&[false, true]);
        group_column.vectorized_equal_to(&[0, 1], &sparse, &[0, 1], &mut equals);
        assert!(!equals.get_bit(0));
        assert!(!equals.get_bit(1));

        // The dictionary-cardinality lookup branch obeys the same invariant.
        let mut equals = equal_results(&[false, true, true]);
        group_column.vectorized_equal_to(&[0, 0, 1], &input, &[0, 0, 1], &mut equals);
        assert!(!equals.get_bit(0));
        assert!(equals.get_bit(1));
        assert!(equals.get_bit(2));
    }

    #[test]
    fn take_n_compacts_values_and_remaps_remaining_keys() {
        let mut group_column = col();
        let first = i32_dict(
            &[Some(0), Some(1), None, Some(2)],
            &[Some("a"), Some("b"), Some("c")],
        );
        group_column
            .vectorized_append(&first, &[0, 1, 2, 3])
            .unwrap();
        let emitted = group_column.take_n(2);
        assert_eq!(emitted.as_dictionary::<Int32Type>().values().len(), 2);
        assert_eq!(
            plain_values(&emitted),
            vec![Some("a".into()), Some("b".into())]
        );

        let second = i32_dict(&[None, Some(0)], &[Some("z")]);
        group_column.vectorized_append(&second, &[0, 1]).unwrap();
        let mut equals = equal_results(&[true, true]);
        group_column.vectorized_equal_to(&[0, 1], &second, &[0, 1], &mut equals);
        assert!(equals.get_bit(0));
        assert!(!equals.get_bit(1));
        assert_eq!(
            plain_values(&Box::new(group_column).build()),
            vec![None, Some("c".into()), None, Some("z".into())]
        );
    }

    #[test]
    fn null_does_not_consume_an_int8_dictionary_key() {
        let mut group_column = int8_col();
        group_column
            .append_val(&i8_dict(&[None], &[Some("x")]), 0)
            .unwrap();
        let values: Vec<String> = (0..128).map(|index| format!("v{index}")).collect();
        let refs: Vec<Option<&str>> =
            values.iter().map(|value| Some(value.as_str())).collect();
        let keys: Vec<Option<i8>> = (0..128).map(|index| Some(index as i8)).collect();
        let input = i8_dict(&keys, &refs);
        group_column
            .vectorized_append(&input, &(0..128).collect::<Vec<_>>())
            .unwrap();
        assert!(
            group_column
                .append_val(&i8_dict(&[Some(0)], &[Some("overflow")]), 0)
                .is_err()
        );

        // The failing append intentionally leaves its partially inserted value
        // in the builder, so construct a fresh capacity-sized builder before
        // validating output construction.
        let mut group_column = int8_col();
        group_column
            .append_val(&i8_dict(&[None], &[Some("x")]), 0)
            .unwrap();
        group_column
            .vectorized_append(&input, &(0..128).collect::<Vec<_>>())
            .unwrap();
        let output = Box::new(group_column).build();
        assert!(output.as_dictionary::<Int8Type>().key(0).is_none());
        assert_eq!(output.as_dictionary::<Int8Type>().values().len(), 128);
    }

    /// A cache rebuild resets every `val_to_inner` entry to `usize::MAX`, so
    /// an entry resolved by an earlier batch surviving a later one proves the
    /// dictionary was not re-hashed in between.
    #[test]
    fn cache_survives_batches_sharing_a_values_array() {
        let values: ArrayRef =
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")]));
        let mut col = col();

        col.vectorized_append(&dict_with_values(&[Some(0)], &values), &[0])
            .unwrap();
        let slot_a = col.val_to_inner[0];
        assert_ne!(slot_a, usize::MAX);

        col.vectorized_append(&dict_with_values(&[Some(1)], &values), &[0])
            .unwrap();

        assert_eq!(col.val_to_inner[0], slot_a, "cache was rebuilt");
        assert_ne!(col.val_to_inner[1], usize::MAX);
        assert_eq!(
            col.val_to_inner[2],
            usize::MAX,
            "unreferenced value resolved"
        );
        assert!(
            col.cached_values
                .as_ref()
                .is_some_and(|c| Arc::ptr_eq(c, &values))
        );
        assert_eq!(col.inner.len(), 2);
    }

    /// `take` clones the values `Arc` and rewrites only the keys, which is the
    /// property the `ptr_eq` check relies on. Pin that arrow behaviour.
    #[test]
    fn take_preserves_values_arc_so_cache_hits() {
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b")]));
        let batch = dict_with_values(&[Some(0), Some(1), Some(0)], &values);
        let taken = take(&*batch, &Int32Array::from(vec![2, 0]), None).unwrap();
        assert!(
            Arc::ptr_eq(taken.as_dictionary::<Int32Type>().values(), &values),
            "take must not copy the values array"
        );

        let mut col = col();
        col.vectorized_append(&batch, &[0, 1, 2]).unwrap();
        let slot_b = col.val_to_inner[1];
        col.vectorized_append(&taken, &[0, 1]).unwrap();

        assert_eq!(col.val_to_inner[1], slot_b, "cache was rebuilt");
        assert_eq!(col.inner.len(), 2);
    }

    /// A genuinely different values array must rebuild the cache, and equal
    /// values across the two arrays must still dedup to one inner slot.
    #[test]
    fn distinct_values_arcs_rebuild_cache_but_still_dedup() {
        let mut col = col();
        let first = i32_dict(&[Some(0)], &[Some("a")]);
        let second = i32_dict(&[Some(0)], &[Some("a")]);
        col.vectorized_append(&first, &[0]).unwrap();
        col.vectorized_append(&second, &[0]).unwrap();

        assert!(
            col.cached_values.as_ref().is_some_and(|c| Arc::ptr_eq(
                c,
                second.as_dictionary::<Int32Type>().values()
            )),
            "cache should track the most recent values array"
        );
        assert_eq!(col.inner.len(), 1, "'a' must dedup across both arrays");
        assert_eq!(col.len(), 2);
    }

    /// `take_n` remaps inner slots, so the cached `val_idx → inner_slot` map
    /// must not survive it.
    #[test]
    fn take_n_invalidates_value_cache() {
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b")]));
        let mut col = col();
        col.vectorized_append(&dict_with_values(&[Some(0), Some(1)], &values), &[0, 1])
            .unwrap();

        let _emitted = col.take_n(1);
        assert!(
            col.cached_values.is_none(),
            "take_n remaps inner slots and must drop the cache"
        );

        col.vectorized_append(&dict_with_values(&[Some(0), Some(1)], &values), &[0, 1])
            .unwrap();

        let out = Box::new(col).build();
        let d = out.as_dictionary::<Int32Type>();
        let vals = d.values().as_string::<i32>();
        let got: Vec<&str> = d
            .keys()
            .iter()
            .map(|k| vals.value(k.unwrap() as usize))
            .collect();
        assert_eq!(got, vec!["b", "a", "b"]);
    }

    /// `append_val` fills `val_hashes` but not `val_to_inner`; mixing it with
    /// `vectorized_append` must not duplicate a value or reuse a stale slot.
    #[test]
    fn scalar_and_vectorized_append_share_cache_consistently() {
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b")]));
        let batch = dict_with_values(&[Some(0), Some(1)], &values);
        let mut col = col();

        col.append_val(&batch, 0).unwrap();
        assert!(
            col.cached_values
                .as_ref()
                .is_some_and(|c| Arc::ptr_eq(c, &values))
        );

        col.vectorized_append(&batch, &[0, 1]).unwrap();
        assert_eq!(col.inner.len(), 2, "'a' must not be duplicated");
        assert_eq!(col.len(), 3);
    }

    /// On a cache miss `vectorized_equal_to` compares against the group's own
    /// slot: an equal result proves the slot and is cached, an unequal one
    /// only rules that slot out and must leave the entry unresolved.
    #[test]
    fn vectorized_equal_to_caches_only_equal_results() {
        let mut col = col();
        col.vectorized_append(
            &i32_dict(&[Some(0), Some(1)], &[Some("a"), Some("b")]),
            &[0, 1],
        )
        .unwrap();
        let (group_a, group_b) = (0, 1);
        let slot_a = col.group_to_inner[group_a];
        let slot_b = col.group_to_inner[group_b];

        // A new values array in the opposite order, so `val_idx` != slot.
        // Two rows for two values keeps the per-row fallback out of the way.
        let (idx_b, idx_a) = (0, 1);
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("b"), Some("a")]));
        let a_then_b =
            dict_with_values(&[Some(idx_a as i32), Some(idx_b as i32)], &values);

        // "a" vs group a is equal, "b" vs group a is not.
        let mut buf = all_true(2);
        col.vectorized_equal_to(&[group_a, group_a], &a_then_b, &[0, 1], &mut buf);
        assert_eq!(bool_vec(&buf), vec![true, false]);
        assert_eq!(col.val_to_inner[idx_a], slot_a);
        assert_eq!(
            col.val_to_inner[idx_b],
            usize::MAX,
            "unequal result was cached"
        );

        // Later, "b" vs group b fills the entry that was left unresolved.
        let only_b = dict_with_values(&[Some(idx_b as i32)], &values);
        let mut buf = all_true(1);
        col.vectorized_equal_to(&[group_b], &only_b, &[0], &mut buf);
        assert_eq!(bool_vec(&buf), vec![true]);
        assert_eq!(col.val_to_inner[idx_b], slot_b);
    }

    /// A non-null key pointing at a null value and a null key must both
    /// compare equal to the null group, and only the former goes through
    /// `val_to_inner`.
    #[test]
    fn vectorized_equal_to_matches_null_value_and_null_key_to_null_group() {
        let mut col = col();
        col.vectorized_append(&i32_dict(&[None, Some(0)], &[Some("a")]), &[0, 1])
            .unwrap();
        let (null_group, group_a) = (0, 1);
        let null_slot = col.null_inner_slot.unwrap();

        let idx_null = 0;
        let values: ArrayRef = Arc::new(StringArray::from(vec![None, Some("a")]));
        let batch = dict_with_values(
            &[Some(idx_null as i32), Some(idx_null as i32), None],
            &values,
        );

        // null value vs null group, null value vs group a, null key vs null group.
        let mut buf = all_true(3);
        col.vectorized_equal_to(
            &[null_group, group_a, null_group],
            &batch,
            &[0, 1, 2],
            &mut buf,
        );
        assert_eq!(bool_vec(&buf), vec![true, false, true]);
        assert_eq!(col.val_to_inner[idx_null], null_slot);
    }

    /// Fewer rows than values in an uncached values array takes the per-row
    /// fallback, which must not rebuild the cache for that array.
    #[test]
    fn vectorized_equal_to_small_batch_of_new_values_skips_cache_rebuild() {
        let old_values: ArrayRef = Arc::new(StringArray::from(vec![Some("a")]));
        let mut col = col();
        col.vectorized_append(&dict_with_values(&[Some(0)], &old_values), &[0])
            .unwrap();
        let group_a = 0;

        // One row against three values in an array the cache has not seen.
        let new_values: ArrayRef =
            Arc::new(StringArray::from(vec![Some("x"), Some("a"), Some("y")]));
        let idx_a = 1;
        let only_a = dict_with_values(&[Some(idx_a)], &new_values);
        let mut buf = all_true(1);
        col.vectorized_equal_to(&[group_a], &only_a, &[0], &mut buf);
        assert_eq!(bool_vec(&buf), vec![true]);
        assert!(
            col.cached_values
                .as_ref()
                .is_some_and(|c| Arc::ptr_eq(c, &old_values)),
            "fallback must leave the cache on the previous values array"
        );
    }
}
