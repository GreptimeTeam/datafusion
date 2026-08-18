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
use ahash::RandomState;
use arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, Int64Array, PrimitiveArray,
};
use arrow::compute::take;
use arrow::datatypes::{ArrowDictionaryKeyType, ArrowNativeType, DataType, Field};
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
    /// Reusable scratch buffer mapping `val_idx → inner_slot` across batches.
    val_to_inner: Vec<usize>,
    /// Reusable hash buffer for the dictionary values array.
    val_hashes: Vec<u64>,
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
        self.val_hashes.clear();
        self.val_hashes.resize(values.len(), 0);
        create_hashes(
            std::slice::from_ref(values),
            &self.random_state,
            &mut self.val_hashes,
        )
        .unwrap();
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

    fn find_or_insert_null(&mut self) -> Result<usize> {
        if let Some(slot) = self.null_inner_slot {
            return Ok(slot);
        }
        let slot = self.inner.len();
        self.inner.append_val(&self.null_array, 0)?;
        self.null_inner_slot = Some(slot);
        Ok(slot)
    }

    fn build_lookup_table(
        &self,
        dict_values: &ArrayRef,
        val_hashes: &[u64],
    ) -> Vec<usize> {
        let num_distinct = dict_values.len();
        let mut table = vec![usize::MAX; num_distinct + 1];
        let inner = &*self.inner;
        for val_idx in 0..num_distinct {
            if dict_values.is_null(val_idx) {
                table[val_idx] = self.null_inner_slot.unwrap_or(usize::MAX);
            } else {
                let hash = val_hashes[val_idx];
                if let Some(&(_, slot)) =
                    self.value_dedup.find(hash, |&(entry_hash, slot)| {
                        entry_hash == hash && inner.equal_to(slot, dict_values, val_idx)
                    })
                {
                    table[val_idx] = slot;
                }
            }
        }
        table[num_distinct] = self.null_inner_slot.unwrap_or(usize::MAX);
        table
    }

    /// Per-row fallback for `vectorized_equal_to` used when the number of rows
    /// to check is smaller than the dictionary cardinality, making the O(D)
    /// lookup-table build more expensive than direct value comparison.
    ///
    /// `#[cold]` + `#[inline(never)]` keeps this code out of the hot
    /// lookup-table loops in `vectorized_equal_to` so LLVM can pipeline them.
    #[cold]
    #[inline(never)]
    fn equal_to_per_row(
        &self,
        lhs_rows: &[usize],
        dict_values: &ArrayRef,
        dict: &DictionaryArray<K>,
        rhs_rows: &[usize],
        equal_to_results: &mut [bool],
    ) {
        let group_to_inner = self.group_to_inner.as_slice();
        for (idx, (&lhs_row, &rhs_row)) in
            lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
        {
            if !equal_to_results[idx] {
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
                equal_to_results[idx] = false;
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
                let single = dict_values.slice(val_idx, 1);
                self.hash_values(&single);
                self.find_or_insert_value(dict_values, val_idx, self.val_hashes[0])?
            }
        };
        self.group_to_inner.push(inner_slot);
        self.check_key_overflow()
    }

    fn vectorized_equal_to(
        &self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut [bool],
    ) {
        let dict = array.as_dictionary::<K>();
        let dict_keys = dict.keys();
        let dict_values = dict.values();
        let num_distinct = dict_values.len();

        // The fallback is in a separate #[cold] function so its code does not
        // appear inline here and cannot prevent LLVM from pipelining / unrolling
        // the hot lookup-table loops below.
        if rhs_rows.len() < num_distinct {
            self.equal_to_per_row(
                lhs_rows,
                dict_values,
                dict,
                rhs_rows,
                equal_to_results,
            );
            return;
        }

        let mut val_hashes = vec![0u64; dict_values.len()];
        create_hashes(
            std::slice::from_ref(dict_values),
            &self.random_state,
            &mut val_hashes,
        )
        .unwrap();
        let lookup = self.build_lookup_table(dict_values, &val_hashes);

        let group_to_inner = self.group_to_inner.as_slice();

        if dict_keys.null_count() == 0 {
            // No null keys : skip the get_bit guard: we only ever write false,
            // so overwriting an already-false bit is a no-op.
            let raw_keys = dict_keys.values();
            for (idx, (&lhs_row, &rhs_row)) in
                lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
            {
                let rhs_slot = lookup[raw_keys[rhs_row].as_usize()];
                if rhs_slot == usize::MAX || group_to_inner[lhs_row] != rhs_slot {
                    equal_to_results[idx] = false;
                }
            }
        } else {
            let null_buf = dict_keys.nulls().unwrap();
            let raw_keys = dict_keys.values();
            for (idx, (&lhs_row, &rhs_row)) in
                lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
            {
                if equal_to_results[idx] {
                    let val_idx = if null_buf.is_null(rhs_row) {
                        num_distinct
                    } else {
                        raw_keys[rhs_row].as_usize()
                    };
                    let rhs_slot = lookup[val_idx];
                    if rhs_slot == usize::MAX || group_to_inner[lhs_row] != rhs_slot {
                        equal_to_results[idx] = false;
                    }
                }
            }
        }
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        let dict = array.as_dictionary::<K>();
        let dict_keys = dict.keys();
        let dict_values = dict.values();
        let num_distinct = dict_values.len();

        self.hash_values(dict_values);
        self.val_to_inner.clear();
        self.val_to_inner.resize(num_distinct, usize::MAX);

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

        // Null deferred to last so null_inner_slot is always the highest index
        // and check_key_overflow can subtract it without a false overflow.
        let remaining = self.group_to_inner[n..].to_vec();
        let mut old_to_new = vec![usize::MAX; old_inner_len];
        let mut new_to_old: Vec<usize> = Vec::new();
        let mut null_old_slot: Option<usize> = None;
        for &old in &remaining {
            if all_inner_values.is_null(old) {
                if null_old_slot.is_none() {
                    null_old_slot = Some(old);
                }
                continue;
            }
            if old_to_new[old] == usize::MAX {
                old_to_new[old] = new_to_old.len();
                new_to_old.push(old);
            }
        }
        if let Some(old) = null_old_slot {
            old_to_new[old] = new_to_old.len();
            new_to_old.push(old);
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
    use arrow::array::{Array, ArrayRef, DictionaryArray, Int32Array, StringArray};
    use arrow::compute::cast;
    use arrow::datatypes::{DataType, Int8Type, Int32Type};
    use datafusion_physical_expr::binary_map::OutputType;
    use std::sync::Arc;

    fn col() -> DictionaryGroupValuesColumn<Int32Type> {
        let field = Field::new("", DataType::Utf8, true);
        DictionaryGroupValuesColumn::new(
            Box::new(ByteGroupValueBuilder::<i32>::new(OutputType::Utf8)),
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
        let mut equals = vec![true];
        group_column.vectorized_equal_to(&[0], &other_encoding, &[0], &mut equals);
        assert_eq!(equals, vec![true]);
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
        let mut equals = vec![true, true];
        group_column.vectorized_equal_to(&[0, 1], &second, &[0, 1], &mut equals);
        assert_eq!(equals, vec![true, false]);
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
}
