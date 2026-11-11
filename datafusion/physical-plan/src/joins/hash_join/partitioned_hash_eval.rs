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

//! Hash computation and hash table lookup expressions for dynamic filtering

use std::{fmt::Display, hash::Hash, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, UInt64Array},
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::hash_utils::{create_hashes, with_hashes};
#[cfg(feature = "proto")]
use datafusion_common::internal_err;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr_common::physical_expr::{
    DynHash, PhysicalExpr, PhysicalExprRef,
};

use crate::joins::Map;
use crate::joins::key_range_bitmap::KeyRangeBitmap;

/// RandomState wrapper that preserves the seed used to create it.
///
/// This is needed because `RandomState` doesn't expose its seed after creation,
/// but we need them for serialization (e.g., protobuf serde).
#[derive(Clone, Debug)]
pub struct SeededRandomState {
    random_state: RandomState,
    seed: u64,
}

impl SeededRandomState {
    /// Create a new SeededRandomState with the given seed.
    pub const fn with_seed(k: u64) -> Self {
        Self {
            random_state: RandomState::with_seed(k),
            seed: k,
        }
    }

    /// Get the inner RandomState.
    pub fn random_state(&self) -> &RandomState {
        &self.random_state
    }

    /// Get the seed used to create this RandomState.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Physical expression that computes hash values for a set of columns
///
/// This expression computes the hash of join key columns using a specific RandomState.
/// It returns a UInt64Array containing the hash values.
///
/// This is used for:
/// - Computing routing hashes (with RepartitionExec's 0,0,0,0 seeds)
/// - Computing lookup hashes (with HashJoin's 'J','O','I','N' seeds)
pub struct HashExpr {
    /// Columns to hash
    on_columns: Vec<PhysicalExprRef>,
    /// Random state for hashing (with seeds preserved for serialization)
    random_state: SeededRandomState,
    /// Description for display
    description: String,
}

impl HashExpr {
    /// Create a new HashExpr
    ///
    /// # Arguments
    /// * `on_columns` - Columns to hash
    /// * `random_state` - SeededRandomState for hashing
    /// * `description` - Description for debugging (e.g., "hash_repartition", "hash_join")
    pub fn new(
        on_columns: Vec<PhysicalExprRef>,
        random_state: SeededRandomState,
        description: String,
    ) -> Self {
        Self {
            on_columns,
            random_state,
            description,
        }
    }

    /// Get the columns being hashed.
    pub fn on_columns(&self) -> &[PhysicalExprRef] {
        &self.on_columns
    }

    /// Get the seed used for hashing.
    pub fn seed(&self) -> u64 {
        self.random_state.seed()
    }

    /// Get the description.
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl std::fmt::Debug for HashExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cols = self
            .on_columns
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let seed = self.seed();
        write!(f, "{}({cols}, [{seed}])", self.description)
    }
}

impl Hash for HashExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.on_columns.dyn_hash(state);
        self.description.hash(state);
        self.seed().hash(state);
    }
}

impl PartialEq for HashExpr {
    fn eq(&self, other: &Self) -> bool {
        self.on_columns == other.on_columns
            && self.description == other.description
            && self.seed() == other.seed()
    }
}

impl Eq for HashExpr {}

impl Display for HashExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl PhysicalExpr for HashExpr {
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.on_columns.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(HashExpr::new(
            children,
            self.random_state.clone(),
            self.description.clone(),
        )))
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = batch.num_rows();

        // Evaluate columns
        let keys_values = evaluate_columns(&self.on_columns, batch)?;

        // Compute hashes
        let mut hashes_buffer = vec![0; num_rows];
        create_hashes(
            &keys_values,
            self.random_state.random_state(),
            &mut hashes_buffer,
        )?;

        Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(
            hashes_buffer,
        ))))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }

    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        ctx: &datafusion_physical_expr_common::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalExprNode>> {
        use datafusion_proto_models::protobuf;
        let on_columns = ctx.encode_children_expressions(&self.on_columns)?;
        Ok(Some(protobuf::PhysicalExprNode {
            expr_id: None,
            expr_type: Some(protobuf::physical_expr_node::ExprType::HashExpr(
                protobuf::PhysicalHashExprNode {
                    on_columns,
                    seed0: self.seed(),
                    description: self.description.clone(),
                },
            )),
        }))
    }
}

#[cfg(feature = "proto")]
impl HashExpr {
    /// Reconstruct a [`HashExpr`] from its protobuf representation.
    ///
    /// Takes the whole [`PhysicalExprNode`], the exact inverse of what
    /// [`PhysicalExpr::try_to_proto`] produces, so every expression's
    /// `try_from_proto` shares one signature. Child sub-expressions are
    /// decoded recursively via [`PhysicalExprDecodeCtx::decode`].
    ///
    /// [`PhysicalExprNode`]: datafusion_proto_models::protobuf::PhysicalExprNode
    /// [`PhysicalExpr::try_to_proto`]: datafusion_physical_expr_common::physical_expr::PhysicalExpr::try_to_proto
    /// [`PhysicalExprDecodeCtx::decode`]: datafusion_physical_expr_common::physical_expr::proto_decode::PhysicalExprDecodeCtx::decode
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalExprNode,
        ctx: &datafusion_physical_expr_common::physical_expr::proto_decode::PhysicalExprDecodeCtx<'_>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        use datafusion_proto_models::protobuf;
        let hash_expr = match &node.expr_type {
            Some(protobuf::physical_expr_node::ExprType::HashExpr(h)) => h,
            _ => return internal_err!("PhysicalExprNode is not a HashExpr"),
        };
        let on_columns = ctx.decode_children_expressions(&hash_expr.on_columns)?;
        Ok(Arc::new(HashExpr::new(
            on_columns,
            SeededRandomState::with_seed(hash_expr.seed0),
            hash_expr.description.clone(),
        )))
    }
}

/// Physical expression that checks join keys in a [`Map`] (hash table or array map).
///
/// Returns a [`BooleanArray`](arrow::array::BooleanArray) indicating if join keys (from `on_columns`) exist in the map.
// TODO: rename to MapLookupExpr
pub struct HashTableLookupExpr {
    /// Columns in the ON clause used to compute the join key for lookups
    on_columns: Vec<PhysicalExprRef>,
    /// Random state for hashing (with seeds preserved for serialization)
    random_state: SeededRandomState,
    /// Map to check against (hash table or array map)
    map: Arc<Map>,
    /// Retain the build reservation as long as the published map is retained.
    build_lease: Option<Arc<super::exec::JoinBuildData>>,
    /// Description for display
    description: String,
    /// Shared pruning-only view of the build side's key range
    pruning_bitmap: Option<Arc<KeyRangeBitmap>>,
}
impl HashTableLookupExpr {
    /// Create a new HashTableLookupExpr
    ///
    /// # Arguments
    /// * `on_columns` - Columns in the ON clause used to compute the join key
    /// * `random_state` - SeededRandomState for hashing
    /// * `map` - Map to check membership (hash table or array map)
    /// * `description` - Description for debugging
    /// * `pruning_bitmap` - key-range summary for pruning only, or `None`
    /// # Note
    /// This is public for internal testing purposes only and is not
    /// guaranteed to be stable across versions.
    pub fn new(
        on_columns: Vec<PhysicalExprRef>,
        random_state: SeededRandomState,
        map: Arc<Map>,
        description: String,
        pruning_bitmap: Option<Arc<KeyRangeBitmap>>,
    ) -> Self {
        Self {
            on_columns,
            random_state,
            map,
            build_lease: None,
            description,
            pruning_bitmap,
        }
    }

    pub(super) fn with_build_lease(
        mut self,
        lease: Option<Arc<super::exec::JoinBuildData>>,
    ) -> Self {
        self.build_lease = lease;
        self
    }

    /// Which parts of this build side's key range hold keys, for container pruning.
    pub fn pruning_bitmap(&self) -> Option<&Arc<KeyRangeBitmap>> {
        self.pruning_bitmap.as_ref()
    }

    /// Read-only view of the build-side membership this lookup checks rows against.
    ///
    /// Returns `None` unless the membership can be exposed exactly as a bounded,
    /// single-column UTF-8 value list:
    ///
    /// * the lookup must carry a real build lease. Lookups built with
    ///   [`HashTableLookupExpr::new`] (tests, deserialized plans) have no build
    ///   data, so their membership is deliberately unavailable;
    /// * the join key must be exactly one column, and the lease must hold exactly
    ///   one evaluated key array for it;
    /// * that array must be non-dictionary `Utf8`. `Utf8View`, `LargeUtf8`,
    ///   dictionary encodings, numeric, decimal, float and every other type are
    ///   rejected: the membership is meant for predicates that must agree with
    ///   join equality, and e.g. signed zero or dictionary key domains would
    ///   need a different representation;
    /// * the array must be non-empty, NULL-free (an `IN` list cannot express
    ///   `NULL` equality, so a NULL key would turn into a narrower predicate),
    ///   and within the caller's `max_values` / `max_bytes` bounds.
    ///
    /// This is a read-only accessor for consumers that re-encode the membership
    /// predicate elsewhere; it does not change how the lookup evaluates rows.
    ///
    /// # Ownership and lifetime
    ///
    /// The returned references borrow from the immutable
    /// [`JoinBuildData`](super::exec::JoinBuildData) that the expression's lease
    /// keeps alive. Published build data is never mutated, the lease holds the
    /// build-side memory reservation (and therefore the underlying buffers) for
    /// as long as the published map is reachable, and `with_new_children`
    /// propagates the same lease to derived expressions. Every borrow is
    /// therefore valid for the lifetime of `&self` and observes one fixed build
    /// side: the same arrays the lookup's map was built from.
    pub fn build_key_membership(
        &self,
        max_values: usize,
        max_bytes: usize,
    ) -> Option<LookupBuildKey<'_>> {
        let lease = self.build_lease.as_ref()?;
        // Only a single-column key can be written as `column IN (...)`.
        let [key] = self.on_columns.as_slice() else {
            return None;
        };
        let [values] = lease.values.as_slice() else {
            return None;
        };
        if values.data_type() != &DataType::Utf8 {
            return None;
        }
        // An empty build has no membership, and NULL build keys match nothing
        // via `IN` yet may match via `NULL`-equal joins: fail open instead.
        if values.is_empty() || values.null_count() != 0 {
            return None;
        }
        if values.len() > max_values || values.get_array_memory_size() > max_bytes {
            return None;
        }
        Some(LookupBuildKey { key, values })
    }
}

/// Bounded, read-only build-side membership of a [`HashTableLookupExpr`].
///
/// Both references borrow from the build data the lookup's lease keeps alive;
/// see [`HashTableLookupExpr::build_key_membership`] for the exact contract.
#[derive(Debug)]
pub struct LookupBuildKey<'a> {
    /// Probe-side key expression the membership applies to.
    key: &'a PhysicalExprRef,
    /// Build-side key values, one entry per build row.
    values: &'a ArrayRef,
}

impl<'a> LookupBuildKey<'a> {
    /// The single probe-side key expression of the lookup.
    pub fn key(&self) -> &'a PhysicalExprRef {
        self.key
    }

    /// The build-side key values, in build order and without duplicates removed.
    pub fn values(&self) -> &'a ArrayRef {
        self.values
    }
}
impl std::fmt::Debug for HashTableLookupExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cols = self
            .on_columns
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let seed = self.random_state.seed();
        write!(f, "{}({cols}, [{seed}])", self.description)
    }
}

impl Hash for HashTableLookupExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.on_columns.dyn_hash(state);
        self.description.hash(state);
        self.random_state.seed().hash(state);
        // Hash immutable map identity, consistently with pointer equality below.
        Arc::as_ptr(&self.map).hash(state);
    }
}

impl PartialEq for HashTableLookupExpr {
    fn eq(&self, other: &Self) -> bool {
        // Sharing an immutable map implies equal contents, including across
        // prepared-build consumers. Distinct maps need no expensive deep comparison.
        self.on_columns == other.on_columns
            && self.description == other.description
            && self.random_state.seed() == other.random_state.seed()
            && Arc::ptr_eq(&self.map, &other.map)
    }
}

impl Eq for HashTableLookupExpr {}

impl Display for HashTableLookupExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl PhysicalExpr for HashTableLookupExpr {
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.on_columns.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(HashTableLookupExpr {
            on_columns: children,
            random_state: self.random_state.clone(),
            map: Arc::clone(&self.map),
            build_lease: self.build_lease.clone(),
            description: self.description.clone(),
            pruning_bitmap: self.pruning_bitmap.clone(),
        }))
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        // Evaluate columns
        let join_keys = evaluate_columns(&self.on_columns, batch)?;

        match self.map.as_ref() {
            Map::HashMap(map) => {
                with_hashes(&join_keys, self.random_state.random_state(), |hashes| {
                    let array = map.contain_hashes(hashes);
                    Ok(ColumnarValue::Array(Arc::new(array)))
                })
            }
            Map::ArrayMap(map) => {
                let array = map.contain_keys(&join_keys)?;
                Ok(ColumnarValue::Array(Arc::new(array)))
            }
        }
    }
    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        _ctx: &datafusion_physical_expr_common::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalExprNode>> {
        use datafusion_proto_models::protobuf;
        use datafusion_proto_models::protobuf::physical_expr_node::ExprType;

        // HashTableLookupExpr holds a runtime Arc<Map> (the build-side hash
        // table) that cannot be serialized, so it is replaced with lit(true).
        //
        // Dynamic filtering is a performance optimisation only — replacing the
        // lookup with lit(true) preserves correctness by allowing all rows
        // through.
        //
        // If a plan is serialized before execution, HashTableLookupExpr is not
        // yet present in the dynamic filter expression.
        //
        // If a plan is serialized after execution, any runtime-created
        // HashTableLookupExpr is replaced during serialization. Re-executing
        // the plan requires reset_state(), after which HashJoinExec rebuilds
        // fresh dynamic filters at runtime.
        let value = datafusion_proto_common::ScalarValue {
            value: Some(datafusion_proto_common::scalar_value::Value::BoolValue(
                true,
            )),
        };
        Ok(Some(protobuf::PhysicalExprNode {
            expr_id: None,
            expr_type: Some(ExprType::Literal(value)),
        }))
    }
    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}

fn evaluate_columns(
    columns: &[PhysicalExprRef],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    let num_rows = batch.num_rows();
    columns
        .iter()
        .map(|c| c.evaluate(batch)?.into_array(num_rows))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::hash_join::exec::JoinBuildData;
    use crate::joins::join_hash_map::JoinHashMapU32;
    use datafusion_common::ScalarValue;
    use datafusion_physical_expr::expressions::Column;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    fn compute_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// The pruning bitmap must survive `with_new_children`, which a scan calls
    /// once per file.
    #[test]
    fn test_pruning_bitmap_shared_with_derived_children() {
        let keys: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![0, 1_000_000]));
        let bitmap = Arc::new(
            KeyRangeBitmap::try_new(
                &keys,
                &ScalarValue::Int64(Some(0)),
                &ScalarValue::Int64(Some(1_000_000)),
                2,
            )
            .expect("two keys a million apart leave prunable gaps"),
        );
        let expr = Arc::new(HashTableLookupExpr::new(
            vec![Arc::new(Column::new("a", 0))],
            SeededRandomState::with_seed(1),
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10)))),
            "hash_lookup".to_string(),
            Some(Arc::clone(&bitmap)),
        ));

        let derived = expr
            .with_new_children(vec![Arc::new(Column::new("a", 7))])
            .unwrap();
        let derived = derived.downcast_ref::<HashTableLookupExpr>().unwrap();
        assert_eq!(derived.children()[0].to_string(), "a@7");
        assert!(Arc::ptr_eq(&bitmap, derived.pruning_bitmap().unwrap()));
    }

    #[test]
    fn test_hash_expr_eq_same() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let col_b: PhysicalExprRef = Arc::new(Column::new("b", 1));

        let expr1 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        let expr2 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        assert_eq!(expr1, expr2);
    }

    #[test]
    fn test_hash_expr_eq_different_columns() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let col_b: PhysicalExprRef = Arc::new(Column::new("b", 1));
        let col_c: PhysicalExprRef = Arc::new(Column::new("c", 2));

        let expr1 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        let expr2 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_c)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_expr_eq_different_description() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));

        let expr1 = HashExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            "hash_one".to_string(),
        );

        let expr2 = HashExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            "hash_two".to_string(),
        );

        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_expr_eq_different_seeds() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));

        let expr1 = HashExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        let expr2 = HashExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(5),
            "test_hash".to_string(),
        );

        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_expr_hash_consistency() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let col_b: PhysicalExprRef = Arc::new(Column::new("b", 1));

        let expr1 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        let expr2 = HashExpr::new(
            vec![Arc::clone(&col_a), Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            "test_hash".to_string(),
        );

        // Equal expressions should have equal hashes
        assert_eq!(expr1, expr2);
        assert_eq!(compute_hash(&expr1), compute_hash(&expr2));
    }

    #[cfg(feature = "proto")]
    mod proto_tests {
        use super::*;
        use arrow::datatypes::{DataType, Field};
        use datafusion_common::internal_datafusion_err;
        use datafusion_physical_expr_common::physical_expr::proto_decode::{
            PhysicalExprDecode, PhysicalExprDecodeCtx,
        };
        use datafusion_physical_expr_common::physical_expr::proto_encode::{
            PhysicalExprEncode, PhysicalExprEncodeCtx,
        };
        use datafusion_proto_models::protobuf;

        struct TestEncoder;

        impl PhysicalExprEncode for TestEncoder {
            fn encode(
                &self,
                expr: &Arc<dyn PhysicalExpr>,
            ) -> Result<protobuf::PhysicalExprNode> {
                let ctx = PhysicalExprEncodeCtx::new(self);
                expr.try_to_proto(&ctx)?.ok_or_else(|| {
                    internal_datafusion_err!("test encoder cannot encode {expr:?}")
                })
            }
        }

        struct TestDecoder;

        impl PhysicalExprDecode for TestDecoder {
            fn decode(
                &self,
                node: &protobuf::PhysicalExprNode,
                schema: &Schema,
            ) -> Result<Arc<dyn PhysicalExpr>> {
                let ctx = PhysicalExprDecodeCtx::new(schema, self);
                match &node.expr_type {
                    Some(protobuf::physical_expr_node::ExprType::Column(_)) => {
                        Column::try_from_proto(node, &ctx)
                    }
                    _ => internal_err!("test decoder cannot decode {node:?}"),
                }
            }
        }

        fn test_decode_ctx<'a>(
            schema: &'a Schema,
            decoder: &'a TestDecoder,
        ) -> PhysicalExprDecodeCtx<'a> {
            PhysicalExprDecodeCtx::new(schema, decoder)
        }

        #[test]
        fn hash_expr_try_to_proto() {
            let expr = HashExpr::new(
                vec![Arc::new(Column::new("a", 0)), Arc::new(Column::new("b", 1))],
                SeededRandomState::with_seed(42),
                "hash_join".to_string(),
            );
            let encoder = TestEncoder;
            let ctx = PhysicalExprEncodeCtx::new(&encoder);

            let proto = expr.try_to_proto(&ctx).unwrap().unwrap();

            assert_eq!(proto.expr_id, None);
            let hash_expr = match proto.expr_type.unwrap() {
                protobuf::physical_expr_node::ExprType::HashExpr(hash_expr) => hash_expr,
                other => panic!("expected HashExpr, got {other:?}"),
            };
            assert_eq!(hash_expr.seed0, 42);
            assert_eq!(hash_expr.description, "hash_join");
            assert_eq!(hash_expr.on_columns.len(), 2);
            assert!(
                hash_expr
                    .on_columns
                    .iter()
                    .all(|expr| expr.expr_id.is_none())
            );
        }

        #[test]
        fn hash_expr_try_from_proto() {
            let schema = Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Utf8, true),
            ]);
            let decoder = TestDecoder;
            let ctx = test_decode_ctx(&schema, &decoder);
            let proto = protobuf::PhysicalExprNode {
                expr_id: None,
                expr_type: Some(protobuf::physical_expr_node::ExprType::HashExpr(
                    protobuf::PhysicalHashExprNode {
                        on_columns: vec![
                            protobuf::PhysicalExprNode {
                                expr_id: None,
                                expr_type: Some(
                                    protobuf::physical_expr_node::ExprType::Column(
                                        protobuf::PhysicalColumn {
                                            name: "a".to_string(),
                                            index: 0,
                                        },
                                    ),
                                ),
                            },
                            protobuf::PhysicalExprNode {
                                expr_id: None,
                                expr_type: Some(
                                    protobuf::physical_expr_node::ExprType::Column(
                                        protobuf::PhysicalColumn {
                                            name: "b".to_string(),
                                            index: 1,
                                        },
                                    ),
                                ),
                            },
                        ],
                        seed0: 42,
                        description: "hash_join".to_string(),
                    },
                )),
            };

            let expr = HashExpr::try_from_proto(&proto, &ctx).unwrap();
            let expr = expr.downcast_ref::<HashExpr>().unwrap();

            assert_eq!(expr.seed(), 42);
            assert_eq!(expr.description(), "hash_join");
            assert_eq!(expr.on_columns().len(), 2);
            assert_eq!(
                expr.on_columns()[0]
                    .downcast_ref::<Column>()
                    .map(|col| (col.name(), col.index())),
                Some(("a", 0))
            );
            assert_eq!(
                expr.on_columns()[1]
                    .downcast_ref::<Column>()
                    .map(|col| (col.name(), col.index())),
                Some(("b", 1))
            );
        }

        #[test]
        fn hash_expr_try_from_proto_rejects_wrong_node_type() {
            let schema = Schema::empty();
            let decoder = TestDecoder;
            let ctx = test_decode_ctx(&schema, &decoder);
            let proto = protobuf::PhysicalExprNode {
                expr_id: None,
                expr_type: Some(protobuf::physical_expr_node::ExprType::Column(
                    protobuf::PhysicalColumn {
                        name: "a".to_string(),
                        index: 0,
                    },
                )),
            };

            let err = HashExpr::try_from_proto(&proto, &ctx).unwrap_err();
            assert!(
                err.to_string()
                    .contains("PhysicalExprNode is not a HashExpr"),
                "{err}"
            );
        }
    }

    #[test]
    fn test_hash_table_lookup_expr_eq_same() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let hash_map =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));

        let expr1 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        let expr2 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        assert_eq!(expr1, expr2);
    }

    #[test]
    fn test_hash_table_lookup_expr_eq_different_columns() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let col_b: PhysicalExprRef = Arc::new(Column::new("b", 1));

        let hash_map =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));

        let expr1 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        let expr2 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_b)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_table_lookup_expr_eq_different_description() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let hash_map =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));

        let expr1 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup_one".to_string(),
            None,
        );

        let expr2 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup_two".to_string(),
            None,
        );

        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_table_lookup_expr_eq_different_hash_map() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));

        // Two different Arc pointers (even with same content) should not be equal
        let hash_map1 =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));
        let hash_map2 =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));
        let expr1 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            hash_map1,
            "lookup".to_string(),
            None,
        );

        let expr2 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            hash_map2,
            "lookup".to_string(),
            None,
        );

        // Different Arc pointers means not equal (uses Arc::ptr_eq)
        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_hash_table_lookup_expr_hash_consistency() {
        let col_a: PhysicalExprRef = Arc::new(Column::new("a", 0));
        let hash_map =
            Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(10))));

        let expr1 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        let expr2 = HashTableLookupExpr::new(
            vec![Arc::clone(&col_a)],
            SeededRandomState::with_seed(1),
            Arc::clone(&hash_map),
            "lookup".to_string(),
            None,
        );

        // Equal expressions should have equal hashes
        assert_eq!(expr1, expr2);
        assert_eq!(compute_hash(&expr1), compute_hash(&expr2));
    }

    /// Builds the smallest real build lease: an immutable `JoinBuildData` holding
    /// the evaluated build key values, exactly like `collect_build_side` publishes.
    fn test_build_lease(
        values: Vec<ArrayRef>,
        columns: Vec<ArrayRef>,
    ) -> Arc<JoinBuildData> {
        use datafusion_execution::memory_pool::{
            GreedyMemoryPool, MemoryConsumer, MemoryPool,
        };

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let reservation =
            MemoryConsumer::new("build key membership test").register(&pool);
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .enumerate()
                .map(|(index, column)| {
                    arrow::datatypes::Field::new(
                        format!("k{index}"),
                        column.data_type().clone(),
                        true,
                    )
                })
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema, columns).unwrap();
        Arc::new(JoinBuildData {
            map: Arc::new(Map::HashMap(Box::new(JoinHashMapU32::with_capacity(
                values[0].len(),
            )))),
            batch,
            values,
            bounds: None,
            reservation,
        })
    }

    fn utf8_values(values: &[&str]) -> ArrayRef {
        Arc::new(arrow::array::StringArray::from(values.to_vec()))
    }

    fn lookup_with_lease(
        on_columns: Vec<PhysicalExprRef>,
        lease: Arc<JoinBuildData>,
    ) -> HashTableLookupExpr {
        HashTableLookupExpr::new(
            on_columns,
            SeededRandomState::with_seed(1),
            Arc::clone(&lease.map),
            "lookup".to_string(),
            None,
        )
        .with_build_lease(Some(lease))
    }

    #[test]
    fn build_key_membership_exposes_utf8_build_values_of_a_real_lease() {
        let values = utf8_values(&["trace-a", "trace-b"]);
        let lease =
            test_build_lease(vec![Arc::clone(&values)], vec![Arc::clone(&values)]);
        let key: PhysicalExprRef = Arc::new(Column::new("trace_id", 0));
        let lookup = lookup_with_lease(vec![Arc::clone(&key)], lease);

        let membership = lookup.build_key_membership(16, 4096).unwrap();
        assert_eq!(membership.key().to_string(), key.to_string());
        assert_eq!(
            membership
                .values()
                .as_any()
                .downcast_ref::<arrow::array::StringArray>(),
            values.as_any().downcast_ref::<arrow::array::StringArray>()
        );

        // The borrow is tied to the expression, so it stays valid while the
        // expression (and therefore the lease) is alive.
        let kept = Arc::new(lookup);
        assert_eq!(
            kept.build_key_membership(16, 4096).unwrap().values().len(),
            2
        );
    }

    #[test]
    fn build_key_membership_is_none_without_a_build_lease() {
        let values = utf8_values(&["trace-a"]);
        let lease =
            test_build_lease(vec![Arc::clone(&values)], vec![Arc::clone(&values)]);
        let lookup = HashTableLookupExpr::new(
            vec![Arc::new(Column::new("trace_id", 0))],
            SeededRandomState::with_seed(1),
            Arc::clone(&lease.map),
            "lookup".to_string(),
            None,
        );

        assert!(lookup.build_key_membership(16, 4096).is_none());
    }

    #[test]
    fn build_key_membership_is_none_for_composite_keys() {
        let values = utf8_values(&["trace-a"]);
        let lease =
            test_build_lease(vec![Arc::clone(&values)], vec![Arc::clone(&values)]);
        let lookup = lookup_with_lease(
            vec![
                Arc::new(Column::new("trace_id", 0)),
                Arc::new(Column::new("host", 1)),
            ],
            lease,
        );

        assert!(lookup.build_key_membership(16, 4096).is_none());
    }

    #[test]
    fn build_key_membership_is_none_for_unsupported_key_types() {
        // Numeric keys, `Utf8View` and dictionary keys are all out of scope: an
        // exact `IN` list would need a different value domain.
        let int_values: ArrayRef = Arc::new(arrow::array::Int32Array::from(vec![1, 2]));
        let dictionary_values: ArrayRef = Arc::new(arrow::array::DictionaryArray::<
            arrow::datatypes::Int32Type,
        >::from_iter([
            "trace-a", "trace-b",
        ]));
        let view_values: ArrayRef = Arc::new(arrow::array::StringViewArray::from(vec![
            "trace-a", "trace-b",
        ]));

        for values in [int_values, dictionary_values, view_values] {
            let lease =
                test_build_lease(vec![Arc::clone(&values)], vec![Arc::clone(&values)]);
            let lookup =
                lookup_with_lease(vec![Arc::new(Column::new("trace_id", 0))], lease);
            assert!(
                lookup.build_key_membership(16, 4096).is_none(),
                "unexpected membership for {:?}",
                lookup.children()[0].data_type(&Schema::empty()).unwrap()
            );
        }
    }

    #[test]
    fn build_key_membership_is_none_for_empty_null_or_oversized_values() {
        // Empty build side.
        let empty = utf8_values(&[]);
        let empty_lease =
            test_build_lease(vec![Arc::clone(&empty)], vec![Arc::clone(&empty)]);
        assert!(
            lookup_with_lease(vec![Arc::new(Column::new("trace_id", 0))], empty_lease)
                .build_key_membership(16, 4096)
                .is_none()
        );

        // NULL build keys cannot be expressed as `IN` membership.
        let with_null: ArrayRef =
            Arc::new(arrow::array::StringArray::from(vec![Some("trace-a"), None]));
        let null_lease =
            test_build_lease(vec![Arc::clone(&with_null)], vec![Arc::clone(&with_null)]);
        assert!(
            lookup_with_lease(vec![Arc::new(Column::new("trace_id", 0))], null_lease)
                .build_key_membership(16, 4096)
                .is_none()
        );

        // Value count and byte bounds are enforced by the caller.
        let values = utf8_values(&["trace-a", "trace-b"]);
        let lease =
            test_build_lease(vec![Arc::clone(&values)], vec![Arc::clone(&values)]);
        let lookup = lookup_with_lease(vec![Arc::new(Column::new("trace_id", 0))], lease);
        assert!(lookup.build_key_membership(1, 4096).is_none());
        assert!(lookup.build_key_membership(16, 8).is_none());
        assert!(lookup.build_key_membership(16, 4096).is_some());
    }
}
