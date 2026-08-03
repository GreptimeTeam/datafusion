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

//! Simplifier for Physical Expressions

use arrow::datatypes::Schema;
use datafusion_common::{Result, tree_node::TreeNode};
use std::sync::Arc;

use crate::{
    PhysicalExpr,
    simplifier::{
        const_evaluator::create_dummy_batch, unwrap_cast::unwrap_cast_in_comparison,
    },
};

pub mod const_evaluator;
pub mod not;
pub mod unwrap_cast;

const MAX_LOOP_COUNT: usize = 5;

/// Simplifies physical expressions by applying various optimizations
///
/// This can be useful after adapting expressions from a table schema
/// to a file schema. For example, casts added to match the types may
/// potentially be unwrapped.
pub struct PhysicalExprSimplifier<'a> {
    schema: &'a Schema,
}

impl<'a> PhysicalExprSimplifier<'a> {
    /// Create a new physical expression simplifier
    pub fn new(schema: &'a Schema) -> Self {
        Self { schema }
    }

    /// Simplify a physical expression
    pub fn simplify(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let mut current_expr = expr;
        let mut count = 0;
        let schema = self.schema;

        let batch = create_dummy_batch()?;

        while count < MAX_LOOP_COUNT {
            count += 1;
            let result = current_expr.transform(|node| {
                #[cfg(debug_assertions)]
                let original_type = node.data_type(schema).unwrap();

                // Apply NOT expression simplification first, then unwrap cast optimization,
                // then constant expression evaluation
                #[expect(deprecated, reason = "`simplify_not_expr` is marked as deprecated until it's made private.")]
                let rewritten = not::simplify_not_expr(node, schema)?
                    .transform_data(|node| unwrap_cast_in_comparison(node, schema))?
                    .transform_data(|node| {
                        const_evaluator::simplify_const_expr_immediate(node, batch)
                    })?;

                #[cfg(debug_assertions)]
                assert_eq!(
                    rewritten.data.data_type(schema).unwrap(),
                    original_type,
                    "Simplified expression should have the same data type as the original"
                );

                Ok(rewritten)
            })?;

            if !result.transformed {
                return Ok(result.data);
            }
            current_expr = result.data;
        }
        Ok(current_expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{
        BinaryExpr, CastExpr, Literal, NotExpr, TryCastExpr, col, in_list, lit,
    };
    use arrow::{
        array::{
            Array, ArrayRef, BooleanArray, Date64Array, Decimal128Array, Int32Array,
            Int64Array, StringArray, StringDictionaryBuilder, TimestampMillisecondArray,
            TimestampNanosecondArray,
        },
        compute::CastOptions,
        datatypes::{DataType, Field, TimeUnit, UInt8Type},
        record_batch::RecordBatch,
    };
    use datafusion_common::{ScalarValue, format::DEFAULT_FORMAT_OPTIONS};
    use datafusion_expr::Operator;
    use std::collections::HashMap;

    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int64, false),
            Field::new("c3", DataType::Utf8, false),
        ])
    }

    fn not_test_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Boolean, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int32, false),
        ])
    }

    /// Helper function to extract a Literal from a PhysicalExpr
    fn as_literal(expr: &Arc<dyn PhysicalExpr>) -> &Literal {
        expr.downcast_ref::<Literal>()
            .unwrap_or_else(|| panic!("Expected Literal, got: {expr}"))
    }

    /// Helper function to extract a BinaryExpr from a PhysicalExpr
    fn as_binary(expr: &Arc<dyn PhysicalExpr>) -> &BinaryExpr {
        expr.downcast_ref::<BinaryExpr>()
            .unwrap_or_else(|| panic!("Expected BinaryExpr, got: {expr}"))
    }

    /// Assert that simplifying `input` produces `expected`
    fn assert_not_simplify(
        simplifier: &PhysicalExprSimplifier,
        input: Arc<dyn PhysicalExpr>,
        expected: Arc<dyn PhysicalExpr>,
    ) {
        let result = simplifier.simplify(Arc::clone(&input)).unwrap();
        assert_eq!(
            &result, &expected,
            "Simplification should transform:\n  input: {input}\n  to:    {expected}\n  got:   {result}"
        );
    }

    fn boolean_values(
        expr: &Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
    ) -> Result<Vec<Option<bool>>> {
        let values = expr.evaluate(batch)?.into_array(batch.num_rows())?;
        let values = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("comparison expression must evaluate to a BooleanArray");
        Ok((0..values.len())
            .map(|index| (!values.is_null(index)).then(|| values.value(index)))
            .collect())
    }

    /// Verifies that a cast rewrite preserves every row's Boolean or NULL result.
    fn assert_cast_predicate_equivalent(
        name: &str,
        schema: &Schema,
        batch: &RecordBatch,
        original: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        let simplified = PhysicalExprSimplifier::new(schema)
            .simplify(Arc::clone(&original))
            .unwrap();
        let original_values =
            boolean_values(&original, batch).map_err(|error| error.to_string());
        let simplified_values =
            boolean_values(&simplified, batch).map_err(|error| error.to_string());
        assert_eq!(
            original_values, simplified_values,
            "{name}: cast unwrapping changed predicate results\n  original: {original}\n  simplified: {simplified}"
        );
        simplified
    }

    fn assert_comparison_left_cast_removed(name: &str, expr: &Arc<dyn PhysicalExpr>) {
        let binary = as_binary(expr);
        assert!(
            binary.left().downcast_ref::<CastExpr>().is_none()
                && binary.left().downcast_ref::<TryCastExpr>().is_none(),
            "{name}: simplification must remove the comparison's left cast, got: {expr}"
        );
    }

    fn contains_cast(expr: &Arc<dyn PhysicalExpr>) -> bool {
        expr.downcast_ref::<CastExpr>().is_some()
            || expr.downcast_ref::<TryCastExpr>().is_some()
            || expr.children().into_iter().any(contains_cast)
    }

    #[test]
    fn narrowing_try_cast_equality_retains_cast_and_nulls() {
        let schema = Schema::new(vec![Field::new("i", DataType::Int64, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![
                Some(0),
                Some(i64::from(i32::MAX) + 1),
                None,
            ])) as ArrayRef],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(TryCastExpr::new(
                col("i", &schema).unwrap(),
                DataType::Int32,
            )),
            Operator::Eq,
            lit(0i32),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "Int64 -> Int32 equality",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn decimal_scale_narrowing_equality_retains_cast() {
        let schema =
            Schema::new(vec![Field::new("d", DataType::Decimal128(10, 3), true)]);
        let values = Decimal128Array::from(vec![Some(1_001), Some(1_000), None])
            .with_precision_and_scale(10, 3)
            .unwrap();
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)])
                .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("d", &schema).unwrap(),
                DataType::Decimal128(10, 2),
                None,
            )),
            Operator::Eq,
            lit(ScalarValue::Decimal128(Some(100), 10, 2)),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "decimal scale narrowing equality",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn decimal_scale_narrowing_literal_left_inequality_retains_cast() {
        let schema =
            Schema::new(vec![Field::new("d", DataType::Decimal128(10, 3), false)]);
        let values = Decimal128Array::from(vec![1_001])
            .with_precision_and_scale(10, 3)
            .unwrap();
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)])
                .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit(ScalarValue::Decimal128(Some(100), 10, 2)),
            Operator::Lt,
            Arc::new(CastExpr::new(
                col("d", &schema).unwrap(),
                DataType::Decimal128(10, 2),
                None,
            )),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "decimal scale narrowing literal-left strict inequality",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn decimal_scale_narrowing_distinctness_retains_cast() {
        let schema =
            Schema::new(vec![Field::new("d", DataType::Decimal128(10, 3), true)]);
        let values = Decimal128Array::from(vec![Some(1_001), None])
            .with_precision_and_scale(10, 3)
            .unwrap();
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)])
                .unwrap();

        let outcomes =
            [Operator::IsDistinctFrom, Operator::IsNotDistinctFrom].map(|op| {
                let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    Arc::new(CastExpr::new(
                        col("d", &schema).unwrap(),
                        DataType::Decimal128(10, 2),
                        None,
                    )),
                    op,
                    lit(ScalarValue::Decimal128(Some(100), 10, 2)),
                ));
                let simplified = PhysicalExprSimplifier::new(&schema)
                    .simplify(Arc::clone(&original))
                    .unwrap();
                (
                    boolean_values(&original, &batch).unwrap(),
                    boolean_values(&simplified, &batch).unwrap(),
                    contains_cast(&simplified),
                )
            });
        let [
            (distinct_original, distinct_simplified, distinct_retains_cast),
            (not_distinct_original, not_distinct_simplified, not_distinct_retains_cast),
        ] = outcomes;
        assert_eq!(
            (distinct_original, not_distinct_original),
            (distinct_simplified, not_distinct_simplified),
            "decimal scale narrowing changed nullable IS DISTINCT FROM and IS NOT DISTINCT FROM results"
        );
        assert!(distinct_retains_cast && not_distinct_retains_cast);
    }

    #[test]
    fn date64_to_int32_try_cast_retains_cast() {
        let schema = Schema::new(vec![Field::new("d", DataType::Date64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Date64Array::from(vec![2_147_483_648]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(TryCastExpr::new(
                col("d", &schema).unwrap(),
                DataType::Int32,
            )),
            Operator::Eq,
            lit(i32::MIN),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "Date64 -> Int32 equality",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn timestamp_narrowing_equality_uses_half_open_range() {
        let source_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampNanosecondArray::from(vec![
                -1_000_001, -1_000_000, -1, 0, 1, 999_999, 1_000_000, 1_000_001,
            ]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("ts", &schema).unwrap(),
                DataType::Timestamp(TimeUnit::Millisecond, None),
                None,
            )),
            Operator::Eq,
            lit(ScalarValue::TimestampMillisecond(Some(0), None)),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "timestamp ns -> ms buckets",
            &schema,
            &batch,
            original,
        );
        let range = as_binary(&simplified);
        assert_eq!(*range.op(), Operator::And);
        assert!(!contains_cast(&simplified));
    }

    #[test]
    fn timestamp_narrowing_literal_left_swaps_to_source_range() {
        let source_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampNanosecondArray::from(vec![
                -1, 0, 1_000_000,
            ]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit(ScalarValue::TimestampMillisecond(Some(-1), None)),
            Operator::Lt,
            Arc::new(CastExpr::new(
                col("ts", &schema).unwrap(),
                DataType::Timestamp(TimeUnit::Millisecond, None),
                None,
            )),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "timestamp narrowing literal-left strict inequality",
            &schema,
            &batch,
            original,
        );
        let comparison = as_binary(&simplified);
        assert_eq!(*comparison.op(), Operator::GtEq);
        assert_eq!(
            as_literal(comparison.right()).value(),
            &ScalarValue::TimestampNanosecond(Some(-999_999), None)
        );
        assert!(!contains_cast(&simplified));
    }

    #[test]
    fn timestamp_narrowing_distinctness_uses_null_aware_range() {
        let source_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampNanosecondArray::from(vec![
                Some(-1),
                Some(0),
                Some(1_000_000),
                None,
            ]))],
        )
        .unwrap();

        for op in [Operator::IsDistinctFrom, Operator::IsNotDistinctFrom] {
            let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::new(CastExpr::new(
                    col("ts", &schema).unwrap(),
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    None,
                )),
                op,
                lit(ScalarValue::TimestampMillisecond(Some(0), None)),
            ));
            let simplified = assert_cast_predicate_equivalent(
                "timestamp narrowing nullable distinctness",
                &schema,
                &batch,
                original,
            );
            assert!(!contains_cast(&simplified));
        }
    }

    #[test]
    fn timestamp_timezone_metadata_change_retains_cast() {
        let source_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let target_type =
            DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("+05:30")));
        let schema = Schema::new(vec![Field::new("ts", source_type, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampNanosecondArray::from(vec![0]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("ts", &schema).unwrap(),
                target_type,
                None,
            )),
            Operator::Eq,
            lit(ScalarValue::TimestampNanosecond(
                Some(0),
                Some(Arc::from("+05:30")),
            )),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "timestamp timezone metadata",
            &schema,
            &batch,
            original,
        );
        assert!(
            simplified
                .downcast_ref::<BinaryExpr>()
                .unwrap()
                .left()
                .downcast_ref::<CastExpr>()
                .is_some(),
            "timezone metadata cast must not be removed: {simplified}"
        );
    }

    #[test]
    fn timestamp_coarse_to_fine_cast_overflow_equality_retains_cast() {
        let source_type = DataType::Timestamp(TimeUnit::Millisecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampMillisecondArray::from(vec![Some(
                9_223_372_036_855,
            )]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("ts", &schema).unwrap(),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                None,
            )),
            Operator::Eq,
            lit(ScalarValue::TimestampNanosecond(Some(0), None)),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "timestamp coarse -> fine Cast overflow with zero literal",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn timestamp_coarse_to_fine_try_cast_overflow_equality_retains_cast() {
        let source_type = DataType::Timestamp(TimeUnit::Millisecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampMillisecondArray::from(vec![Some(
                9_223_372_036_855,
            )]))],
        )
        .unwrap();
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(TryCastExpr::new(
                col("ts", &schema).unwrap(),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            )),
            Operator::Eq,
            lit(ScalarValue::TimestampNanosecond(Some(0), None)),
        ));

        let simplified = assert_cast_predicate_equivalent(
            "timestamp coarse -> fine TryCast overflow with zero literal",
            &schema,
            &batch,
            original,
        );
        assert!(contains_cast(&simplified));
    }

    #[test]
    fn timestamp_coarse_to_fine_ordered_overflow_is_accepted_policy() {
        let source_type = DataType::Timestamp(TimeUnit::Millisecond, None);
        let schema = Schema::new(vec![Field::new("ts", source_type, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(TimestampMillisecondArray::from(vec![Some(
                9_223_372_036_855,
            )]))],
        )
        .unwrap();

        for try_cast in [false, true] {
            let cast_expr: Arc<dyn PhysicalExpr> = if try_cast {
                Arc::new(TryCastExpr::new(
                    col("ts", &schema).unwrap(),
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                ))
            } else {
                Arc::new(CastExpr::new(
                    col("ts", &schema).unwrap(),
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                    None,
                ))
            };
            let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                cast_expr,
                Operator::GtEq,
                lit(ScalarValue::TimestampNanosecond(Some(0), None)),
            ));
            let simplified = PhysicalExprSimplifier::new(&schema)
                .simplify(Arc::clone(&original))
                .unwrap();
            let comparison = as_binary(&simplified);
            assert_eq!(*comparison.op(), Operator::GtEq);
            assert_eq!(
                as_literal(comparison.right()).value(),
                &ScalarValue::TimestampMillisecond(Some(0), None),
            );
            assert!(!contains_cast(&simplified));

            if try_cast {
                assert_eq!(boolean_values(&original, &batch).unwrap(), vec![None]);
            } else {
                assert!(boolean_values(&original, &batch).is_err());
            }
            assert_eq!(
                boolean_values(&simplified, &batch).unwrap(),
                vec![Some(true)]
            );
        }
    }

    #[test]
    fn dictionary_encoding_overflow_retains_cast_and_error() {
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, false)]);
        let values =
            StringArray::from_iter_values((0..257).map(|index| format!("value-{index}")));
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)])
                .unwrap();
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("s", &schema).unwrap(),
                dictionary_type,
                None,
            )),
            Operator::Eq,
            lit("value-256"),
        ));

        let simplified = PhysicalExprSimplifier::new(&schema)
            .simplify(Arc::clone(&original))
            .unwrap();
        let original_error = boolean_values(&original, &batch)
            .expect_err(
                "257 distinct Utf8 values must overflow Dictionary(UInt8, Utf8) keys",
            )
            .to_string();
        let simplified_result = boolean_values(&simplified, &batch);
        let simplified_retains_cast = as_binary(&simplified)
            .left()
            .downcast_ref::<CastExpr>()
            .is_some();
        assert_eq!(
            (
                original_error.contains("Dictionary key bigger than the key type"),
                simplified_result.is_err(),
                simplified_retains_cast,
            ),
            (true, true, true),
            "dictionary encoding must preserve its key-overflow error and retain CastExpr\n  original: {original}\n  simplified: {simplified}"
        );
    }

    #[test]
    fn dictionary_identity_and_decode_controls_remain_equivalent() {
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let schema = Schema::new(vec![Field::new("d", dictionary_type.clone(), true)]);
        let mut builder = StringDictionaryBuilder::<UInt8Type>::new();
        builder.append("one").unwrap();
        builder.append("two").unwrap();
        builder.append_null();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(builder.finish())],
        )
        .unwrap();

        let identity: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("d", &schema).unwrap(),
                dictionary_type,
                None,
            )),
            Operator::Eq,
            lit("one"),
        ));
        let simplified = assert_cast_predicate_equivalent(
            "dictionary identity",
            &schema,
            &batch,
            identity,
        );
        assert_comparison_left_cast_removed("dictionary identity", &simplified);

        let decode: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("d", &schema).unwrap(),
                DataType::Utf8,
                None,
            )),
            Operator::Eq,
            lit("one"),
        ));
        let simplified = assert_cast_predicate_equivalent(
            "dictionary decode",
            &schema,
            &batch,
            decode,
        );
        assert_comparison_left_cast_removed("dictionary decode", &simplified);
    }

    #[test]
    fn nondefault_cast_options_and_target_fields_retain_cast() {
        let schema = Schema::new(vec![Field::new("i", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let target_type = DataType::Int64;
        let cases = [
            (
                "safe cast options",
                Arc::new(CastExpr::new(
                    col("i", &schema).unwrap(),
                    target_type.clone(),
                    Some(CastOptions {
                        safe: true,
                        format_options: DEFAULT_FORMAT_OPTIONS,
                    }),
                )) as Arc<dyn PhysicalExpr>,
            ),
            (
                "format cast options",
                Arc::new(CastExpr::new(
                    col("i", &schema).unwrap(),
                    target_type.clone(),
                    Some(CastOptions {
                        safe: false,
                        format_options: DEFAULT_FORMAT_OPTIONS.with_null("NULL"),
                    }),
                )) as Arc<dyn PhysicalExpr>,
            ),
            (
                "named target field",
                Arc::new(CastExpr::new_with_target_field(
                    col("i", &schema).unwrap(),
                    Arc::new(Field::new("named_target", target_type.clone(), true)),
                    None,
                )) as Arc<dyn PhysicalExpr>,
            ),
            (
                "nonnullable target field",
                Arc::new(CastExpr::new_with_target_field(
                    col("i", &schema).unwrap(),
                    Arc::new(Field::new("", target_type.clone(), false)),
                    None,
                )) as Arc<dyn PhysicalExpr>,
            ),
            (
                "metadata-bearing target field",
                Arc::new(CastExpr::new_with_target_field(
                    col("i", &schema).unwrap(),
                    Arc::new(Field::new("", target_type, true).with_metadata(
                        HashMap::from([("semantic".to_string(), "preserve".to_string())]),
                    )),
                    None,
                )) as Arc<dyn PhysicalExpr>,
            ),
        ];

        let retained = cases.map(|(name, cast)| {
            let original: Arc<dyn PhysicalExpr> =
                Arc::new(BinaryExpr::new(cast, Operator::Eq, lit(1i64)));
            let simplified =
                assert_cast_predicate_equivalent(name, &schema, &batch, original);
            simplified
                .downcast_ref::<BinaryExpr>()
                .unwrap()
                .left()
                .downcast_ref::<CastExpr>()
                .is_some()
        });
        assert_eq!(
            retained, [true; 5],
            "case order: safe options, format options, named field, nonnullable field, metadata-bearing field"
        );
    }

    #[test]
    fn default_cast_options_and_target_field_control_rewrites() {
        let schema = Schema::new(vec![Field::new("i", DataType::Int32, false)]);
        let simplifier = PhysicalExprSimplifier::new(&schema);
        let original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("i", &schema).unwrap(),
                DataType::Int64,
                None,
            )),
            Operator::Eq,
            lit(1i64),
        ));
        let simplified = simplifier.simplify(original).unwrap();
        assert!(
            simplified
                .downcast_ref::<BinaryExpr>()
                .unwrap()
                .left()
                .downcast_ref::<CastExpr>()
                .is_none(),
            "default CastExpr should remain eligible for unwrapping"
        );
    }

    #[test]
    fn widening_cast_controls_unwrap_safely() {
        let int_schema = Schema::new(vec![Field::new("i", DataType::Int32, true)]);
        let int_batch = RecordBatch::try_new(
            Arc::new(int_schema.clone()),
            vec![Arc::new(Int32Array::from(vec![Some(0), Some(1), None]))],
        )
        .unwrap();
        let int_original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("i", &int_schema).unwrap(),
                DataType::Int64,
                None,
            )),
            Operator::Gt,
            lit(0i64),
        ));
        let simplified = assert_cast_predicate_equivalent(
            "Int32 -> Int64 widening",
            &int_schema,
            &int_batch,
            int_original,
        );
        assert!(!contains_cast(&simplified));

        let decimal_schema =
            Schema::new(vec![Field::new("d", DataType::Decimal128(10, 2), false)]);
        let values = Decimal128Array::from(vec![101])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let decimal_batch = RecordBatch::try_new(
            Arc::new(decimal_schema.clone()),
            vec![Arc::new(values)],
        )
        .unwrap();
        let decimal_original: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                col("d", &decimal_schema).unwrap(),
                DataType::Decimal128(20, 4),
                None,
            )),
            Operator::Eq,
            lit(ScalarValue::Decimal128(Some(10_100), 20, 4)),
        ));
        let simplified = assert_cast_predicate_equivalent(
            "decimal scale widening",
            &decimal_schema,
            &decimal_batch,
            decimal_original,
        );
        assert!(!contains_cast(&simplified));
    }

    #[test]
    fn test_simplify() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create: cast(c2 as INT32) != INT32(99)
        let column_expr = col("c2", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int32, None));
        let literal_expr = lit(ScalarValue::Int32(Some(99)));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::NotEq, literal_expr));

        // Apply full simplification (uses TreeNodeRewriter)
        let optimized = simplifier.simplify(binary_expr).unwrap();

        let optimized_binary = as_binary(&optimized);

        // Int64 -> Int32 is narrowing, so it must remain cast-bearing.
        let left_expr = optimized_binary.left();
        assert!(
            left_expr.downcast_ref::<CastExpr>().is_some()
                || left_expr.downcast_ref::<TryCastExpr>().is_some()
        );
    }

    #[test]
    fn test_nested_expression_simplification() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create nested expression: (cast(c1 as INT64) > INT64(5)) OR (cast(c2 as INT32) <= INT32(10))
        let c1_expr = col("c1", &schema).unwrap();
        let c1_cast = Arc::new(CastExpr::new(c1_expr, DataType::Int64, None));
        let c1_literal = lit(ScalarValue::Int64(Some(5)));
        let c1_binary = Arc::new(BinaryExpr::new(c1_cast, Operator::Gt, c1_literal));

        let c2_expr = col("c2", &schema).unwrap();
        let c2_cast = Arc::new(CastExpr::new(c2_expr, DataType::Int32, None));
        let c2_literal = lit(ScalarValue::Int32(Some(10)));
        let c2_binary = Arc::new(BinaryExpr::new(c2_cast, Operator::LtEq, c2_literal));

        let or_expr = Arc::new(BinaryExpr::new(c1_binary, Operator::Or, c2_binary));

        // Apply simplification
        let optimized = simplifier.simplify(or_expr).unwrap();

        let or_binary = as_binary(&optimized);

        // Verify left side: c1 > INT32(5)
        let left_binary = as_binary(or_binary.left());
        let left_left_expr = left_binary.left();
        assert!(
            left_left_expr.downcast_ref::<CastExpr>().is_none()
                && left_left_expr.downcast_ref::<TryCastExpr>().is_none()
        );
        let left_literal = as_literal(left_binary.right());
        assert_eq!(left_literal.value(), &ScalarValue::Int32(Some(5)));

        // Verify right side remains cast-bearing: Int64 -> Int32 narrows.
        let right_binary = as_binary(or_binary.right());
        let right_left_expr = right_binary.left();
        assert!(
            right_left_expr.downcast_ref::<CastExpr>().is_some()
                || right_left_expr.downcast_ref::<TryCastExpr>().is_some()
        );
    }

    #[test]
    fn test_double_negation_elimination() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(c > 5)) -> c > 5
        let inner_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Gt,
            lit(ScalarValue::Int32(Some(5))),
        ));
        let inner_not = Arc::new(NotExpr::new(Arc::clone(&inner_expr)));
        let double_not: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(inner_not));

        let expected = inner_expr;
        assert_not_simplify(&simplifier, double_not, expected);
        Ok(())
    }

    #[test]
    fn test_not_literal() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(TRUE) -> FALSE
        let not_true = Arc::new(NotExpr::new(lit(ScalarValue::Boolean(Some(true)))));
        let expected = lit(ScalarValue::Boolean(Some(false)));
        assert_not_simplify(&simplifier, not_true, expected);

        // NOT(FALSE) -> TRUE
        let not_false = Arc::new(NotExpr::new(lit(ScalarValue::Boolean(Some(false)))));
        let expected = lit(ScalarValue::Boolean(Some(true)));
        assert_not_simplify(&simplifier, not_false, expected);

        Ok(())
    }

    #[test]
    fn test_negate_comparison() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c = 5) -> c != 5
        let not_eq = Arc::new(NotExpr::new(Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(5))),
        ))));
        let expected = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::NotEq,
            lit(ScalarValue::Int32(Some(5))),
        ));
        assert_not_simplify(&simplifier, not_eq, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_law_and() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(a AND b) -> NOT a OR NOT b
        let and_expr = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::And,
            col("b", &schema)?,
        ));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(col("a", &schema)?)),
            Operator::Or,
            Arc::new(NotExpr::new(col("b", &schema)?)),
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_law_or() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(a OR b) -> NOT a AND NOT b
        let or_expr = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::Or,
            col("b", &schema)?,
        ));
        let not_or: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(or_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(col("a", &schema)?)),
            Operator::And,
            Arc::new(NotExpr::new(col("b", &schema)?)),
        ));
        assert_not_simplify(&simplifier, not_or, expected);

        Ok(())
    }

    #[test]
    fn test_demorgans_with_comparison_simplification() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c = 1 AND c = 2) -> c != 1 OR c != 2
        let eq1 = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(1))),
        ));
        let eq2 = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Eq,
            lit(ScalarValue::Int32(Some(2))),
        ));
        let and_expr = Arc::new(BinaryExpr::new(eq1, Operator::And, eq2));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                col("c", &schema)?,
                Operator::NotEq,
                lit(ScalarValue::Int32(Some(1))),
            )),
            Operator::Or,
            Arc::new(BinaryExpr::new(
                col("c", &schema)?,
                Operator::NotEq,
                lit(ScalarValue::Int32(Some(2))),
            )),
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_not_of_not_and_not() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(a) AND NOT(b)) -> a OR b
        let not_a = Arc::new(NotExpr::new(col("a", &schema)?));
        let not_b = Arc::new(NotExpr::new(col("b", &schema)?));
        let and_expr = Arc::new(BinaryExpr::new(not_a, Operator::And, not_b));
        let not_and: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(and_expr));

        let expected: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("a", &schema)?,
            Operator::Or,
            col("b", &schema)?,
        ));
        assert_not_simplify(&simplifier, not_and, expected);

        Ok(())
    }

    #[test]
    fn test_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c IN (1, 2, 3)) -> c NOT IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let in_list_expr = in_list(col("c", &schema)?, list.clone(), &false, &schema)?;
        let not_in: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(in_list_expr));

        let expected = in_list(col("c", &schema)?, list, &true, &schema)?;
        assert_not_simplify(&simplifier, not_in, expected);

        Ok(())
    }

    #[test]
    fn test_not_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(c NOT IN (1, 2, 3)) -> c IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let not_in_list_expr = in_list(col("c", &schema)?, list.clone(), &true, &schema)?;
        let not_not_in: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(not_in_list_expr));

        let expected = in_list(col("c", &schema)?, list, &false, &schema)?;
        assert_not_simplify(&simplifier, not_not_in, expected);

        Ok(())
    }

    #[test]
    fn test_double_not_in_list() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // NOT(NOT(c IN (1, 2, 3))) -> c IN (1, 2, 3)
        let list = vec![
            lit(ScalarValue::Int32(Some(1))),
            lit(ScalarValue::Int32(Some(2))),
            lit(ScalarValue::Int32(Some(3))),
        ];
        let in_list_expr = in_list(col("c", &schema)?, list.clone(), &false, &schema)?;
        let not_in = Arc::new(NotExpr::new(in_list_expr));
        let double_not: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(not_in));

        let expected = in_list(col("c", &schema)?, list, &false, &schema)?;
        assert_not_simplify(&simplifier, double_not, expected);

        Ok(())
    }

    #[test]
    fn test_deeply_nested_not() -> Result<()> {
        let schema = not_test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // Create a deeply nested NOT expression: NOT(NOT(NOT(...NOT(c > 5)...)))
        // This tests that we don't get stack overflow with many nested NOTs.
        // With recursive_protection enabled (default), this should work by
        // automatically growing the stack as needed.
        let inner_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c", &schema)?,
            Operator::Gt,
            lit(ScalarValue::Int32(Some(5))),
        ));

        let mut expr = Arc::clone(&inner_expr);
        // Create 200 layers of NOT to test deep recursion handling
        for _ in 0..200 {
            expr = Arc::new(NotExpr::new(expr));
        }

        // With 200 NOTs (even number), should simplify back to the original expression
        let expected = inner_expr;
        assert_not_simplify(&simplifier, Arc::clone(&expr), expected);

        // Manually dismantle the deep input expression to avoid Stack Overflow on Drop
        // If we just let `expr` go out of scope, Rust's recursive Drop will blow the stack
        // even with recursive_protection, because Drop doesn't use the #[recursive] attribute.
        // We peel off layers one by one to avoid deep recursion in Drop.
        while let Some(not_expr) = expr.downcast_ref::<NotExpr>() {
            // Clone the child (Arc increment).
            // Now child has 2 refs: one in parent, one in `child`.
            let child = Arc::clone(not_expr.arg());

            // Reassign `expr` to `child`.
            // This drops the old `expr` (Parent).
            // Parent refcount -> 0, Parent is dropped.
            // Parent drops its reference to Child.
            // Child refcount decrements 2 -> 1.
            // Child is NOT dropped recursively because we still hold it in `expr`
            expr = child;
        }

        Ok(())
    }

    #[test]
    fn test_simplify_literal_binary_expr() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 1 + 2 -> 3
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(3)));
    }

    #[test]
    fn test_simplify_literal_comparison() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 5 > 3 -> true
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(5i32), Operator::Gt, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Boolean(Some(true)));

        // 2 > 3 -> false
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(2i32), Operator::Gt, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Boolean(Some(false)));
    }

    #[test]
    fn test_simplify_nested_literal_expr() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // (1 + 2) * 3 -> 9
        let inner: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(inner, Operator::Multiply, lit(3i32)));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(9)));
    }

    #[test]
    fn test_simplify_deeply_nested_literals() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // ((1 + 2) * 3) + ((4 - 1) * 2) -> 9 + 6 -> 15
        let left: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32))),
            Operator::Multiply,
            lit(3i32),
        ));
        let right: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(lit(4i32), Operator::Minus, lit(1i32))),
            Operator::Multiply,
            lit(2i32),
        ));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(left, Operator::Plus, right));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(literal.value(), &ScalarValue::Int32(Some(15)));
    }

    #[test]
    fn test_no_simplify_with_column() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // c1 + 2 should NOT be simplified (has column reference)
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("c1", &schema).unwrap(),
            Operator::Plus,
            lit(2i32),
        ));
        let result = simplifier.simplify(expr).unwrap();
        // Should remain a BinaryExpr, not become a Literal
        assert!(result.downcast_ref::<BinaryExpr>().is_some());
    }

    #[test]
    fn test_partial_simplify_with_column() {
        let schema = test_schema();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // (1 + 2) + c1 should simplify the literal part: 3 + c1
        let literal_part: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lit(1i32), Operator::Plus, lit(2i32)));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            literal_part,
            Operator::Plus,
            col("c1", &schema).unwrap(),
        ));
        let result = simplifier.simplify(expr).unwrap();

        // Should be a BinaryExpr with a Literal(3) on the left
        let binary = as_binary(&result);
        let left_literal = as_literal(binary.left());
        assert_eq!(left_literal.value(), &ScalarValue::Int32(Some(3)));
    }

    /// Regression test for https://github.com/apache/datafusion/issues/22367.
    ///
    /// A leaf `PhysicalExpr` that is neither a `Literal` nor a `Column`
    /// (nor volatile) must not be const-folded: it has no children to
    /// derive constness from, and evaluating it against the dummy batch
    /// produces a value unrelated to its real runtime semantics. Without
    /// the zero-children guard, `all(empty)` would vacuously hold and the
    /// node would be replaced with whatever scalar fell out of the dummy
    /// evaluation. Verify the node is left untouched.
    #[test]
    fn test_no_simplify_opaque_leaf_expr() {
        use arrow::array::ArrayRef;
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use datafusion_expr_common::columnar_value::ColumnarValue;
        use datafusion_physical_expr_common::physical_expr::PhysicalExpr as PhysicalExprTrait;
        use std::fmt;

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        struct OpaqueLeaf;

        impl fmt::Display for OpaqueLeaf {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "OpaqueLeaf")
            }
        }

        impl PhysicalExprTrait for OpaqueLeaf {
            fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
                Ok(DataType::Int32)
            }
            fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
                Ok(true)
            }
            fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
                // Simulate the broken FFI Column path: when handed a dummy
                // batch, return whatever scalar happens to materialize. If
                // the simplifier ever reaches this branch for a leaf node,
                // the predicate has already been silently corrupted.
                let arr: ArrayRef = Arc::new(Int32Array::from(vec![0; batch.num_rows()]));
                Ok(ColumnarValue::Array(arr))
            }
            fn children(&self) -> Vec<&Arc<dyn PhysicalExprTrait>> {
                vec![]
            }
            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn PhysicalExprTrait>>,
            ) -> Result<Arc<dyn PhysicalExprTrait>> {
                Ok(self)
            }
            fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "OpaqueLeaf")
            }
        }

        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        let opaque: Arc<dyn PhysicalExpr> = Arc::new(OpaqueLeaf);
        let result = simplifier.simplify(Arc::clone(&opaque)).unwrap();

        assert!(
            result.downcast_ref::<Literal>().is_none(),
            "opaque leaf must not be rewritten to a Literal, got: {result}"
        );
        assert_eq!(&result, &opaque);
    }

    #[test]
    fn test_simplify_literal_string_concat() {
        let schema = Schema::empty();
        let simplifier = PhysicalExprSimplifier::new(&schema);

        // 'hello' || ' world' -> 'hello world'
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lit("hello"),
            Operator::StringConcat,
            lit(" world"),
        ));
        let result = simplifier.simplify(expr).unwrap();
        let literal = as_literal(&result);
        assert_eq!(
            literal.value(),
            &ScalarValue::Utf8(Some("hello world".to_string()))
        );
    }
}
