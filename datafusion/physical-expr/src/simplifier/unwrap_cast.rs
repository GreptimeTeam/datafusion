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

//! Unwrap casts in binary comparisons for physical expressions
//!
//! This module provides optimization for physical expressions similar to the logical
//! optimizer's unwrap_cast module. It attempts to remove casts from comparisons to
//! literals by applying the casts to the literals if possible.
//!
//! The optimization improves performance by:
//! 1. Reducing runtime cast operations on column data
//! 2. Enabling better predicate pushdown opportunities
//! 3. Optimizing filter expressions in physical plans
//!
//! # Example
//!
//! Physical expression: `cast(column as INT64) > INT64(10)`
//! Optimized to: `column > INT32(10)` (assuming column is INT32)

use std::sync::Arc;

use arrow::compute::CastOptions;
use arrow::datatypes::{DataType, Schema};
use datafusion_common::datatype::DataTypeExt;
use datafusion_common::format::DEFAULT_FORMAT_OPTIONS;
use datafusion_common::{Result, ScalarValue, tree_node::Transformed};
use datafusion_expr::Operator;
use datafusion_expr_common::casts::{CastPredicatePreimage, cast_predicate_preimage};

use crate::PhysicalExpr;
use crate::expressions::{
    BinaryExpr, CastExpr, Literal, TryCastExpr, is_not_null, is_null, lit,
};

const DEFAULT_CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: DEFAULT_FORMAT_OPTIONS,
};

/// Attempts to unwrap casts in comparison expressions.
pub(crate) fn unwrap_cast_in_comparison(
    expr: Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>()
        && let Some(unwrapped) = try_unwrap_cast_binary(binary, schema)?
    {
        return Ok(Transformed::yes(unwrapped));
    }
    Ok(Transformed::no(expr))
}

/// Try to unwrap casts in binary expressions
fn try_unwrap_cast_binary(
    binary: &BinaryExpr,
    schema: &Schema,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    // Case 1: cast(left_expr) op literal
    if let (Some((inner_expr, cast_type)), Some(literal)) = (
        extract_cast_info(binary.left()),
        binary.right().downcast_ref::<Literal>(),
    ) && is_supported_comparison_operator(*binary.op())
        && let Some(unwrapped) = try_unwrap_cast_comparison(
            Arc::clone(inner_expr),
            cast_type,
            literal.value(),
            *binary.op(),
            schema,
        )?
    {
        return Ok(Some(unwrapped));
    }

    // Case 2: literal op cast(right_expr)
    if let (Some(literal), Some((inner_expr, cast_type))) = (
        binary.left().downcast_ref::<Literal>(),
        extract_cast_info(binary.right()),
    ) {
        // For literal op cast(expr), we need to swap the operator
        if let Some(swapped_op) = binary.op().swap()
            && is_supported_comparison_operator(*binary.op())
            && let Some(unwrapped) = try_unwrap_cast_comparison(
                Arc::clone(inner_expr),
                cast_type,
                literal.value(),
                swapped_op,
                schema,
            )?
        {
            return Ok(Some(unwrapped));
        }
        // If the operator cannot be swapped, we skip this optimization case
        // but don't prevent other optimizations
    }

    Ok(None)
}

/// This rewrite has a deliberately closed operator contract. In particular,
/// `supports_propagation()` also admits operators (such as regex matching) for
/// which a cast preimage is not defined.
fn is_supported_comparison_operator(op: Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
}

/// Extract cast information from a physical expression
///
/// If the expression is a CAST(expr, datatype) or TRY_CAST(expr, datatype),
/// returns Some((inner_expr, target_datatype)). Otherwise returns None.
fn extract_cast_info(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<(&Arc<dyn PhysicalExpr>, &DataType)> {
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        // CastExpr can carry execution options and field semantics that are
        // observable even though its data type is unchanged. Only the legacy,
        // type-only default cast is safe to remove.
        (cast.cast_options() == &DEFAULT_CAST_OPTIONS
            && cast.target_field() == &cast.cast_type().clone().into_nullable_field_ref())
            .then_some((cast.expr(), cast.cast_type()))
    } else if let Some(try_cast) = expr.downcast_ref::<TryCastExpr>() {
        Some((try_cast.expr(), try_cast.cast_type()))
    } else {
        None
    }
}

/// Try to unwrap a cast in comparison by moving the cast to the literal
fn try_unwrap_cast_comparison(
    inner_expr: Arc<dyn PhysicalExpr>,
    cast_type: &DataType,
    literal_value: &ScalarValue,
    op: Operator,
    schema: &Schema,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    // Get the data type of the inner expression
    let inner_type = inner_expr.data_type(schema)?;

    match cast_predicate_preimage(&inner_type, cast_type, op, literal_value)? {
        Some(CastPredicatePreimage::Exact(casted_literal)) => {
            Ok(Some(binary(inner_expr, op, lit(casted_literal))))
        }
        Some(CastPredicatePreimage::Range(interval)) => {
            Ok(Some(rewrite_with_preimage(interval, op, inner_expr)?))
        }
        None => Ok(None),
    }
}

/// Rewrites a predicate over a half-open source-domain interval `[lower, upper)`.
fn rewrite_with_preimage(
    interval: datafusion_expr_common::interval_arithmetic::Interval,
    op: Operator,
    expr: Arc<dyn PhysicalExpr>,
) -> Result<Arc<dyn PhysicalExpr>> {
    let (lower, upper) = interval.into_bounds();
    let (lower, upper) = (lit(lower), lit(upper));

    Ok(match op {
        Operator::Lt => binary(Arc::clone(&expr), Operator::Lt, lower),
        Operator::LtEq => binary(Arc::clone(&expr), Operator::Lt, upper),
        Operator::Gt => binary(Arc::clone(&expr), Operator::GtEq, upper),
        Operator::GtEq => binary(Arc::clone(&expr), Operator::GtEq, lower),
        Operator::Eq => binary(
            binary(Arc::clone(&expr), Operator::GtEq, lower),
            Operator::And,
            binary(expr, Operator::Lt, upper),
        ),
        Operator::NotEq => binary(
            binary(Arc::clone(&expr), Operator::Lt, lower),
            Operator::Or,
            binary(expr, Operator::GtEq, upper),
        ),
        // The range comparisons are nullable. Distinctness is not, so retain
        // explicit null semantics around the half-open range.
        Operator::IsNotDistinctFrom => binary(
            binary(
                is_not_null(Arc::clone(&expr))?,
                Operator::And,
                binary(Arc::clone(&expr), Operator::GtEq, lower),
            ),
            Operator::And,
            binary(expr, Operator::Lt, upper),
        ),
        Operator::IsDistinctFrom => binary(
            binary(
                binary(Arc::clone(&expr), Operator::Lt, lower),
                Operator::Or,
                binary(Arc::clone(&expr), Operator::GtEq, upper),
            ),
            Operator::Or,
            is_null(expr)?,
        ),
        _ => unreachable!("preimage only supports comparison operators"),
    })
}

fn binary(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, op, right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::col;
    use arrow::datatypes::{Field, TimeUnit};
    use datafusion_common::tree_node::TreeNode;

    /// Check if an expression is a cast expression
    fn is_cast_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
        expr.downcast_ref::<CastExpr>().is_some()
            || expr.downcast_ref::<TryCastExpr>().is_some()
    }

    /// Check if a binary expression is suitable for cast unwrapping
    fn is_binary_expr_with_cast_and_literal(binary: &BinaryExpr) -> bool {
        // Check if left is cast and right is literal
        let left_cast_right_literal = is_cast_expr(binary.left())
            && binary.right().downcast_ref::<Literal>().is_some();

        // Check if left is literal and right is cast
        let left_literal_right_cast = binary.left().downcast_ref::<Literal>().is_some()
            && is_cast_expr(binary.right());

        left_cast_right_literal || left_literal_right_cast
    }

    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int64, false),
            Field::new("c3", DataType::Utf8, false),
        ])
    }

    fn timestamp_scalar(
        unit: TimeUnit,
        value: i64,
        timezone: Option<Arc<str>>,
    ) -> ScalarValue {
        match unit {
            TimeUnit::Second => ScalarValue::TimestampSecond(Some(value), timezone),
            TimeUnit::Millisecond => {
                ScalarValue::TimestampMillisecond(Some(value), timezone)
            }
            TimeUnit::Microsecond => {
                ScalarValue::TimestampMicrosecond(Some(value), timezone)
            }
            TimeUnit::Nanosecond => {
                ScalarValue::TimestampNanosecond(Some(value), timezone)
            }
        }
    }

    fn assert_timestamp_widening_unwrap(
        source_unit: TimeUnit,
        target_unit: TimeUnit,
        timezone: Option<Arc<str>>,
        op: Operator,
        target_value: i64,
        expected_value: i64,
        try_cast: bool,
    ) {
        let source_type = DataType::Timestamp(source_unit, timezone.clone());
        let target_type = DataType::Timestamp(target_unit, timezone.clone());
        let schema = Schema::new(vec![Field::new("ts", source_type, true)]);
        let inner = col("ts", &schema).unwrap();
        let cast_expr: Arc<dyn PhysicalExpr> = if try_cast {
            Arc::new(TryCastExpr::new(inner, target_type))
        } else {
            Arc::new(CastExpr::new(inner, target_type, None))
        };
        let input: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            cast_expr,
            op,
            lit(timestamp_scalar(
                target_unit,
                target_value,
                timezone.clone(),
            )),
        ));

        let result = unwrap_cast_in_comparison(input, &schema).unwrap();
        assert!(result.transformed);
        let binary = result.data.downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(*binary.op(), op);
        assert!(!is_cast_expr(binary.left()));
        assert_eq!(
            binary.right().downcast_ref::<Literal>().unwrap().value(),
            &timestamp_scalar(source_unit, expected_value, timezone),
        );
    }

    #[test]
    fn test_timestamp_widening_ordered_cast_and_try_cast() {
        for (source_unit, target_unit, quotient) in [
            (TimeUnit::Second, TimeUnit::Millisecond, 1_000_i128),
            (TimeUnit::Second, TimeUnit::Microsecond, 1_000_000),
            (TimeUnit::Second, TimeUnit::Nanosecond, 1_000_000_000),
            (TimeUnit::Millisecond, TimeUnit::Microsecond, 1_000),
            (TimeUnit::Millisecond, TimeUnit::Nanosecond, 1_000_000),
            (TimeUnit::Microsecond, TimeUnit::Nanosecond, 1_000),
        ] {
            for (target_value, floor, ceil) in [
                (quotient + 1, 1, 2),
                (1 - quotient, -1, 0),
                (123 * quotient, 123, 123),
                (-123 * quotient, -123, -123),
            ] {
                for (op, expected) in [
                    (Operator::GtEq, ceil),
                    (Operator::Gt, floor),
                    (Operator::Lt, ceil),
                    (Operator::LtEq, floor),
                ] {
                    for try_cast in [false, true] {
                        assert_timestamp_widening_unwrap(
                            source_unit,
                            target_unit,
                            None,
                            op,
                            i64::try_from(target_value).unwrap(),
                            expected,
                            try_cast,
                        );
                    }
                }
            }

            for target_value in [i128::from(i64::MIN), i128::from(i64::MAX)] {
                let floor = target_value.div_euclid(quotient);
                let ceil = floor
                    + if target_value.rem_euclid(quotient) != 0 {
                        1
                    } else {
                        0
                    };
                for (op, expected) in [
                    (Operator::GtEq, ceil),
                    (Operator::Gt, floor),
                    (Operator::Lt, ceil),
                    (Operator::LtEq, floor),
                ] {
                    assert_timestamp_widening_unwrap(
                        source_unit,
                        target_unit,
                        None,
                        op,
                        i64::try_from(target_value).unwrap(),
                        i64::try_from(expected).unwrap(),
                        false,
                    );
                }
            }
        }
    }

    #[test]
    fn test_timestamp_widening_literal_left_and_timezone() {
        let timezone = Some(Arc::from("+05:30"));
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, timezone.clone()),
            true,
        )]);
        for (op, expected_op, expected_value) in [
            (Operator::Lt, Operator::Gt, 123),
            (Operator::LtEq, Operator::GtEq, 124),
            (Operator::Gt, Operator::Lt, 124),
            (Operator::GtEq, Operator::LtEq, 123),
        ] {
            for try_cast in [false, true] {
                let inner = col("ts", &schema).unwrap();
                let target = DataType::Timestamp(TimeUnit::Nanosecond, timezone.clone());
                let cast_expr: Arc<dyn PhysicalExpr> = if try_cast {
                    Arc::new(TryCastExpr::new(inner, target))
                } else {
                    Arc::new(CastExpr::new(inner, target, None))
                };
                let input: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                    lit(timestamp_scalar(
                        TimeUnit::Nanosecond,
                        123_456_789,
                        timezone.clone(),
                    )),
                    op,
                    cast_expr,
                ));
                let result = unwrap_cast_in_comparison(input, &schema).unwrap();
                assert!(result.transformed);
                let binary = result.data.downcast_ref::<BinaryExpr>().unwrap();
                assert_eq!(*binary.op(), expected_op);
                assert_eq!(
                    binary.right().downcast_ref::<Literal>().unwrap().value(),
                    &timestamp_scalar(
                        TimeUnit::Millisecond,
                        expected_value,
                        timezone.clone()
                    ),
                );
            }
        }
    }

    #[test]
    fn test_timestamp_widening_equality_and_guards_retain_cast() {
        let source = DataType::Timestamp(TimeUnit::Millisecond, None);
        let target = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![Field::new("ts", source, true)]);
        for (op, literal) in [
            (
                Operator::Eq,
                ScalarValue::TimestampNanosecond(Some(123_000_000), None),
            ),
            (Operator::GtEq, ScalarValue::TimestampNanosecond(None, None)),
            (
                Operator::GtEq,
                ScalarValue::TimestampMicrosecond(Some(123_456), None),
            ),
        ] {
            let input: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::new(CastExpr::new(
                    col("ts", &schema).unwrap(),
                    target.clone(),
                    None,
                )),
                op,
                lit(literal),
            ));
            assert!(
                !unwrap_cast_in_comparison(input, &schema)
                    .unwrap()
                    .transformed
            );
        }
    }

    #[test]
    fn test_unwrap_cast_in_binary_comparison() {
        let schema = test_schema();

        // Create: cast(c1 as INT64) > INT64(10)
        let column_expr = col("c1", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let literal_expr = lit(10i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Gt, literal_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should be transformed
        assert!(result.transformed);

        // The result should be: c1 > INT32(10)
        let optimized = result.data;
        let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

        // Check that left side is no longer a cast
        assert!(!is_cast_expr(optimized_binary.left()));

        // Check that right side is a literal with the correct type and value
        let right_literal = optimized_binary.right().downcast_ref::<Literal>().unwrap();
        assert_eq!(right_literal.value(), &ScalarValue::Int32(Some(10)));
    }

    #[test]
    fn test_unwrap_cast_with_literal_on_left() {
        let schema = test_schema();

        // Create: INT64(10) < cast(c1 as INT64)
        let column_expr = col("c1", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let literal_expr = lit(10i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(literal_expr, Operator::Lt, cast_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should be transformed
        assert!(result.transformed);

        // The result should be equivalent to: c1 > INT32(10)
        let optimized = result.data;
        let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

        // Check the operator was swapped
        assert_eq!(*optimized_binary.op(), Operator::Gt);
    }

    #[test]
    fn test_no_unwrap_date64_to_date32_narrowing() {
        let schema = Schema::new(vec![Field::new("d64", DataType::Date64, false)]);

        // cast(d64 AS Date32) = Date32(20089) must NOT unwrap: narrowing a Date64
        // column to Date32 truncates milliseconds to the day (many-to-one), so the
        // rewritten `d64 = <midnight ms>` would drop sub-day rows.
        let column_expr = col("d64", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Date32, None));
        let literal_expr = lit(ScalarValue::Date32(Some(20089)));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Eq, literal_expr));

        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();
        assert!(!result.transformed);
    }

    #[test]
    fn test_no_unwrap_when_types_unsupported() {
        let schema = Schema::new(vec![Field::new("f1", DataType::Float32, false)]);

        // Create: cast(f1 as FLOAT64) > FLOAT64(10.5)
        let column_expr = col("f1", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Float64, None));
        let literal_expr = lit(10.5f64);
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Gt, literal_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should NOT be transformed (floating point types not supported)
        assert!(!result.transformed);
    }

    #[test]
    fn test_is_binary_expr_with_cast_and_literal() {
        let schema = test_schema();

        let column_expr = col("c1", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let literal_expr = lit(10i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Gt, literal_expr));
        assert!(is_binary_expr_with_cast_and_literal(&binary_expr));
    }

    #[test]
    fn test_unwrap_cast_literal_on_left_side() {
        // Test case for: literal <= cast(column)
        // This was the specific case that caused the bug
        let schema = Schema::new(vec![Field::new(
            "decimal_col",
            DataType::Decimal128(9, 2),
            true,
        )]);

        // Create: Decimal128(400) <= cast(decimal_col as Decimal128(22, 2))
        let column_expr = col("decimal_col", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(
            column_expr,
            DataType::Decimal128(22, 2),
            None,
        ));
        let literal_expr = lit(ScalarValue::Decimal128(Some(400), 22, 2));
        let binary_expr =
            Arc::new(BinaryExpr::new(literal_expr, Operator::LtEq, cast_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should be transformed
        assert!(result.transformed);

        // The result should be: decimal_col >= Decimal128(400, 9, 2)
        let optimized = result.data;
        let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

        // Check operator was swapped correctly
        assert_eq!(*optimized_binary.op(), Operator::GtEq);

        // Check that left side is the column without cast
        assert!(!is_cast_expr(optimized_binary.left()));

        // Check that right side is a literal with the correct type
        let right_literal = optimized_binary.right().downcast_ref::<Literal>().unwrap();
        assert_eq!(
            right_literal.value().data_type(),
            DataType::Decimal128(9, 2)
        );
    }

    #[test]
    fn test_unwrap_cast_with_different_comparison_operators() {
        let schema = Schema::new(vec![Field::new("int_col", DataType::Int32, false)]);

        // Test all comparison operators with literal on the left
        let operators = vec![
            (Operator::Lt, Operator::Gt),
            (Operator::LtEq, Operator::GtEq),
            (Operator::Gt, Operator::Lt),
            (Operator::GtEq, Operator::LtEq),
            (Operator::Eq, Operator::Eq),
            (Operator::NotEq, Operator::NotEq),
        ];

        for (original_op, expected_op) in operators {
            // Create: INT64(100) op cast(int_col as INT64)
            let column_expr = col("int_col", &schema).unwrap();
            let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
            let literal_expr = lit(100i64);
            let binary_expr =
                Arc::new(BinaryExpr::new(literal_expr, original_op, cast_expr));

            // Apply unwrap cast optimization
            let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

            // Should be transformed
            assert!(result.transformed);

            let optimized = result.data;
            let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

            // Check the operator was swapped correctly
            assert_eq!(
                *optimized_binary.op(),
                expected_op,
                "Failed for operator {original_op:?} -> {expected_op:?}"
            );

            // Check that left side has no cast
            assert!(!is_cast_expr(optimized_binary.left()));

            // Check that the literal was cast to the column type
            let right_literal =
                optimized_binary.right().downcast_ref::<Literal>().unwrap();
            assert_eq!(right_literal.value(), &ScalarValue::Int32(Some(100)));
        }
    }

    #[test]
    fn test_unwrap_cast_with_decimal_types() {
        // Test various decimal precision/scale combinations
        let test_cases = vec![
            // (column_precision, column_scale, cast_precision, cast_scale, value)
            (9, 2, 22, 2, 400),
            (10, 3, 20, 3, 1000),
            (5, 1, 10, 1, 99),
        ];

        for (col_p, col_s, cast_p, cast_s, value) in test_cases {
            let schema = Schema::new(vec![Field::new(
                "decimal_col",
                DataType::Decimal128(col_p, col_s),
                true,
            )]);

            // Test both: cast(column) op literal AND literal op cast(column)

            // Case 1: cast(column) > literal
            let column_expr = col("decimal_col", &schema).unwrap();
            let cast_expr = Arc::new(CastExpr::new(
                Arc::clone(&column_expr),
                DataType::Decimal128(cast_p, cast_s),
                None,
            ));
            let literal_expr = lit(ScalarValue::Decimal128(Some(value), cast_p, cast_s));
            let binary_expr =
                Arc::new(BinaryExpr::new(cast_expr, Operator::Gt, literal_expr));

            let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();
            assert!(result.transformed);

            // Case 2: literal < cast(column)
            let cast_expr = Arc::new(CastExpr::new(
                column_expr,
                DataType::Decimal128(cast_p, cast_s),
                None,
            ));
            let literal_expr = lit(ScalarValue::Decimal128(Some(value), cast_p, cast_s));
            let binary_expr =
                Arc::new(BinaryExpr::new(literal_expr, Operator::Lt, cast_expr));

            let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();
            assert!(result.transformed);
        }
    }

    #[test]
    fn test_unwrap_cast_with_null_literals() {
        // Test with NULL literals to ensure they're handled correctly
        let schema = Schema::new(vec![Field::new("int_col", DataType::Int32, true)]);

        // Create: cast(int_col as INT64) = NULL
        let column_expr = col("int_col", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let null_literal = lit(ScalarValue::Int64(None));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Eq, null_literal));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should be transformed
        assert!(result.transformed);

        // Verify the NULL was cast to the column type
        let optimized = result.data;
        let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();
        let right_literal = optimized_binary.right().downcast_ref::<Literal>().unwrap();
        assert_eq!(right_literal.value(), &ScalarValue::Int32(None));
    }

    #[test]
    fn test_unwrap_cast_with_try_cast() {
        // Test that TryCast expressions are also unwrapped correctly
        let schema = Schema::new(vec![Field::new("str_col", DataType::Utf8, true)]);

        // Create: try_cast(str_col as INT64) > INT64(100)
        let column_expr = col("str_col", &schema).unwrap();
        let try_cast_expr = Arc::new(TryCastExpr::new(column_expr, DataType::Int64));
        let literal_expr = lit(100i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(try_cast_expr, Operator::Gt, literal_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should NOT be transformed (string to int cast not supported)
        assert!(!result.transformed);
    }

    #[test]
    fn test_unwrap_cast_preserves_non_comparison_operators() {
        // Test that non-comparison operators in AND/OR expressions are preserved
        let schema = Schema::new(vec![Field::new("int_col", DataType::Int32, false)]);

        // Create: cast(int_col as INT64) > INT64(10) AND cast(int_col as INT64) < INT64(20)
        let column_expr = col("int_col", &schema).unwrap();

        let cast1 = Arc::new(CastExpr::new(
            Arc::clone(&column_expr),
            DataType::Int64,
            None,
        ));
        let lit1 = lit(10i64);
        let compare1 = Arc::new(BinaryExpr::new(cast1, Operator::Gt, lit1));

        let cast2 = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let lit2 = lit(20i64);
        let compare2 = Arc::new(BinaryExpr::new(cast2, Operator::Lt, lit2));

        let and_expr = Arc::new(BinaryExpr::new(compare1, Operator::And, compare2));

        // Apply unwrap cast optimization recursively
        let result = (and_expr as Arc<dyn PhysicalExpr>)
            .transform_down(|node| unwrap_cast_in_comparison(node, &schema))
            .unwrap();

        // Should be transformed
        assert!(result.transformed);

        // Verify the AND operator is preserved
        let optimized = result.data;
        let and_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(*and_binary.op(), Operator::And);

        // Both sides should have their casts unwrapped
        let left_binary = and_binary.left().downcast_ref::<BinaryExpr>().unwrap();
        let right_binary = and_binary.right().downcast_ref::<BinaryExpr>().unwrap();

        assert!(!is_cast_expr(left_binary.left()));
        assert!(!is_cast_expr(right_binary.left()));
    }

    #[test]
    fn test_try_cast_unwrapping() {
        let schema = test_schema();

        // Create: try_cast(c1 as INT64) <= INT64(100)
        let column_expr = col("c1", &schema).unwrap();
        let try_cast_expr = Arc::new(TryCastExpr::new(column_expr, DataType::Int64));
        let literal_expr = lit(100i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(try_cast_expr, Operator::LtEq, literal_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should be transformed to: c1 <= INT32(100)
        assert!(result.transformed);

        let optimized = result.data;
        let optimized_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

        // Verify the try_cast was removed
        assert!(!is_cast_expr(optimized_binary.left()));

        // Verify the literal was converted
        let right_literal = optimized_binary.right().downcast_ref::<Literal>().unwrap();
        assert_eq!(right_literal.value(), &ScalarValue::Int32(Some(100)));
    }

    #[test]
    fn test_non_swappable_operator() {
        // Test case with an operator that cannot be swapped
        let schema = Schema::new(vec![Field::new("int_col", DataType::Int32, false)]);

        // Create: INT64(10) + cast(int_col as INT64)
        // The Plus operator cannot be swapped, so this should not be transformed
        let column_expr = col("int_col", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let literal_expr = lit(10i64);
        let binary_expr =
            Arc::new(BinaryExpr::new(literal_expr, Operator::Plus, cast_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should NOT be transformed because Plus cannot be swapped
        assert!(!result.transformed);
    }

    #[test]
    fn test_cast_that_cannot_be_unwrapped_overflow() {
        // Test case where the literal value would overflow the target type
        let schema = Schema::new(vec![Field::new("small_int", DataType::Int8, false)]);

        // Create: cast(small_int as INT64) > INT64(1000)
        // This should NOT be unwrapped because 1000 cannot fit in Int8 (max value is 127)
        let column_expr = col("small_int", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(column_expr, DataType::Int64, None));
        let literal_expr = lit(1000i64); // Value too large for Int8
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Gt, literal_expr));

        // Apply unwrap cast optimization
        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        // Should NOT be transformed due to overflow
        assert!(!result.transformed);
    }

    #[test]
    fn test_unwrap_timestamp_precision_narrowing_to_range() {
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        )]);

        let column_expr = col("ts", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(
            column_expr,
            DataType::Timestamp(TimeUnit::Millisecond, None),
            None,
        ));
        let literal_expr = lit(ScalarValue::TimestampMillisecond(Some(1), None));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Eq, literal_expr));

        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        assert!(result.transformed);
        let range = result.data.downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(*range.op(), Operator::And);
        let lower = range.left().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(*lower.op(), Operator::GtEq);
        assert_eq!(
            lower.right().downcast_ref::<Literal>().unwrap().value(),
            &ScalarValue::TimestampNanosecond(Some(1_000_000), None)
        );
        let upper = range.right().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(*upper.op(), Operator::Lt);
        assert_eq!(
            upper.right().downcast_ref::<Literal>().unwrap().value(),
            &ScalarValue::TimestampNanosecond(Some(2_000_000), None)
        );
    }

    #[test]
    fn test_not_unwrap_timestamp_precision_widening_equality() {
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        )]);

        let column_expr = col("ts", &schema).unwrap();
        let cast_expr = Arc::new(CastExpr::new(
            column_expr,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            None,
        ));
        let literal_expr = lit(ScalarValue::TimestampNanosecond(Some(1_000_000), None));
        let binary_expr =
            Arc::new(BinaryExpr::new(cast_expr, Operator::Eq, literal_expr));

        let result = unwrap_cast_in_comparison(binary_expr, &schema).unwrap();

        assert!(!result.transformed);
        let unchanged = result.data.downcast_ref::<BinaryExpr>().unwrap();
        assert!(is_cast_expr(unchanged.left()));
    }

    #[test]
    fn test_complex_nested_expression() {
        let schema = test_schema();

        // Create a more complex expression with nested casts
        // (cast(c1 as INT64) > INT64(10)) AND (cast(c2 as INT32) = INT32(20))
        let c1_expr = col("c1", &schema).unwrap();
        let c1_cast = Arc::new(CastExpr::new(c1_expr, DataType::Int64, None));
        let c1_literal = lit(10i64);
        let c1_binary = Arc::new(BinaryExpr::new(c1_cast, Operator::Gt, c1_literal));

        let c2_expr = col("c2", &schema).unwrap();
        let c2_cast = Arc::new(CastExpr::new(c2_expr, DataType::Int32, None));
        let c2_literal = lit(20i32);
        let c2_binary = Arc::new(BinaryExpr::new(c2_cast, Operator::Eq, c2_literal));

        // Create AND expression
        let and_expr = Arc::new(BinaryExpr::new(c1_binary, Operator::And, c2_binary));

        // Apply unwrap cast optimization recursively
        let result = (and_expr as Arc<dyn PhysicalExpr>)
            .transform_down(|node| unwrap_cast_in_comparison(node, &schema))
            .unwrap();

        // Should be transformed
        assert!(result.transformed);

        // Verify both sides of the AND were optimized
        let optimized = result.data;
        let and_binary = optimized.downcast_ref::<BinaryExpr>().unwrap();

        // Left side should be: c1 > INT32(10)
        let left_binary = and_binary.left().downcast_ref::<BinaryExpr>().unwrap();
        assert!(!is_cast_expr(left_binary.left()));
        let left_literal = left_binary.right().downcast_ref::<Literal>().unwrap();
        assert_eq!(left_literal.value(), &ScalarValue::Int32(Some(10)));

        // Right side remains CAST(c2 AS INT32) = INT32(20): narrowing casts
        // have no exact source-domain preimage.
        let right_binary = and_binary.right().downcast_ref::<BinaryExpr>().unwrap();
        assert!(is_cast_expr(right_binary.left()));
    }
}
