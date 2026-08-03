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

//! Preimage rewrites for cast comparisons.

use arrow::datatypes::DataType;
use datafusion_common::{Result, internal_err, tree_node::Transformed};
use datafusion_expr::expr::InList;
use datafusion_expr::{
    BinaryExpr, Cast, Expr, Operator, TryCast, lit, simplify::SimplifyContext,
};
use datafusion_expr_common::casts::{
    CastPredicatePreimage, cast_predicate_preimage, exact_preimage_cast,
};

use super::udf_preimage::rewrite_with_preimage;

pub(super) fn rewrite_cast_predicate_for_binary(
    info: &SimplifyContext,
    cast_expr: Expr,
    literal: Expr,
    op: Operator,
) -> Result<Transformed<Expr>> {
    let Some((expr, target_type)) = cast_input_and_type(cast_expr) else {
        return internal_err!("Expect cast expr");
    };
    let Expr::Literal(lit_value, _) = literal else {
        return internal_err!("Expect literal expr");
    };

    let source_type = info.get_data_type(&expr)?;
    match cast_predicate_preimage(&source_type, &target_type, op, &lit_value)? {
        Some(CastPredicatePreimage::Range(interval)) => {
            rewrite_with_preimage(interval, op, *expr)
        }
        Some(CastPredicatePreimage::Exact(value)) => {
            Ok(Transformed::yes(Expr::BinaryExpr(BinaryExpr {
                left: expr,
                op,
                right: Box::new(lit(value)),
            })))
        }
        None => internal_err!(
            "Can't compute cast predicate preimage for source type {} target type {} literal {:?}",
            source_type,
            target_type,
            lit_value
        ),
    }
}

pub(super) fn supports_cast_predicate_for_binary(
    info: &SimplifyContext,
    expr: &Expr,
    op: Operator,
    literal: &Expr,
) -> bool {
    if !matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    ) {
        return false;
    }

    let Some((inner_expr, target_type)) = cast_input_and_type_ref(expr) else {
        return false;
    };
    let Expr::Literal(lit_value, _) = literal else {
        return false;
    };
    let Ok(source_type) = info.get_data_type(inner_expr) else {
        return false;
    };

    cast_predicate_preimage(&source_type, target_type, op, lit_value)
        .ok()
        .flatten()
        .is_some()
}

pub(super) fn supports_cast_predicate_for_inlist(
    info: &SimplifyContext,
    expr: &Expr,
    list: &[Expr],
) -> bool {
    let Some((inner_expr, target_type)) = cast_input_and_type_ref(expr) else {
        return false;
    };
    let Ok(source_type) = info.get_data_type(inner_expr) else {
        return false;
    };

    // IN-list rewrites only support singleton exact cast preimages. They do
    // not use range preimages (such as timestamp precision narrowing) or
    // binary-operator-only integer-to-string equality handling.
    list.iter().all(|right| match right {
        Expr::Literal(lit_value, _) => {
            exact_preimage_cast(&source_type, target_type, lit_value).is_some()
        }
        _ => false,
    })
}

pub(super) fn rewrite_cast_predicate_for_inlist(
    info: &SimplifyContext,
    expr: Expr,
    list: Vec<Expr>,
    negated: bool,
) -> Result<Transformed<Expr>> {
    let Some((inner_expr, target_type)) = cast_input_and_type(expr) else {
        return internal_err!("Expect cast expr");
    };
    let source_type = info.get_data_type(&inner_expr)?;

    let list = list
        .into_iter()
        .map(|right| match right {
            Expr::Literal(lit_value, _) => {
                let Some(value) =
                    exact_preimage_cast(&source_type, &target_type, &lit_value)
                else {
                    return internal_err!(
                        "Can't cast the list expr {:?} to type {}",
                        lit_value,
                        source_type
                    );
                };
                Ok(lit(value))
            }
            other_expr => internal_err!(
                "Only support literal expr to optimize, but the expr is {:?}",
                other_expr
            ),
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Transformed::yes(Expr::InList(InList {
        expr: inner_expr,
        list,
        negated,
    })))
}

fn cast_input_and_type(cast_expr: Expr) -> Option<(Box<Expr>, DataType)> {
    match cast_expr {
        Expr::TryCast(TryCast { expr, field, .. })
        | Expr::Cast(Cast { expr, field, .. }) => Some((expr, field.data_type().clone())),
        _ => None,
    }
}

fn cast_input_and_type_ref(cast_expr: &Expr) -> Option<(&Expr, &DataType)> {
    match cast_expr {
        Expr::TryCast(TryCast { expr, field, .. })
        | Expr::Cast(Cast { expr, field, .. }) => {
            Some((expr.as_ref(), field.data_type()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::datatypes::{Field, TimeUnit};
    use datafusion_common::{DFSchema, DFSchemaRef, ScalarValue};
    use datafusion_expr::simplify::SimplifyContext;
    use datafusion_expr::{binary_expr, cast, col, in_list, lit, try_cast};

    use super::*;
    use crate::simplify_expressions::ExprSimplifier;

    #[test]
    fn exact_preimage_unwraps_binary_and_literal_left() {
        let schema = expr_test_schema();

        let expr = cast(col("c1"), DataType::Int64).gt(lit(10_i64));
        assert_eq!(optimize_test(expr, &schema), col("c1").gt(lit(10_i32)));

        let expr = try_cast(col("c1"), DataType::Int64).gt(lit(10_i64));
        assert_eq!(optimize_test(expr, &schema), col("c1").gt(lit(10_i32)));

        let expr = lit(10_i64).lt(cast(col("c1"), DataType::Int64));
        assert_eq!(optimize_test(expr, &schema), col("c1").gt(lit(10_i32)));
    }

    #[test]
    fn integer_string_round_trip_uses_exact_preimage() {
        let schema = expr_test_schema();

        let expr = cast(col("c1"), DataType::Utf8).eq(lit("123"));
        assert_eq!(optimize_test(expr, &schema), col("c1").eq(lit(123_i32)));

        let expr = cast(col("c1"), DataType::Utf8).eq(lit("0123"));
        assert_eq!(optimize_test(expr.clone(), &schema), expr);
    }

    #[test]
    fn exact_preimage_in_and_not_in_preserve_typed_nulls() {
        let schema = expr_test_schema();
        for negated in [false, true] {
            let expr = in_list(
                cast(col("c1"), DataType::Int64),
                vec![
                    lit(0_i64),
                    lit(ScalarValue::Int64(None)),
                    lit(1_i64),
                    lit(2_i64),
                    lit(3_i64),
                    lit(4_i64),
                ],
                negated,
            );
            let expected = in_list(
                col("c1"),
                vec![
                    lit(0_i32),
                    lit(ScalarValue::Int32(None)),
                    lit(1_i32),
                    lit(2_i32),
                    lit(3_i32),
                    lit(4_i32),
                ],
                negated,
            );
            assert_eq!(optimize_test(expr, &schema), expected);
        }
    }

    #[test]
    fn timestamp_narrowing_uses_half_open_truncation_toward_zero_ranges() {
        let schema = expr_test_schema();
        for (literal_ms, expected) in [
            (
                1000,
                col("ts_nano")
                    .gt_eq(lit_timestamp_nano(1_000_000_000))
                    .and(col("ts_nano").lt(lit_timestamp_nano(1_001_000_000))),
            ),
            (
                0,
                col("ts_nano")
                    .gt_eq(lit_timestamp_nano(-999_999))
                    .and(col("ts_nano").lt(lit_timestamp_nano(1_000_000))),
            ),
            (
                -1,
                col("ts_nano")
                    .gt_eq(lit_timestamp_nano(-1_999_999))
                    .and(col("ts_nano").lt(lit_timestamp_nano(-999_999))),
            ),
        ] {
            let expr = cast(col("ts_nano"), timestamp_millis_type())
                .eq(lit_timestamp_millis(literal_ms));
            assert_eq!(optimize_test(expr, &schema), expected);
        }
    }

    #[test]
    fn nested_timestamp_cast_uses_a_range_over_the_inner_cast() {
        let schema = Arc::new(
            DFSchema::from_unqualified_fields(
                vec![Field::new("n", DataType::Int64, false)].into(),
                HashMap::new(),
            )
            .unwrap(),
        );
        let inner_cast = cast(col("n"), timestamp_nano_type());
        let expr =
            cast(inner_cast.clone(), timestamp_millis_type()).eq(lit_timestamp_millis(1));
        let expected = inner_cast
            .clone()
            .gt_eq(lit_timestamp_nano(1_000_000))
            .and(inner_cast.lt(lit_timestamp_nano(2_000_000)));

        let optimized = ExprSimplifier::new(
            SimplifyContext::builder()
                .with_schema(Arc::clone(&schema))
                .build(),
        )
        .simplify(expr)
        .unwrap();

        assert_eq!(optimized, expected);
    }

    #[test]
    fn timestamp_narrowing_uses_correct_inequality_and_distinctness_preimages() {
        let schema = expr_test_schema();
        for (op, literal_ms, expected) in [
            (
                Operator::Gt,
                1000,
                col("ts_nano").gt_eq(lit_timestamp_nano(1_001_000_000)),
            ),
            (
                Operator::Lt,
                0,
                col("ts_nano").lt(lit_timestamp_nano(-999_999)),
            ),
            (
                Operator::IsNotDistinctFrom,
                0,
                col("ts_nano")
                    .gt_eq(lit_timestamp_nano(-999_999))
                    .and(col("ts_nano").lt(lit_timestamp_nano(1_000_000))),
            ),
            (
                Operator::IsDistinctFrom,
                0,
                col("ts_nano")
                    .lt(lit_timestamp_nano(-999_999))
                    .or(col("ts_nano").gt_eq(lit_timestamp_nano(1_000_000))),
            ),
        ] {
            let expr = binary_expr(
                cast(col("ts_nano"), timestamp_millis_type()),
                op,
                lit_timestamp_millis(literal_ms),
            );
            assert_eq!(optimize_test(expr, &schema), expected);
        }
    }

    #[test]
    fn timestamp_narrowing_swaps_literal_left_operators() {
        let schema = expr_test_schema();
        for (op, expected) in [
            (
                Operator::Lt,
                col("ts_nano").gt_eq(lit_timestamp_nano(1_001_000_000)),
            ),
            (
                Operator::LtEq,
                col("ts_nano").gt_eq(lit_timestamp_nano(1_000_000_000)),
            ),
            (
                Operator::Gt,
                col("ts_nano").lt(lit_timestamp_nano(1_000_000_000)),
            ),
            (
                Operator::Eq,
                col("ts_nano")
                    .gt_eq(lit_timestamp_nano(1_000_000_000))
                    .and(col("ts_nano").lt(lit_timestamp_nano(1_001_000_000))),
            ),
        ] {
            let expr = binary_expr(
                lit_timestamp_millis(1000),
                op,
                cast(col("ts_nano"), timestamp_millis_type()),
            );
            assert_eq!(optimize_test(expr, &schema), expected);
        }
    }

    #[test]
    fn unsafe_narrowing_and_timezone_casts_remain() {
        let schema = expr_test_schema();

        for expr in [
            cast(col("c2"), DataType::Int32).eq(lit(5_i32)),
            try_cast(col("c2"), DataType::Int32).eq(lit(5_i32)),
            cast(col("c3"), DataType::Decimal128(18, 1)).eq(lit_decimal(12, 18, 1)),
            cast(col("d"), DataType::Int32).eq(lit(0_i32)),
            cast(
                col("ts_nano_none"),
                DataType::Timestamp(TimeUnit::Nanosecond, Some("+05:30".into())),
            )
            .eq(lit(ScalarValue::TimestampNanosecond(
                Some(0),
                Some("+05:30".into()),
            ))),
        ] {
            assert_eq!(optimize_test(expr.clone(), &schema), expr);
        }
    }

    #[test]
    fn timestamp_widening_equality_and_in_remain() {
        let schema = expr_test_schema();
        for expr in [
            cast(col("ts_milli"), timestamp_nano_type())
                .eq(lit_timestamp_nano(123_000_000)),
            try_cast(col("ts_milli"), timestamp_nano_type())
                .eq(lit_timestamp_nano(123_000_000)),
            cast(
                col("ts_seconds"),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            )
            .eq(lit_timestamp_nano(0)),
        ] {
            assert_eq!(optimize_test(expr.clone(), &schema), expr);
        }

        let expr = in_list(
            cast(col("ts_milli"), timestamp_nano_type()),
            vec![
                lit_timestamp_nano(123_000_000),
                lit_timestamp_nano(123_456_789),
            ],
            false,
        );
        assert_eq!(optimize_test(expr.clone(), &schema), expr);
    }

    #[test]
    fn timestamp_widening_ordered_cast_and_try_cast_rewrite() {
        let schema = expr_test_schema();

        for (literal, floor, ceil) in
            [(123_456_789, 123, 124), (-123_456_789, -124, -123)]
        {
            for (op, bound) in [
                (Operator::GtEq, ceil),
                (Operator::Gt, floor),
                (Operator::Lt, ceil),
                (Operator::LtEq, floor),
            ] {
                for cast_expr in [
                    cast(col("ts_milli"), timestamp_nano_type()),
                    try_cast(col("ts_milli"), timestamp_nano_type()),
                ] {
                    let expr = binary_expr(cast_expr, op, lit_timestamp_nano(literal));
                    let expected =
                        binary_expr(col("ts_milli"), op, lit_timestamp_millis(bound));
                    assert_eq!(optimize_test(expr, &schema), expected);
                }
            }
        }

        for (op, expected_op, expected_value) in [
            (Operator::Lt, Operator::Gt, 123),
            (Operator::LtEq, Operator::GtEq, 124),
            (Operator::Gt, Operator::Lt, 124),
            (Operator::GtEq, Operator::LtEq, 123),
        ] {
            let expr = binary_expr(
                lit_timestamp_nano(123_456_789),
                op,
                cast(col("ts_milli"), timestamp_nano_type()),
            );
            let expected = binary_expr(
                col("ts_milli"),
                expected_op,
                lit_timestamp_millis(expected_value),
            );
            assert_eq!(optimize_test(expr, &schema), expected);
        }
    }

    #[test]
    fn in_lists_only_rewrite_exact_preimages() {
        let schema = expr_test_schema();
        for negated in [false, true] {
            let expr = in_list(
                cast(col("c2"), DataType::Int32),
                vec![
                    lit(5_i32),
                    lit(ScalarValue::Int32(None)),
                    lit(5_i32),
                    lit(6_i32),
                ],
                negated,
            );
            assert_eq!(optimize_test(expr.clone(), &schema), expr);

            let expr = in_list(
                cast(col("ts_nano"), timestamp_millis_type()),
                vec![
                    lit_timestamp_millis(0),
                    lit_timestamp_millis(1),
                    lit_timestamp_millis(2),
                    lit_timestamp_millis(3),
                ],
                negated,
            );
            assert_eq!(optimize_test(expr.clone(), &schema), expr);
        }
    }

    #[test]
    fn integer_string_distinctness_uses_the_exact_preimage() {
        let schema = expr_test_schema();

        let expr = binary_expr(
            cast(col("c1"), DataType::Utf8),
            Operator::IsNotDistinctFrom,
            lit("123"),
        );
        assert_eq!(
            optimize_test(expr, &schema),
            binary_expr(col("c1"), Operator::IsNotDistinctFrom, lit(123_i32))
        );

        let expr = binary_expr(
            cast(col("c1"), DataType::Utf8),
            Operator::IsDistinctFrom,
            lit("0123"),
        );
        assert_eq!(optimize_test(expr.clone(), &schema), expr);
    }

    fn optimize_test(expr: Expr, schema: &DFSchemaRef) -> Expr {
        let simplifier = ExprSimplifier::new(
            SimplifyContext::builder()
                .with_schema(Arc::clone(schema))
                .build(),
        );
        simplifier.simplify(expr).unwrap()
    }

    fn expr_test_schema() -> DFSchemaRef {
        Arc::new(
            DFSchema::from_unqualified_fields(
                vec![
                    Field::new("c1", DataType::Int32, false),
                    Field::new("c2", DataType::Int64, false),
                    Field::new("c3", DataType::Decimal128(18, 2), false),
                    Field::new("d", DataType::Date64, false),
                    Field::new(
                        "ts_seconds",
                        DataType::Timestamp(TimeUnit::Second, None),
                        false,
                    ),
                    Field::new("ts_milli", timestamp_millis_type(), false),
                    Field::new("ts_nano", timestamp_nano_type(), false),
                    Field::new(
                        "ts_nano_none",
                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                        false,
                    ),
                ]
                .into(),
                HashMap::new(),
            )
            .unwrap(),
        )
    }

    fn lit_decimal(value: i128, precision: u8, scale: i8) -> Expr {
        lit(ScalarValue::Decimal128(Some(value), precision, scale))
    }

    fn lit_timestamp_nano(value: i64) -> Expr {
        lit(ScalarValue::TimestampNanosecond(Some(value), None))
    }

    fn lit_timestamp_millis(value: i64) -> Expr {
        lit(ScalarValue::TimestampMillisecond(Some(value), None))
    }

    fn timestamp_nano_type() -> DataType {
        DataType::Timestamp(TimeUnit::Nanosecond, None)
    }

    fn timestamp_millis_type() -> DataType {
        DataType::Timestamp(TimeUnit::Millisecond, None)
    }
}
