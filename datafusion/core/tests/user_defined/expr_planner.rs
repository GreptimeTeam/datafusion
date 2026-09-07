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

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use datafusion::common::test_util::batches_to_string;
use std::sync::Arc;

use datafusion::common::DFSchema;
use datafusion::error::Result;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use datafusion::sql::sqlparser::ast::BinaryOperator;
use datafusion_common::ScalarValue;
use datafusion_expr::expr::Alias;
use datafusion_expr::planner::{
    ExprPlanner, PlannerResult, RawBinaryExpr, RawScalarExpr,
};
use datafusion_expr::{BinaryExpr, ColumnarValue, Volatility};

#[derive(Debug)]
struct MyCustomPlanner;

impl ExprPlanner for MyCustomPlanner {
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        _schema: &DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>> {
        match &expr.op {
            BinaryOperator::Arrow => {
                Ok(PlannerResult::Planned(Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(expr.left.clone()),
                    right: Box::new(expr.right.clone()),
                    op: Operator::StringConcat,
                })))
            }
            BinaryOperator::LongArrow => {
                Ok(PlannerResult::Planned(Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(expr.left.clone()),
                    right: Box::new(expr.right.clone()),
                    op: Operator::Plus,
                })))
            }
            BinaryOperator::Question => {
                Ok(PlannerResult::Planned(Expr::Alias(Alias::new(
                    Expr::Literal(ScalarValue::Boolean(Some(true)), None),
                    None::<&str>,
                    format!("{} ? {}", expr.left, expr.right),
                ))))
            }
            _ => Ok(PlannerResult::Original(expr)),
        }
    }
}

#[derive(Debug)]
struct ScalarUdfArgPlanner;

impl ExprPlanner for ScalarUdfArgPlanner {
    fn plan_scalar(
        &self,
        mut expr: RawScalarExpr,
    ) -> Result<PlannerResult<RawScalarExpr>> {
        expr.args = vec![lit(2_i64)];
        Ok(PlannerResult::Original(expr))
    }
}

async fn plan_and_collect(sql: &str) -> Result<Vec<RecordBatch>> {
    let config =
        SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgres");
    let mut ctx = SessionContext::new_with_config(config);
    ctx.register_expr_planner(Arc::new(MyCustomPlanner))?;
    ctx.sql(sql).await?.collect().await
}

#[tokio::test]
async fn test_scalar_udf_args_are_planned() -> Result<()> {
    let mut ctx = SessionContext::new();
    ctx.register_udf(create_udf(
        "replace_scalar_arg",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
    ));

    ctx.register_expr_planner(Arc::new(ScalarUdfArgPlanner))?;

    let dataframe = ctx.sql("SELECT replace_scalar_arg(1)").await?;

    assert_eq!(
        format!("{}", dataframe.logical_plan()),
        "Projection: replace_scalar_arg(Int64(2))\n  EmptyRelation: rows=1"
    );

    Ok(())
}

#[tokio::test]
async fn test_custom_operators_arrow() {
    let actual = plan_and_collect("select 'foo'->'bar';").await.unwrap();
    insta::assert_snapshot!(batches_to_string(&actual), @r#"
    +----------------------------+
    | Utf8("foo") || Utf8("bar") |
    +----------------------------+
    | foobar                     |
    +----------------------------+
    "#);
}

#[tokio::test]
async fn test_custom_operators_long_arrow() {
    let actual = plan_and_collect("select 1->>2;").await.unwrap();
    insta::assert_snapshot!(batches_to_string(&actual), @r"
    +---------------------+
    | Int64(1) + Int64(2) |
    +---------------------+
    | 3                   |
    +---------------------+
    ");
}

#[tokio::test]
async fn test_question_select() {
    let actual = plan_and_collect("select a ? 2 from (select 1 as a);")
        .await
        .unwrap();
    insta::assert_snapshot!(batches_to_string(&actual), @r"
    +--------------+
    | a ? Int64(2) |
    +--------------+
    | true         |
    +--------------+
    ");
}

#[tokio::test]
async fn test_question_filter() {
    let actual = plan_and_collect("select a from (select 1 as a) where a ? 2;")
        .await
        .unwrap();
    insta::assert_snapshot!(batches_to_string(&actual), @r"
    +---+
    | a |
    +---+
    | 1 |
    +---+
    ");
}
