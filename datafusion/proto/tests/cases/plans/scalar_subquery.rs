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

//! `ScalarSubqueryExec` and the results it scopes to its subtree.

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::expressions::{BinaryExpr, binary, col};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::scalar_subquery::{
    ScalarSubqueryExec, ScalarSubqueryLink,
};
use datafusion::physical_plan::union::UnionExec;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_common::tree_node::TreeNode;
use datafusion_expr::physical_planning_context::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_proto_converter,
    physical_plan_to_bytes_with_proto_converter,
};
use datafusion_proto::physical_plan::{
    DeduplicatingProtoConverter, DefaultPhysicalExtensionCodec,
};
use std::sync::Arc;
use std::vec;

/// Verify that ScalarSubqueryExpr nodes in the input plan are connected to the
/// same shared results container as ScalarSubqueryExec after a proto round-trip.
#[test]
fn roundtrip_scalar_subquery_exec() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let results = ScalarSubqueryResults::new(1);

    // Build the input plan: a filter whose predicate references the
    // scalar subquery result via ScalarSubqueryExpr.
    let sq_expr = Arc::new(ScalarSubqueryExpr::new(
        DataType::Int64,
        true,
        SubqueryIndex::new(0),
        results.clone(),
    ));
    let predicate = binary(col("a", &schema)?, Operator::Eq, sq_expr, &schema)?;
    let filter =
        FilterExec::try_new(predicate, Arc::new(EmptyExec::new(schema.clone())))?;

    // Build a trivial subquery plan.
    let subquery_plan =
        Arc::new(EmptyExec::new(Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Int64,
            true,
        )]))));

    let exec: Arc<dyn ExecutionPlan> = Arc::new(ScalarSubqueryExec::new(
        Arc::new(filter),
        vec![ScalarSubqueryLink {
            plan: subquery_plan,
            index: SubqueryIndex::new(0),
        }],
        results,
    ));

    // Perform the round-trip using DeduplicatingProtoConverter, which
    // creates a DeduplicatingDeserializer that threads scalar subquery
    // results through expression deserialization.
    let codec = DefaultPhysicalExtensionCodec {};
    let converter = DeduplicatingProtoConverter {};
    let bytes = physical_plan_to_bytes_with_proto_converter(
        Arc::clone(&exec),
        &codec,
        &converter,
    )?;
    let ctx = SessionContext::new();
    let deserialized = physical_plan_from_bytes_with_proto_converter(
        bytes.as_ref(),
        ctx.task_ctx().as_ref(),
        &codec,
        &converter,
    )?;

    // Verify the deserialized ScalarSubqueryExec's results container is
    // shared with the ScalarSubqueryExpr in the input plan.
    let sq_exec = deserialized
        .downcast_ref::<ScalarSubqueryExec>()
        .expect("expected ScalarSubqueryExec");
    let exec_results = sq_exec.results();

    // Walk the input plan to find the ScalarSubqueryExpr and verify it
    // points to the same results container.
    let filter_exec = sq_exec
        .input()
        .downcast_ref::<FilterExec>()
        .expect("expected FilterExec");
    let binary_expr = filter_exec
        .predicate()
        .downcast_ref::<BinaryExpr>()
        .expect("expected BinaryExpr");
    let deserialized_sq_expr = binary_expr
        .right()
        .downcast_ref::<ScalarSubqueryExpr>()
        .expect("expected ScalarSubqueryExpr");

    assert!(
        ScalarSubqueryResults::ptr_eq(exec_results, deserialized_sq_expr.results()),
        "ScalarSubqueryExpr should share the same results container as ScalarSubqueryExec"
    );
    Ok(())
}

/// Verify that nested ScalarSubqueryExec nodes deserialize with distinct
/// scoped results containers, and that each ScalarSubqueryExpr is wired to the
/// container for its own surrounding ScalarSubqueryExec.
#[test]
fn roundtrip_nested_scalar_subquery_exec_scopes_results() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let subquery_schema =
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));

    let inner_results = ScalarSubqueryResults::new(1);
    let inner_sq_expr = Arc::new(ScalarSubqueryExpr::new(
        DataType::Int64,
        true,
        SubqueryIndex::new(0),
        inner_results.clone(),
    ));
    let inner_predicate =
        binary(col("a", &schema)?, Operator::Eq, inner_sq_expr, &schema)?;
    let inner_filter = Arc::new(FilterExec::try_new(
        inner_predicate,
        Arc::new(EmptyExec::new(schema.clone())),
    )?);
    let inner_exec: Arc<dyn ExecutionPlan> = Arc::new(ScalarSubqueryExec::new(
        inner_filter,
        vec![ScalarSubqueryLink {
            plan: Arc::new(EmptyExec::new(subquery_schema.clone())),
            index: SubqueryIndex::new(0),
        }],
        inner_results,
    ));

    let outer_results = ScalarSubqueryResults::new(1);
    let outer_sq_expr = Arc::new(ScalarSubqueryExpr::new(
        DataType::Int64,
        true,
        SubqueryIndex::new(0),
        outer_results.clone(),
    ));
    let outer_predicate =
        binary(col("a", &schema)?, Operator::Eq, outer_sq_expr, &schema)?;
    let outer_filter = Arc::new(FilterExec::try_new(outer_predicate, inner_exec)?);
    let outer_exec: Arc<dyn ExecutionPlan> = Arc::new(ScalarSubqueryExec::new(
        outer_filter,
        vec![ScalarSubqueryLink {
            plan: Arc::new(EmptyExec::new(subquery_schema)),
            index: SubqueryIndex::new(0),
        }],
        outer_results,
    ));

    let bytes = datafusion_proto::bytes::physical_plan_to_bytes(Arc::clone(&outer_exec))?;
    let ctx = SessionContext::new();
    let deserialized = datafusion_proto::bytes::physical_plan_from_bytes(
        bytes.as_ref(),
        ctx.task_ctx().as_ref(),
    )?;

    let outer_exec = deserialized
        .downcast_ref::<ScalarSubqueryExec>()
        .expect("expected outer ScalarSubqueryExec");
    let outer_results = outer_exec.results();
    let outer_filter = outer_exec
        .input()
        .downcast_ref::<FilterExec>()
        .expect("expected outer FilterExec");
    let outer_binary = outer_filter
        .predicate()
        .downcast_ref::<BinaryExpr>()
        .expect("expected outer BinaryExpr");
    let outer_sq_expr = outer_binary
        .right()
        .downcast_ref::<ScalarSubqueryExpr>()
        .expect("expected outer ScalarSubqueryExpr");

    let inner_exec = outer_filter
        .input()
        .downcast_ref::<ScalarSubqueryExec>()
        .expect("expected inner ScalarSubqueryExec");
    let inner_results = inner_exec.results();
    let inner_filter = inner_exec
        .input()
        .downcast_ref::<FilterExec>()
        .expect("expected inner FilterExec");
    let inner_binary = inner_filter
        .predicate()
        .downcast_ref::<BinaryExpr>()
        .expect("expected inner BinaryExpr");
    let inner_sq_expr = inner_binary
        .right()
        .downcast_ref::<ScalarSubqueryExpr>()
        .expect("expected inner ScalarSubqueryExpr");

    assert!(
        ScalarSubqueryResults::ptr_eq(outer_results, outer_sq_expr.results()),
        "outer ScalarSubqueryExpr should use outer ScalarSubqueryExec results"
    );
    assert!(
        ScalarSubqueryResults::ptr_eq(inner_results, inner_sq_expr.results()),
        "inner ScalarSubqueryExpr should use inner ScalarSubqueryExec results"
    );
    assert!(
        !ScalarSubqueryResults::ptr_eq(outer_results, inner_results),
        "nested ScalarSubqueryExec nodes should not share results containers"
    );
    assert!(
        !ScalarSubqueryResults::ptr_eq(outer_results, inner_sq_expr.results()),
        "inner ScalarSubqueryExpr must not read from outer results"
    );
    assert!(
        !ScalarSubqueryResults::ptr_eq(inner_results, outer_sq_expr.results()),
        "outer ScalarSubqueryExpr must not read from inner results"
    );

    Ok(())
}

/// Verify that the default physical plan bytes round-trip preserves executable
/// scalar subquery plans.
#[tokio::test]
async fn roundtrip_scalar_subquery_exec_with_default_converter_executes() -> Result<()> {
    let ctx = SessionContext::new();
    let sql = "SELECT x + (SELECT max(y) FROM (VALUES (10), (20)) AS u(y)) AS s \
               FROM (VALUES (2), (1)) AS t(x) \
               ORDER BY s";

    let initial_plan = ctx.sql(sql).await?.create_physical_plan().await?;
    assert!(
        format!("{initial_plan:?}").contains("ScalarSubqueryExec"),
        "expected ScalarSubqueryExec in plan:\n{initial_plan:?}"
    );

    let bytes =
        datafusion_proto::bytes::physical_plan_to_bytes(Arc::clone(&initial_plan))?;
    let roundtripped = datafusion_proto::bytes::physical_plan_from_bytes(
        bytes.as_ref(),
        ctx.task_ctx().as_ref(),
    )?;
    assert!(
        format!("{roundtripped:?}").contains("ScalarSubqueryExec"),
        "expected ScalarSubqueryExec after roundtrip:\n{roundtripped:?}"
    );

    let batches = datafusion::physical_plan::common::collect(
        roundtripped.execute(0, ctx.task_ctx())?,
    )
    .await?;
    datafusion::assert_batches_eq!(
        &["+----+", "| s  |", "+----+", "| 21 |", "| 22 |", "+----+",],
        &batches
    );

    Ok(())
}

/// Verify that built-in protobuf round-tripping preserves scalar-subquery
/// dynamic-filter bindings, including the shared filter instance used by the
/// input plan, and that execution updates that filter.
#[tokio::test]
async fn roundtrip_scalar_subquery_dynamic_filter_binding_executes() -> Result<()> {
    use datafusion::physical_plan::PhysicalExpr;
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;

    let ctx = SessionContext::new();
    let sql = "SELECT x FROM (VALUES (TIMESTAMP '2023-01-01 00:00:00'), \
                       (TIMESTAMP '2023-01-03 00:00:00')) AS t(x) \
               WHERE x >= (SELECT max(y) FROM \
                       (VALUES (TIMESTAMP '2023-01-02 00:00:00')) AS u(y)) \
                       - INTERVAL '0' DAY";
    let initial_plan = ctx.sql(sql).await?.create_physical_plan().await?;

    initial_plan.gather_filters_for_pushdown(
        datafusion::physical_plan::filter_pushdown::FilterPushdownPhase::Post,
        vec![],
        ctx.state().config_options(),
    )?;
    let scalar_exec = initial_plan
        .downcast_ref::<ScalarSubqueryExec>()
        .expect("expected ScalarSubqueryExec");
    let binding_filter = scalar_exec.dynamic_filter_bindings()[0].1.clone();
    // Two actual input branches consume the same optimized filter instance.
    // The default converter decodes these occurrences independently while
    // retaining their shared expression_id.
    let input_with_consumers = UnionExec::try_new(vec![
        Arc::new(FilterExec::try_new(
            Arc::clone(&binding_filter),
            Arc::clone(scalar_exec.input()),
        )?) as Arc<dyn ExecutionPlan>,
        Arc::new(FilterExec::try_new(
            binding_filter,
            Arc::clone(scalar_exec.input()),
        )?) as Arc<dyn ExecutionPlan>,
    ])?;
    let mut children = vec![input_with_consumers as Arc<dyn ExecutionPlan>];
    children.extend(
        scalar_exec
            .subqueries()
            .iter()
            .map(|link| Arc::clone(&link.plan)),
    );
    let initial_plan = initial_plan.replace_children(
        children,
        datafusion::physical_plan::ReplaceChildrenOptions::new(
            datafusion::physical_plan::ChildrenPropertiesMode::Recompute,
        ),
    )?;
    let mut initial_bindings = 0;
    initial_plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<ScalarSubqueryExec>() {
            let produced = exec.dynamic_expressions_produced();
            if !produced.is_empty() {
                initial_bindings += produced.len();
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    assert_eq!(
        initial_bindings, 1,
        "expected one scalar dynamic-filter binding"
    );

    let bytes =
        datafusion_proto::bytes::physical_plan_to_bytes(Arc::clone(&initial_plan))?;
    let roundtripped = datafusion_proto::bytes::physical_plan_from_bytes(
        bytes.as_ref(),
        ctx.task_ctx().as_ref(),
    )?;

    let mut roundtripped_bindings = 0;
    roundtripped.apply(|node| {
        if let Some(exec) = node.downcast_ref::<ScalarSubqueryExec>() {
            let produced = exec.dynamic_expressions_produced();
            if produced.is_empty() {
                return Ok(TreeNodeRecursion::Continue);
            }
            assert_eq!(produced.len(), 1);
            let binding = exec.dynamic_filter_bindings();
            assert_eq!(binding.len(), 1);
            let bound_filter = &binding[0].1;
            let bound_filter_expr = bound_filter
                .downcast_ref::<DynamicFilterPhysicalExpr>()
                .expect("binding should contain a dynamic filter");
            assert_eq!(
                Some(bound_filter_expr.expression_id().unwrap()),
                produced[0].expression_id()
            );

            let mut consumers = vec![];
            exec.input().apply(|child| {
                child.apply_expressions(&mut |expr| {
                    if expr.expression_id() == produced[0].expression_id()
                        && expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some()
                    {
                        consumers.push(Arc::clone(expr));
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                Ok(TreeNodeRecursion::Continue)
            })?;
            assert_eq!(consumers.len(), 2);
            assert!(!Arc::ptr_eq(&consumers[0], &consumers[1]));
            assert!(Arc::ptr_eq(bound_filter, &consumers[0]));
            roundtripped_bindings += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    assert_eq!(roundtripped_bindings, 1);

    let batches =
        datafusion::physical_plan::collect_partitioned(roundtripped, ctx.task_ctx())
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
    datafusion::assert_batches_eq!(
        &[
            "+---------------------+",
            "| x                   |",
            "+---------------------+",
            "| 2023-01-03T00:00:00 |",
            "| 2023-01-03T00:00:00 |",
            "+---------------------+"
        ],
        &batches
    );
    Ok(())
}
