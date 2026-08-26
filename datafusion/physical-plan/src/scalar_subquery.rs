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

//! Execution plan for uncorrelated scalar subqueries.
//!
//! [`ScalarSubqueryExec`] wraps a main input plan and a set of subquery plans.
//! At execution time, it runs each subquery exactly once, extracts the scalar
//! result, and populates a shared [`ScalarSubqueryResults`] container that
//! [`ScalarSubqueryExpr`] instances hold directly and read from by index.
//!
//! [`ScalarSubqueryExpr`]: datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr

use std::fmt;
use std::sync::{Arc, Mutex};

use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    DataFusionError, Result, ScalarValue, Statistics, exec_err, internal_err,
};
use datafusion_execution::TaskContext;
use datafusion_expr::physical_planning_context::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal, lit,
};
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_expr_common::physical_expr::{
    PhysicalExpr, snapshot_physical_expr,
};

use crate::execution_plan::{
    CardinalityEffect, ExecutionPlan, PlanProperties, plan_contains_expression_id,
};
use crate::filter_pushdown::{
    ChildFilterDescription, ChildPushdownResult, FilterDescription, FilterPushdownPhase,
    FilterPushdownPropagation, PushedDown,
};
use crate::joins::utils::{OnceAsync, OnceFut};
use crate::statistics::{ChildStats, StatisticsArgs};
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ReplaceChildrenOptions,
    SendableRecordBatchStream,
};

use futures::StreamExt;
use futures::TryStreamExt;

/// Links a scalar subquery's execution plan to its index in the shared results
/// container. The [`ScalarSubqueryExec`] that owns these links populates
/// `results[index]` at execution time, and [`ScalarSubqueryExpr`] instances
/// with the same index read from it.
///
/// [`ScalarSubqueryExpr`]: datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr
#[derive(Debug, Clone)]
pub struct ScalarSubqueryLink {
    /// The physical plan for the subquery.
    pub plan: Arc<dyn ExecutionPlan>,
    /// Index into the shared results container.
    pub index: SubqueryIndex,
}

/// Manages execution of uncorrelated scalar subqueries for a single plan
/// level.
///
/// From a query-results perspective, this node is a pass-through: it yields
/// the same batches as its main input and exists only to populate scalar
/// subquery results as a side effect before those batches are produced.
///
/// The first child node is the **main input plan**, whose batches are passed
/// through unchanged. The remaining children are **subquery plans**, each of
/// which must produce exactly zero or one row. Before any batches from the main
/// input are yielded, all subquery plans are executed and their scalar results
/// are stored in a shared [`ScalarSubqueryResults`] container owned by this
/// node. [`ScalarSubqueryExpr`] nodes embedded in the main input's expressions
/// hold the same container and read from it by index.
///
/// All subqueries are evaluated eagerly when the first output partition is
/// requested, before any rows from the main input are produced.
///
/// TODO: Consider overlapping computation of the subqueries with evaluating the
/// main query.
///
/// [`ScalarSubqueryExpr`]: datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr
#[derive(Debug)]
pub struct ScalarSubqueryExec {
    /// The main input plan whose output is passed through.
    input: Arc<dyn ExecutionPlan>,
    /// Subquery plans and their result indexes.
    subqueries: Vec<ScalarSubqueryLink>,
    /// Shared one-time async computation of subquery results.
    subquery_future: Arc<OnceAsync<()>>,
    /// Shared results container; the corresponding `ScalarSubqueryExpr`
    /// nodes in the input plan hold the same underlying container.
    results: ScalarSubqueryResults,
    /// Dynamic filters associated with predicates in the main input.
    /// This state is separate from the input so it survives child replacement.
    bindings: Arc<Mutex<Vec<ScalarSubqueryBinding>>>,
    /// Cached plan properties (copied from input).
    cache: Arc<PlanProperties>,
}

#[derive(Debug, Clone)]
struct ScalarSubqueryBinding {
    /// The predicate produced by this scalar subquery.
    predicate: Arc<dyn PhysicalExpr>,
    /// Every concrete dynamic-filter consumer for this binding's ID.
    ///
    /// Optimized plans commonly share one filter instance between the
    /// producer and its consumers. Proto decoding, however, can produce
    /// multiple independent filter instances with the same expression ID.
    consumers: Vec<Arc<dyn PhysicalExpr>>,
}

impl ScalarSubqueryExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        subqueries: Vec<ScalarSubqueryLink>,
        results: ScalarSubqueryResults,
    ) -> Self {
        let cache = Arc::clone(input.properties());
        Self {
            input,
            subqueries,
            subquery_future: Arc::default(),
            results,
            bindings: Arc::default(),
            cache,
        }
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn subqueries(&self) -> &[ScalarSubqueryLink] {
        &self.subqueries
    }

    pub fn results(&self) -> &ScalarSubqueryResults {
        &self.results
    }

    /// Returns a per-child bool vec that is `true` for the main input
    /// (child 0) and `false` for every subquery child.
    fn true_for_input_only(&self) -> Vec<bool> {
        std::iter::once(true)
            .chain(std::iter::repeat_n(false, self.subqueries.len()))
            .collect()
    }

    fn discover_binding(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
    ) -> Option<ScalarSubqueryBinding> {
        let binary = predicate.downcast_ref::<BinaryExpr>()?;
        if *binary.op() != datafusion_expr::Operator::GtEq {
            return None;
        }
        let left = binary.left().downcast_ref::<Column>()?;
        let subtraction = binary.right().downcast_ref::<BinaryExpr>()?;
        if *subtraction.op() != datafusion_expr::Operator::Minus
            || subtraction.right().downcast_ref::<Literal>().is_none()
        {
            return None;
        }
        let scalar = subtraction.left().downcast_ref::<ScalarSubqueryExpr>()?;
        if !ScalarSubqueryResults::ptr_eq(scalar.results(), &self.results) {
            return None;
        }
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(left.clone()) as Arc<dyn PhysicalExpr>],
            lit(true),
        ));
        Some(ScalarSubqueryBinding {
            predicate: Arc::clone(predicate),
            consumers: vec![filter as Arc<dyn PhysicalExpr>],
        })
    }

    fn discover_bindings(&self) -> Result<()> {
        let mut discovered = Vec::new();
        self.input.apply(|plan| {
            plan.apply_expressions(&mut |root| {
                root.apply(&mut |expr: &Arc<dyn PhysicalExpr>| {
                    if let Some(binding) = self.discover_binding(expr) {
                        discovered.push(binding);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                Ok(TreeNodeRecursion::Continue)
            })?;
            Ok(TreeNodeRecursion::Continue)
        })?;
        let mut bindings = self.bindings.lock().unwrap();
        for binding in discovered {
            if !bindings
                .iter()
                .any(|existing| Arc::ptr_eq(&existing.predicate, &binding.predicate))
            {
                bindings.push(binding);
            }
        }
        Ok(())
    }
}

impl DisplayAs for ScalarSubqueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ScalarSubqueryExec: subqueries={}",
                    self.subqueries.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "")
            }
        }
    }
}

impl ExecutionPlan for ScalarSubqueryExec {
    fn name(&self) -> &'static str {
        "ScalarSubqueryExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.input];
        for sq in &self.subqueries {
            children.push(&sq.plan);
        }
        children
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // First child is the main input, the rest are subquery plans.
        let input = children.remove(0);
        let subqueries = self
            .subqueries
            .iter()
            .zip(children)
            .map(|(sq, new_plan)| ScalarSubqueryLink {
                plan: new_plan,
                index: sq.index,
            })
            .collect();
        let mut new_node =
            ScalarSubqueryExec::new(input, subqueries, self.results.clone());
        new_node.bindings = Arc::clone(&self.bindings);
        Ok(Arc::new(new_node))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        // DynamicFilterPhysicalExpr currently has no supported reset operation.
        // Do not silently reuse a completed filter on a later execution.
        if !self.bindings.lock().unwrap().is_empty() {
            return internal_err!(
                "cannot reset ScalarSubqueryExec with dynamic-filter bindings: DynamicFilterPhysicalExpr has no reset API"
            );
        }
        self.results.clear();
        Ok(Arc::new(ScalarSubqueryExec {
            input: Arc::clone(&self.input),
            subqueries: self.subqueries.clone(),
            subquery_future: Arc::default(),
            results: self.results.clone(),
            bindings: Arc::clone(&self.bindings),
            cache: Arc::clone(&self.cache),
        }))
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        if phase == FilterPushdownPhase::Post {
            self.discover_bindings()?;
        }
        let mut main = ChildFilterDescription::from_child(&parent_filters, self.input())?;
        if phase == FilterPushdownPhase::Post {
            for binding in self.bindings.lock().unwrap().iter() {
                main = main.with_self_filter(Arc::clone(&binding.consumers[0]));
            }
        }
        let mut description = FilterDescription::new().with_child(main);
        for subquery in &self.subqueries {
            description = description
                .with_child(ChildFilterDescription::all_unsupported(&parent_filters));
            let _ = subquery;
        }
        Ok(description)
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let result = FilterPushdownPropagation::if_any(child_pushdown_result.clone());
        if phase == FilterPushdownPhase::Post {
            let accepted = child_pushdown_result
                .self_filters
                .first()
                .into_iter()
                .flatten()
                .filter(|predicate| matches!(predicate.discriminant, PushedDown::Yes))
                .filter_map(|predicate| predicate.predicate.expression_id())
                .collect::<std::collections::HashSet<_>>();
            let mut bindings = self.bindings.lock().unwrap();
            let mut retained = Vec::with_capacity(bindings.len());
            for binding in bindings.drain(..) {
                let keep = match binding.consumers[0].expression_id() {
                    Some(id) => {
                        accepted.contains(&id)
                            || plan_contains_expression_id(&self.input, id)?
                    }
                    None => false,
                };
                if keep {
                    retained.push(binding);
                }
            }
            *bindings = retained;
        }
        Ok(result)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let subqueries = self.subqueries.clone();
        let results = self.results.clone();
        let bindings = Arc::clone(&self.bindings);
        let planning_ctx = Arc::clone(&context);
        let mut subquery_future = self.subquery_future.try_once(move || {
            Ok(async move {
                execute_subqueries(subqueries, results, bindings, planning_ctx).await
            })
        })?;
        let input = Arc::clone(&self.input);
        let schema = self.schema();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move {
                // Execute all subqueries exactly once, even when multiple
                // partitions call execute() concurrently.
                wait_for_subqueries(&mut subquery_future).await?;

                // Now that the subqueries have finished execution, we can
                // safely execute the main input
                input.execute(partition, context)
            })
            .try_flatten(),
        )))
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let bindings = self.bindings.lock().unwrap();
        crate::apply_expression_roots(
            bindings.iter().flat_map(|binding| {
                [
                    Arc::clone(&binding.predicate),
                    Arc::clone(&binding.consumers[0]),
                ]
            }),
            f,
        )
    }

    fn dynamic_expressions_produced(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.bindings
            .lock()
            .unwrap()
            .iter()
            .map(|binding| Arc::clone(&binding.consumers[0]))
            .collect()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Only the main input (first child); subquery children don't contribute
        // to ordering.
        self.true_for_input_only()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // ScalarSubqueryExec is a pass-through coordinator: it does not
        // benefit from repartitioning any child directly below it.
        vec![false; self.subqueries.len() + 1]
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        // Only `self.input` (child 0) is used; the subqueries are skipped.
        let mut requests = vec![ChildStats::Skip; 1 + self.subqueries.len()];
        requests[0] = ChildStats::At(partition);
        requests
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        Ok(Arc::clone(&input_stats[0]))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        ctx: &crate::proto::ExecutionPlanEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalPlanNode>> {
        use datafusion_proto_models::protobuf;

        let ScalarSubqueryExec {
            input: input_plan,
            subqueries,
            subquery_future: _,
            results: _,
            bindings,
            cache: _,
        } = self;
        let input = ctx.encode_child(input_plan)?;
        // Subquery indices are positional and recovered during decoding.
        let subqueries =
            ctx.encode_children(subqueries.iter().map(|subquery| &subquery.plan))?;
        let dynamic_filter_bindings = bindings
            .lock()
            .unwrap()
            .iter()
            .map(|binding| {
                Ok(protobuf::ScalarSubqueryDynamicFilterBindingNode {
                    predicate: Some(ctx.encode_expr(&binding.predicate)?),
                    dynamic_filter_id: Some(
                        binding.consumers[0].expression_id().ok_or_else(|| {
                            DataFusionError::Internal(
                                "ScalarSubquery dynamic filter is missing expression_id"
                                    .to_string(),
                            )
                        })?,
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(protobuf::PhysicalPlanNode {
            physical_plan_type: Some(
                protobuf::physical_plan_node::PhysicalPlanType::ScalarSubquery(Box::new(
                    protobuf::ScalarSubqueryExecNode {
                        input: Some(Box::new(input)),
                        subqueries,
                        dynamic_filter_bindings,
                    },
                )),
            ),
        }))
    }
}

#[cfg(feature = "proto")]
impl ScalarSubqueryExec {
    /// Reconstruct a [`ScalarSubqueryExec`] from its protobuf representation.
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalPlanNode,
        ctx: &crate::proto::ExecutionPlanDecodeCtx<'_>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_proto_models::protobuf;

        let scalar_subquery = crate::expect_plan_variant!(
            node,
            protobuf::physical_plan_node::PhysicalPlanType::ScalarSubquery,
            "ScalarSubqueryExec",
        );
        let results = ScalarSubqueryResults::new(scalar_subquery.subqueries.len());
        let input_node = scalar_subquery.input.as_deref().ok_or_else(|| {
            DataFusionError::Internal(
                "ScalarSubqueryExec is missing required field 'input'".to_string(),
            )
        })?;
        let input =
            ctx.decode_child_with_scalar_subquery_results(input_node, results.clone())?;

        let mut dynamic_filters =
            std::collections::HashMap::<u64, Vec<Arc<dyn PhysicalExpr>>>::new();
        input.apply(|plan| {
            plan.apply_expressions(&mut |root| {
                root.apply(&mut |expr: &Arc<dyn PhysicalExpr>| {
                    if let Some(filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>()
                    {
                        if let Some(id) = filter.expression_id() {
                            dynamic_filters
                                .entry(id)
                                .or_default()
                                .push(Arc::clone(expr));
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
            })?;
            Ok(TreeNodeRecursion::Continue)
        })?;

        let mut bindings =
            Vec::with_capacity(scalar_subquery.dynamic_filter_bindings.len());
        for binding in &scalar_subquery.dynamic_filter_bindings {
            let predicate = ctx.decode_expr_with_scalar_subquery_results(
                binding.predicate.as_ref().ok_or_else(|| {
                    DataFusionError::Internal(
                        "ScalarSubqueryDynamicFilterBindingNode is missing required field 'predicate'".to_string(),
                    )
                })?,
                input.schema().as_ref(),
                results.clone(),
            )?;
            let id = binding.dynamic_filter_id.ok_or_else(|| {
                DataFusionError::Internal(
                    "ScalarSubqueryDynamicFilterBindingNode is missing required field 'dynamic_filter_id'".to_string(),
                )
            })?;
            let filters = dynamic_filters.get(&id).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "ScalarSubquery dynamic filter binding references missing expression_id {id}"
                ))
            })?;
            bindings.push(ScalarSubqueryBinding {
                predicate,
                consumers: filters.clone(),
            });
        }

        let subqueries = scalar_subquery
            .subqueries
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                Ok(ScalarSubqueryLink {
                    plan: ctx.decode_child(plan)?,
                    index: SubqueryIndex::new(index),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let exec = Self::new(input, subqueries, results);
        exec.bindings.lock().unwrap().extend(bindings);
        Ok(Arc::new(exec))
    }
}

/// Wait for the subquery execution future to complete.
async fn wait_for_subqueries(fut: &mut OnceFut<()>) -> Result<()> {
    std::future::poll_fn(|cx| fut.get_shared(cx)).await?;
    Ok(())
}

async fn execute_subqueries(
    subqueries: Vec<ScalarSubqueryLink>,
    results: ScalarSubqueryResults,
    bindings: Arc<Mutex<Vec<ScalarSubqueryBinding>>>,
    context: Arc<TaskContext>,
) -> Result<()> {
    // Evaluate subqueries in parallel; wait for them all to finish evaluation
    // before returning.
    let futures = subqueries.iter().map(|sq| {
        let plan = Arc::clone(&sq.plan);
        let ctx = Arc::clone(&context);
        let results = results.clone();
        let index = sq.index;
        async move {
            let value = execute_scalar_subquery(plan, ctx).await?;
            results.set(index, value)?;
            Ok(()) as Result<()>
        }
    });
    futures::future::try_join_all(futures).await?;
    let bindings = bindings.lock().unwrap();
    for binding in bindings.iter() {
        // Snapshot the producer exactly once, then publish it to every
        // independent runtime state. Derived consumers sharing one state are
        // updated and completed only once.
        let snapshot = snapshot_physical_expr(Arc::clone(&binding.predicate))?;
        let mut updated = Vec::<&DynamicFilterPhysicalExpr>::new();
        for consumer in &binding.consumers {
            let filter = consumer
                .downcast_ref::<DynamicFilterPhysicalExpr>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "ScalarSubquery dynamic-filter binding has an invalid filter"
                            .to_string(),
                    )
                })?;
            if !updated
                .iter()
                .any(|updated_filter| updated_filter.shares_runtime_state(filter))
            {
                filter.update(Arc::clone(&snapshot))?;
                filter.mark_complete();
                updated.push(filter);
            }
        }
    }
    Ok(())
}

/// Execute a single subquery plan and extract the scalar value.
/// Returns NULL for 0 rows, the scalar value for exactly 1 row,
/// or an error for >1 rows.
async fn execute_scalar_subquery(
    plan: Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
) -> Result<ScalarValue> {
    let schema = plan.schema();
    if schema.fields().len() != 1 {
        // Should be enforced by the physical planner.
        return internal_err!(
            "Scalar subquery must return exactly one column, got {}",
            schema.fields().len()
        );
    }

    let mut stream = crate::execute_stream(plan, context)?;
    let mut result: Option<ScalarValue> = None;

    while let Some(batch) = stream.next().await.transpose()? {
        if batch.num_rows() == 0 {
            continue;
        }
        if result.is_some() || batch.num_rows() > 1 {
            return exec_err!("Scalar subquery returned more than one row");
        }
        result = Some(ScalarValue::try_from_array(batch.column(0), 0)?);
    }

    // 0 rows → typed NULL per SQL semantics
    match result {
        Some(v) => Ok(v),
        None => ScalarValue::try_from(schema.field(0).data_type()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{self, TestMemoryExec};
    use crate::{
        execution_plan::reset_plan_states,
        projection::{ProjectionExec, ProjectionExpr},
    };

    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::test::exec::ErrorExec;
    use arrow::array::{Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, IntervalMonthDayNano, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;

    enum ExpectedSubqueryResult {
        Value(ScalarValue),
        Error(&'static str),
    }

    #[derive(Debug)]
    struct ExpressionExec {
        input: Arc<dyn ExecutionPlan>,
        expressions: Arc<Mutex<Vec<Arc<dyn PhysicalExpr>>>>,
    }

    impl ExpressionExec {
        fn new(
            input: Arc<dyn ExecutionPlan>,
            expressions: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Self {
            Self {
                input,
                expressions: Arc::new(Mutex::new(expressions)),
            }
        }

        fn add_expression(&self, expression: Arc<dyn PhysicalExpr>) {
            self.expressions.lock().unwrap().push(expression);
        }
    }

    impl DisplayAs for ExpressionExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            match t {
                DisplayFormatType::Default | DisplayFormatType::Verbose => {
                    write!(f, "ExpressionExec")
                }
                DisplayFormatType::TreeRender => write!(f, ""),
            }
        }
    }

    impl ExecutionPlan for ExpressionExec {
        fn name(&self) -> &'static str {
            "ExpressionExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }

        fn replace_children(
            self: Arc<Self>,
            mut children: Vec<Arc<dyn ExecutionPlan>>,
            _: ReplaceChildrenOptions,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self {
                input: children.remove(0),
                expressions: Arc::clone(&self.expressions),
            }))
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            self.replace_children(
                children,
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.input.execute(partition, context)
        }

        fn apply_expressions(
            &self,
            f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            let expressions = self.expressions.lock().unwrap().clone();
            crate::apply_expression_roots(expressions, f)
        }
    }

    #[derive(Debug)]
    struct CountingExec {
        inner: Arc<dyn ExecutionPlan>,
        execute_calls: Arc<AtomicUsize>,
    }

    impl CountingExec {
        fn new(inner: Arc<dyn ExecutionPlan>, execute_calls: Arc<AtomicUsize>) -> Self {
            Self {
                inner,
                execute_calls,
            }
        }
    }

    impl DisplayAs for CountingExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            match t {
                DisplayFormatType::Default | DisplayFormatType::Verbose => {
                    write!(f, "CountingExec")
                }
                DisplayFormatType::TreeRender => write!(f, ""),
            }
        }
    }

    impl ExecutionPlan for CountingExec {
        fn name(&self) -> &'static str {
            "CountingExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.inner]
        }

        fn replace_children(
            self: Arc<Self>,
            mut children: Vec<Arc<dyn ExecutionPlan>>,
            _: ReplaceChildrenOptions,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(
                children.remove(0),
                Arc::clone(&self.execute_calls),
            )))
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            self.replace_children(
                children,
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.execute_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.execute(partition, context)
        }
    }

    fn make_subquery_plan(batches: Vec<RecordBatch>) -> Arc<dyn ExecutionPlan> {
        let schema = batches[0].schema();
        TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap()
    }

    fn int32_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    fn empty_int64_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![] as Vec<i64>))])
            .unwrap()
    }

    fn placeholder_input() -> Arc<dyn ExecutionPlan> {
        Arc::new(crate::placeholder_row::PlaceholderRowExec::new(
            test::aggr_test_schema(),
        ))
    }

    fn single_subquery_exec(
        input: Arc<dyn ExecutionPlan>,
        subquery_plan: Arc<dyn ExecutionPlan>,
        results: ScalarSubqueryResults,
    ) -> ScalarSubqueryExec {
        ScalarSubqueryExec::new(
            input,
            vec![ScalarSubqueryLink {
                plan: subquery_plan,
                index: SubqueryIndex::new(0),
            }],
            results,
        )
    }

    fn scalar_subquery_projection_input(
        results: ScalarSubqueryResults,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ProjectionExec::try_new(
            vec![ProjectionExpr {
                expr: Arc::new(ScalarSubqueryExpr::new(
                    DataType::Int32,
                    false,
                    SubqueryIndex::new(0),
                    results,
                )),
                alias: "sq".to_string(),
            }],
            placeholder_input(),
        )?))
    }

    fn extract_single_int32_value(batches: &[RecordBatch]) -> i32 {
        assert_eq!(batches.len(), 1);
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.len(), 1);
        values.value(0)
    }

    fn timestamp_predicate(results: ScalarSubqueryResults) -> Arc<dyn PhysicalExpr> {
        let scalar = Arc::new(ScalarSubqueryExpr::new(
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
            SubqueryIndex::new(0),
            results,
        ));
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("timestamp_column", 0)),
            Operator::GtEq,
            Arc::new(BinaryExpr::new(
                scalar,
                Operator::Minus,
                Arc::new(Literal::new(ScalarValue::IntervalMonthDayNano(Some(
                    IntervalMonthDayNano::new(0, 0, 60_000_000_000),
                )))),
            )),
        ))
    }

    fn dynamic_filter(exec: &ScalarSubqueryExec) -> Arc<DynamicFilterPhysicalExpr> {
        let filter = Arc::clone(&exec.bindings.lock().unwrap()[0].consumers[0]);
        Arc::downcast::<DynamicFilterPhysicalExpr>(filter)
            .expect("expected dynamic filter")
    }

    #[test]
    fn test_scalar_subquery_filter_pushdown_no_binding_retention() -> Result<()> {
        for (name, input_contains_filter, expected_bindings) in [
            ("removes_missing_filter", false, 0),
            ("retains_filter_in_rewritten_input", true, 1),
        ] {
            let results = ScalarSubqueryResults::new(1);
            let predicate = timestamp_predicate(results.clone());
            let input = Arc::new(ExpressionExec::new(
                placeholder_input(),
                vec![Arc::clone(&predicate)],
            ));
            let exec = Arc::new(single_subquery_exec(
                input,
                make_subquery_plan(vec![int32_batch(vec![1])]),
                results,
            ));
            exec.discover_bindings()?;
            let filter = dynamic_filter(&exec);
            let mut expressions =
                vec![predicate, Arc::clone(&filter) as Arc<dyn PhysicalExpr>];
            if !input_contains_filter {
                expressions.pop();
            }
            let updated_input =
                Arc::new(ExpressionExec::new(placeholder_input(), expressions));
            let subquery = Arc::clone(&exec.subqueries()[0].plan);
            let updated_exec = exec.replace_children(
                vec![updated_input, subquery],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?;

            updated_exec.handle_child_pushdown_result(
                FilterPushdownPhase::Post,
                ChildPushdownResult {
                    parent_filters: vec![],
                    self_filters: vec![vec![PushedDown::No
                        .wrap_expression(Arc::clone(&filter) as Arc<dyn PhysicalExpr>)]],
                },
                &ConfigOptions::default(),
            )?;
            assert_eq!(
                updated_exec.dynamic_expressions_produced().len(),
                expected_bindings,
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_scalar_subquery_filter_pushdown_retains_accepted_binding() -> Result<()> {
        let results = ScalarSubqueryResults::new(1);
        let predicate = timestamp_predicate(results.clone());
        let input = Arc::new(ExpressionExec::new(
            placeholder_input(),
            vec![Arc::clone(&predicate)],
        ));
        let exec = single_subquery_exec(
            input,
            make_subquery_plan(vec![int32_batch(vec![1])]),
            results,
        );
        exec.discover_bindings()?;
        let filter = dynamic_filter(&exec);

        exec.handle_child_pushdown_result(
            FilterPushdownPhase::Post,
            ChildPushdownResult {
                parent_filters: vec![],
                self_filters: vec![vec![
                    PushedDown::Yes
                        .wrap_expression(Arc::clone(&filter) as Arc<dyn PhysicalExpr>),
                ]],
            },
            &ConfigOptions::default(),
        )?;

        assert!(reset_plan_states(Arc::new(exec)).is_err());
        Ok(())
    }

    #[test]
    fn test_discover_scalar_subquery_bindings_by_results_scope() -> Result<()> {
        let results = ScalarSubqueryResults::new(1);
        let predicate = timestamp_predicate(results.clone());
        let input = Arc::new(ExpressionExec::new(
            placeholder_input(),
            vec![Arc::clone(&predicate)],
        ));
        let exec = single_subquery_exec(
            input.clone(),
            make_subquery_plan(vec![int32_batch(vec![1])]),
            results,
        );

        exec.discover_bindings()?;
        exec.discover_bindings()?;
        assert_eq!(exec.bindings.lock().unwrap().len(), 1);

        input.add_expression(timestamp_predicate(exec.results().clone()));
        exec.discover_bindings()?;
        assert_eq!(exec.bindings.lock().unwrap().len(), 2);

        input.add_expression(timestamp_predicate(ScalarSubqueryResults::new(1)));
        exec.discover_bindings()?;
        assert_eq!(exec.bindings.lock().unwrap().len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_subqueries_updates_shared_runtime_state_once() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let original_column =
            Arc::new(Column::new_with_schema("a", &schema)?) as Arc<dyn PhysicalExpr>;
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&original_column)],
            lit(true),
        ));
        let derived_expr_1 = Arc::clone(&filter)
            .with_new_children(vec![Arc::new(Column::new_with_schema("b", &schema)?)])?;
        let derived_expr_2 = Arc::clone(&filter)
            .with_new_children(vec![Arc::new(Column::new_with_schema("c", &schema)?)])?;
        let derived_1 = derived_expr_1
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .ok_or_else(|| {
                DataFusionError::Internal("expected dynamic filter".to_string())
            })?;
        let derived_2 = derived_expr_2
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .ok_or_else(|| {
                DataFusionError::Internal("expected dynamic filter".to_string())
            })?;
        assert!(filter.shares_runtime_state(derived_1));
        assert!(filter.shares_runtime_state(derived_2));

        let bindings = Arc::new(Mutex::new(vec![ScalarSubqueryBinding {
            predicate: Arc::clone(&original_column),
            consumers: vec![Arc::clone(&derived_expr_1), Arc::clone(&derived_expr_2)],
        }]));
        execute_subqueries(
            vec![],
            ScalarSubqueryResults::new(0),
            bindings,
            Arc::new(TaskContext::default()),
        )
        .await?;

        assert_eq!(filter.snapshot_generation(), 2);
        assert_eq!(derived_1.snapshot_generation(), 2);
        assert_eq!(derived_2.snapshot_generation(), 2);
        assert_eq!(
            derived_1
                .current()?
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            derived_2
                .current()?
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            2
        );
        for (filter, message) in [
            (filter.as_ref(), "shared filter should be complete"),
            (derived_1, "first derived filter should be complete"),
            (derived_2, "second derived filter should be complete"),
        ] {
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                filter.wait_complete(),
            )
            .await
            .expect(message);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_scalar_subquery_updates_dynamic_filter_with_timestamp() -> Result<()> {
        let results = ScalarSubqueryResults::new(1);
        let predicate = timestamp_predicate(results.clone());
        let input = Arc::new(ExpressionExec::new(placeholder_input(), vec![predicate]));
        let exec = single_subquery_exec(
            input,
            make_subquery_plan(vec![int32_batch(vec![1])]),
            results.clone(),
        );
        exec.discover_bindings()?;
        results.set(
            SubqueryIndex::new(0),
            ScalarValue::TimestampMillisecond(
                Some(1_672_574_400_000),
                Some("UTC".into()),
            ),
        )?;

        execute_subqueries(
            vec![],
            results,
            Arc::clone(&exec.bindings),
            Arc::new(TaskContext::default()),
        )
        .await?;

        let filter = dynamic_filter(&exec);
        let current = filter.current()?;
        let predicate = current.downcast_ref::<BinaryExpr>().unwrap();
        let subtraction = predicate.right().downcast_ref::<BinaryExpr>().unwrap();
        let timestamp = subtraction.left().downcast_ref::<Literal>().unwrap();
        assert_eq!(
            timestamp.value(),
            &ScalarValue::TimestampMillisecond(
                Some(1_672_574_400_000),
                Some("UTC".into())
            )
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), filter.wait_complete())
            .await
            .expect("scalar subquery dynamic filter should be complete");
        Ok(())
    }

    #[tokio::test]
    async fn test_failed_scalar_subquery_does_not_update_dynamic_filter() -> Result<()> {
        let results = ScalarSubqueryResults::new(1);
        let predicate = timestamp_predicate(results.clone());
        let input = Arc::new(ExpressionExec::new(placeholder_input(), vec![predicate]));
        let exec = single_subquery_exec(input, Arc::new(ErrorExec::new()), results);
        exec.discover_bindings()?;
        let filter = dynamic_filter(&exec);
        let before = filter.current()?;
        assert!(before.downcast_ref::<Literal>().is_some());

        let result = execute_subqueries(
            exec.subqueries().to_vec(),
            exec.results().clone(),
            Arc::clone(&exec.bindings),
            Arc::new(TaskContext::default()),
        )
        .await;
        assert!(result.is_err());
        assert!(filter.current()?.downcast_ref::<Literal>().is_some());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(50),
                filter.wait_complete()
            )
            .await
            .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_scalar_subquery_row_count_semantics() -> Result<()> {
        for (name, plan, expected) in [
            (
                "single_row",
                make_subquery_plan(vec![int32_batch(vec![42])]),
                ExpectedSubqueryResult::Value(ScalarValue::Int32(Some(42))),
            ),
            (
                "zero_rows",
                make_subquery_plan(vec![empty_int64_batch()]),
                ExpectedSubqueryResult::Value(ScalarValue::Int64(None)),
            ),
            (
                "multiple_rows",
                make_subquery_plan(vec![int32_batch(vec![1, 2, 3])]),
                ExpectedSubqueryResult::Error("more than one row"),
            ),
        ] {
            let actual =
                execute_scalar_subquery(plan, Arc::new(TaskContext::default())).await;
            match expected {
                ExpectedSubqueryResult::Value(expected) => {
                    assert_eq!(actual?, expected, "{name}");
                }
                ExpectedSubqueryResult::Error(expected) => {
                    let err = actual.expect_err(name);
                    assert!(
                        err.to_string().contains(expected),
                        "{name}: expected error containing '{expected}', got {err}"
                    );
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_failed_subquery_is_not_retried() -> Result<()> {
        let execute_calls = Arc::new(AtomicUsize::new(0));
        let subquery_plan = Arc::new(CountingExec::new(
            Arc::new(ErrorExec::new()),
            Arc::clone(&execute_calls),
        ));
        let exec = single_subquery_exec(
            placeholder_input(),
            subquery_plan,
            ScalarSubqueryResults::new(1),
        );

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, Arc::clone(&ctx))?;
        assert!(crate::common::collect(stream).await.is_err());

        let stream = exec.execute(0, ctx)?;
        assert!(crate::common::collect(stream).await.is_err());

        assert_eq!(execute_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_reset_state_clears_results_and_reexecutes_subqueries() -> Result<()> {
        let execute_calls = Arc::new(AtomicUsize::new(0));
        let results = ScalarSubqueryResults::new(1);
        let subquery_plan = Arc::new(CountingExec::new(
            make_subquery_plan(vec![int32_batch(vec![42])]),
            Arc::clone(&execute_calls),
        ));
        let exec: Arc<dyn ExecutionPlan> = Arc::new(single_subquery_exec(
            scalar_subquery_projection_input(results.clone())?,
            subquery_plan,
            results.clone(),
        ));

        let batches =
            crate::common::collect(exec.execute(0, Arc::new(TaskContext::default()))?)
                .await?;
        assert_eq!(extract_single_int32_value(&batches), 42);
        assert_eq!(
            results.get(SubqueryIndex::new(0)),
            Some(ScalarValue::Int32(Some(42)))
        );

        let reset_exec = reset_plan_states(Arc::clone(&exec))?;
        assert_eq!(results.get(SubqueryIndex::new(0)), None);

        let reset_batches = crate::common::collect(
            reset_exec.execute(0, Arc::new(TaskContext::default()))?,
        )
        .await?;
        assert_eq!(extract_single_int32_value(&reset_batches), 42);
        assert_eq!(
            results.get(SubqueryIndex::new(0)),
            Some(ScalarValue::Int32(Some(42)))
        );
        assert_eq!(execute_calls.load(Ordering::SeqCst), 2);

        Ok(())
    }
}
