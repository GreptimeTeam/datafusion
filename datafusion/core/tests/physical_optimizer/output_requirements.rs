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

use std::sync::Arc;

use crate::physical_optimizer::test_utils::{parquet_exec, schema, sort_exec, sort_expr};

use arrow::array::{cast::AsArray, record_batch, types::Int32Type};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::prelude::SessionContext;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::execution_props::{ScalarSubqueryResults, SubqueryIndex};
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion_physical_optimizer::output_requirements::OutputRequirements;
use datafusion_physical_plan::scalar_subquery::{ScalarSubqueryExec, ScalarSubqueryLink};
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{ExecutionPlan, collect, displayable};

#[test]
fn require_top_ordering_descends_through_scalar_subquery() {
    let s = schema();
    let ordering: LexOrdering = [sort_expr("a", &s)].into();
    let sort = sort_exec(ordering, parquet_exec(Arc::clone(&s)));
    let subqueries = vec![ScalarSubqueryLink {
        plan: parquet_exec(Arc::clone(&s)),
        index: SubqueryIndex::new(0),
    }];
    let plan = Arc::new(ScalarSubqueryExec::new(
        sort,
        subqueries,
        ScalarSubqueryResults::new(1),
    )) as Arc<dyn ExecutionPlan>;

    let optimized = OutputRequirements::new_add_mode()
        .optimize(plan, &ConfigOptions::new())
        .expect("add-mode optimize should succeed");

    insta::assert_snapshot!(
        displayable(optimized.as_ref()).indent(true).to_string(),
        @r"
    ScalarSubqueryExec: subqueries=1
      OutputRequirementExec: order_by=[(a@0, asc)], dist_by=SinglePartition
        SortExec: expr=[a@0 ASC], preserve_partitioning=[false]
          DataSourceExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], file_type=parquet
      DataSourceExec: file_groups={1 group: [[x]]}, projection=[a, b, c, d, e], file_type=parquet
    "
    );
}

#[tokio::test]
async fn scalar_subquery_root_preserves_global_ordering_end_to_end() {
    let p1 = record_batch!(("a", Int32, [1, 3, 5, 7])).expect("build partition 1 batch");
    let p2 = record_batch!(("a", Int32, [2, 4, 6, 8])).expect("build partition 2 batch");
    let schema = p1.schema();
    let ordering: LexOrdering = [sort_expr("a", &schema)].into();
    let source = DataSourceExec::from_data_source(
        MemorySourceConfig::try_new(&[vec![p1], vec![p2]], Arc::clone(&schema), None)
            .expect("build memory source config")
            .try_with_sort_information(vec![ordering.clone()])
            .expect("attach sort information to source"),
    );
    let main_input = Arc::new(SortPreservingMergeExec::new(ordering, source));

    let sq_batch = record_batch!(("v", Int32, [42])).expect("build subquery batch");
    let subquery = MemorySourceConfig::try_new_exec(
        &[vec![sq_batch.clone()]],
        sq_batch.schema(),
        None,
    )
    .expect("build subquery exec");
    let plan = Arc::new(ScalarSubqueryExec::new(
        main_input,
        vec![ScalarSubqueryLink {
            plan: subquery,
            index: SubqueryIndex::new(0),
        }],
        ScalarSubqueryResults::new(1),
    )) as Arc<dyn ExecutionPlan>;

    let mut config = ConfigOptions::new();
    config.execution.target_partitions = 4;
    let mut optimized = plan;
    for rule in PhysicalOptimizer::new().rules {
        optimized = rule
            .optimize(optimized, &config)
            .unwrap_or_else(|e| panic!("optimizer rule {} failed: {e}", rule.name()));
    }

    let batches = collect(optimized, SessionContext::new().task_ctx())
        .await
        .expect("execute optimized plan");
    let values: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8]);
}
