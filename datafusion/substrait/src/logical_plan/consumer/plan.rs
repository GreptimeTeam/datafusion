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

use super::utils::{make_renamed_schema, rename_expressions};
use super::{DefaultSubstraitConsumer, SubstraitConsumer};
use crate::extensions::Extensions;
use datafusion::common::{not_impl_err, plan_err};
use datafusion::execution::SessionState;
use datafusion::logical_expr::{Aggregate, LogicalPlan, Projection, col};
use std::sync::Arc;
use substrait::proto::{Plan, plan_rel};

fn rename_with_outer_projection(
    plan: LogicalPlan,
    renamed_schema: &datafusion::common::DFSchema,
) -> datafusion::common::Result<LogicalPlan> {
    Ok(LogicalPlan::Projection(Projection::try_new(
        rename_expressions(
            plan.schema().columns().iter().map(|c| col(c.to_owned())),
            plan.schema(),
            renamed_schema.iter(),
        )?,
        Arc::new(plan),
    )?))
}

/// Convert Substrait Plan to DataFusion LogicalPlan
pub async fn from_substrait_plan(
    state: &SessionState,
    plan: &Plan,
) -> datafusion::common::Result<LogicalPlan> {
    // Register function extension
    let extensions = Extensions::try_from(&plan.extensions)?;
    if !extensions.type_variations.is_empty() {
        return not_impl_err!("Type variation extensions are not supported");
    }

    let consumer = DefaultSubstraitConsumer::new(&extensions, state);
    from_substrait_plan_with_consumer(&consumer, plan).await
}

/// Convert Substrait Plan to DataFusion LogicalPlan using the given consumer
pub async fn from_substrait_plan_with_consumer(
    consumer: &impl SubstraitConsumer,
    plan: &Plan,
) -> datafusion::common::Result<LogicalPlan> {
    match plan.relations.len() {
        1 => {
            match plan.relations[0].rel_type.as_ref() {
                Some(rt) => match rt {
                    plan_rel::RelType::Rel(rel) => Ok(consumer.consume_rel(rel).await?),
                    plan_rel::RelType::Root(root) => {
                        let plan =
                            consumer.consume_rel(root.input.as_ref().unwrap()).await?;
                        if root.names.is_empty() {
                            // Backwards compatibility for plans missing names
                            return Ok(plan);
                        }
                        let renamed_schema =
                            make_renamed_schema(plan.schema(), &root.names)?;
                        if renamed_schema
                            .has_equivalent_names_and_types(plan.schema())
                            .is_ok()
                            && renamed_schema.iter().zip(plan.schema().iter()).all(
                                |((renamed_qualifier, _), (plan_qualifier, _))| {
                                    renamed_qualifier == plan_qualifier
                                },
                            )
                        {
                            // Nothing to do if the schema is already equivalent
                            return Ok(plan);
                        }
                        match plan {
                            // If the last node of the plan produces expressions, bake the renames into those expressions.
                            // This isn't necessary for correctness, but helps with roundtrip tests.
                            LogicalPlan::Projection(p) => {
                                Ok(LogicalPlan::Projection(Projection::try_new(
                                    rename_expressions(
                                        p.expr,
                                        p.input.schema(),
                                        renamed_schema.iter(),
                                    )?,
                                    p.input,
                                )?))
                            }
                            LogicalPlan::Aggregate(a) => {
                                let group_expr_len = a.group_expr.len();
                                let group_output_len = a.group_expr_len()?;
                                if group_output_len == group_expr_len {
                                    let new_group_exprs = rename_expressions(
                                        a.group_expr,
                                        a.input.schema(),
                                        renamed_schema.iter().take(group_output_len),
                                    )?;
                                    let new_aggr_exprs = rename_expressions(
                                        a.aggr_expr,
                                        a.input.schema(),
                                        renamed_schema.iter().skip(group_output_len),
                                    )?;
                                    Ok(LogicalPlan::Aggregate(Aggregate::try_new(
                                        a.input,
                                        new_group_exprs,
                                        new_aggr_exprs,
                                    )?))
                                } else {
                                    rename_with_outer_projection(
                                        LogicalPlan::Aggregate(a),
                                        &renamed_schema,
                                    )
                                }
                            }
                            // There are probably more plans where we could bake things in, can add them later as needed.
                            // Otherwise, add a new Project to handle the renaming.
                            _ => rename_with_outer_projection(plan, &renamed_schema),
                        }
                    }
                },
                None => plan_err!("Cannot parse plan relation: None"),
            }
        }
        _ => not_impl_err!(
            "Substrait plan with more than 1 relation trees not supported. Number of relation trees: {:?}",
            plan.relations.len()
        ),
    }
}
