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

use std::collections::HashSet;
use std::sync::Arc;

use super::{ParquetAccessPlan, ParquetFileMetrics, RowGroupAccess};
use crate::bloom_filter::BloomFilterStatistics;
use crate::metadata::{has_untrusted_byte_array_stats, has_untrusted_min_max_order};
use arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use arrow::compute::nullif;
use arrow::datatypes::Schema;
use datafusion_common::pruning::PruningStatistics;
use datafusion_common::{Column, Result, ScalarValue};
use datafusion_datasource::FileRange;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, IsNullExpr, NotExpr};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprSimplifier};
use datafusion_pruning::{PruningPredicate, PruningPredicateBuilder};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::basic::ColumnOrder;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use parquet::schema::types::SchemaDescriptor;

/// Reduces the [`ParquetAccessPlan`] based on row group level metadata.
///
/// This struct implements the various types of pruning that are applied to a
/// set of row groups within a parquet file, progressively narrowing down the
/// set of row groups (and ranges/selections within those row groups) that
/// should be scanned, based on the available metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RowGroupAccessPlanFilter {
    /// which row groups should be accessed
    access_plan: ParquetAccessPlan,
}

impl RowGroupAccessPlanFilter {
    /// Create a new `RowGroupPlanBuilder` for pruning out the groups to scan
    /// based on metadata and statistics
    pub fn new(access_plan: ParquetAccessPlan) -> Self {
        Self { access_plan }
    }

    /// Return true if there are no row groups
    pub fn is_empty(&self) -> bool {
        self.access_plan.is_empty()
    }

    /// Return the number of row groups that are currently expected to be scanned
    pub fn remaining_row_group_count(&self) -> usize {
        self.access_plan.row_group_index_iter().count()
    }

    /// Return indexes of row groups that still need to be scanned.
    pub fn row_group_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        self.access_plan.row_group_index_iter()
    }

    /// Returns the inner access plan.
    pub fn build(self) -> ParquetAccessPlan {
        self.access_plan
    }

    /// Returns a reference to the inner access plan.
    ///
    /// Test-only accessor used by the shared assertion helpers in
    /// [`crate::test_util`].
    #[cfg(test)]
    pub(crate) fn access_plan(&self) -> &ParquetAccessPlan {
        &self.access_plan
    }

    /// Returns the is_fully_matched vector.
    pub fn is_fully_matched(&self) -> &Vec<bool> {
        self.access_plan.fully_matched()
    }

    /// Prunes the access plan based on the limit and fully contained row groups.
    ///
    /// The pruning works by leveraging the concept of fully matched row groups. Consider a query like:
    /// `WHERE species LIKE 'Alpine%' AND s >= 50 LIMIT N`
    ///
    /// After initial filtering, row groups can be classified into three states:
    ///
    /// 1. Not Matching / Pruned
    /// 2. Partially Matching (Row Group/Page contains some matches)
    /// 3. Fully Matching (Entire range is within predicate)
    ///
    /// +-----------------------------------------------------------------------+
    /// |                            NOT MATCHING                               |
    /// |  Row group 1                                                          |
    /// |  +-----------------------------------+-----------------------------+  |
    /// |  | SPECIES                           | S                           |  |
    /// |  +-----------------------------------+-----------------------------+  |
    /// |  | Snow Vole                         | 7                           |  |
    /// |  | Brown Bear                        | 133 ✅                      |  |
    /// |  | Gray Wolf                         | 82  ✅                      |  |
    /// |  +-----------------------------------+-----------------------------+  |
    /// +-----------------------------------------------------------------------+
    ///
    /// +---------------------------------------------------------------------------+
    /// |                          PARTIALLY MATCHING                               |
    /// |                                                                           |
    /// |  Row group 2                              Row group 4                     |
    /// |  +------------------+--------------+      +------------------+----------+ |
    /// |  | SPECIES          | S            |      | SPECIES          | S        | |
    /// |  +------------------+--------------+      +------------------+----------+ |
    /// |  | Lynx             | 71 ✅        |      | Europ. Mole      | 4        | |
    /// |  | Red Fox          | 40           |      | Polecat          | 16       | |
    /// |  | Alpine Bat  ✅   | 6            |      | Alpine Ibex ✅  | 97 ✅    | |
    /// |  +------------------+--------------+      +------------------+----------+ |
    /// +---------------------------------------------------------------------------+
    ///
    /// +-----------------------------------------------------------------------+
    /// |                           FULLY MATCHING                              |
    /// |  Row group 3                                                          |
    /// |  +-----------------------------------+-----------------------------+  |
    /// |  | SPECIES                           | S                           |  |
    /// |  +-----------------------------------+-----------------------------+  |
    /// |  | Alpine Ibex  ✅                  | 101    ✅                   |  |
    /// |  | Alpine Goat  ✅                  | 76     ✅                   |  |
    /// |  | Alpine Sheep ✅                  | 83     ✅                   |  |
    /// |  +-----------------------------------+-----------------------------+  |
    /// +-----------------------------------------------------------------------+
    ///
    /// ### Identification of Fully Matching Row Groups
    ///
    /// DataFusion identifies row groups where ALL rows satisfy the filter by inverting the
    /// predicate and checking if statistics prove the inverted version is false for the group.
    ///
    /// For example, prefix matches like `species LIKE 'Alpine%'` are pruned using ranges:
    /// 1. Candidate Range: `species >= 'Alpine' AND species < 'Alpinf'`
    /// 2. Inverted Condition (to prove full match): `species < 'Alpine' OR species >= 'Alpinf'`
    /// 3. Statistical Evaluation (check if any row *could* satisfy the inverted condition):
    ///    `min < 'Alpine' OR max >= 'Alpinf'`
    ///
    /// If this evaluation is **false**, it proves no row can fail the original filter,
    /// so the row group is **FULLY MATCHING**.
    ///
    /// ### Impact of Statistics Truncation
    ///
    /// The precision of pruning depends on the metadata quality. Truncated statistics
    /// may prevent the system from proving a full match.
    ///
    /// **Example**: `WHERE species LIKE 'Alpine%'` (Target range: `['Alpine', 'Alpinf')`)
    ///
    /// | Truncation Length | min / max           | Inverted Evaluation                                                 | Status                 |
    /// |-------------------|---------------------|---------------------------------------------------------------------|------------------------|
    /// | **Length 6**      | `Alpine` / `Alpine` | `"Alpine" < "Alpine" (F) OR "Alpine" >= "Alpinf" (F)` -> **false**  | **FULLY MATCHING**     |
    /// | **Length 3**      | `Alp` / `Alq`       | `"Alp" < "Alpine" (T) OR "Alq" >= "Alpinf" (T)` -> **true**         | **PARTIALLY MATCHING** |
    ///
    /// Even though Row Group 3 only contains matching rows, truncation to length 3 makes
    /// the statistics `[Alp, Alq]` too broad to prove it (they could include "Alpha").
    /// The system must conservatively scan the group.
    ///
    /// Without limit pruning: Scan Partition 2 → Partition 3 → Partition 4 (until limit reached)
    /// With limit pruning: If Partition 3 contains enough rows to satisfy the limit,
    /// skip Partitions 2 and 4 entirely and go directly to Partition 3.
    ///
    /// This optimization is particularly effective when:
    /// - The limit is small relative to the total dataset size
    /// - There are row groups that are fully matched by the filter predicates
    /// - The fully matched row groups contain sufficient rows to satisfy the limit
    ///
    /// For more information, see the [paper](https://arxiv.org/pdf/2504.11540)'s "Pruning for LIMIT Queries" part
    pub fn prune_by_limit(
        &mut self,
        limit: usize,
        rg_metadata: &[RowGroupMetaData],
        metrics: &ParquetFileMetrics,
    ) {
        let mut fully_matched_row_group_indexes: Vec<usize> = Vec::new();
        let mut fully_matched_rows_count: usize = 0;

        // Iterate through the currently accessible row groups and try to
        // find a set of matching row groups that can satisfy the limit
        for &idx in self.access_plan.row_group_indexes().iter() {
            if self.access_plan.is_fully_matched(idx) {
                let row_group_row_count = match &self.access_plan.inner()[idx] {
                    RowGroupAccess::Skip => continue,
                    RowGroupAccess::Scan => rg_metadata[idx].num_rows() as usize,
                    RowGroupAccess::Selection(selection) => selection.row_count(),
                };
                fully_matched_row_group_indexes.push(idx);
                fully_matched_rows_count += row_group_row_count;
                if fully_matched_rows_count >= limit {
                    break;
                }
            }
        }

        // If we can satisfy the limit with fully matching row groups,
        // rewrite the plan to do so
        if fully_matched_rows_count >= limit {
            let original_num_accessible_row_groups =
                self.access_plan.row_group_indexes().len();
            let new_num_accessible_row_groups = fully_matched_row_group_indexes.len();
            let pruned_count = original_num_accessible_row_groups
                .saturating_sub(new_num_accessible_row_groups);
            metrics.limit_pruned_row_groups.add_pruned(pruned_count);

            let mut new_access_plan = ParquetAccessPlan::new_none(rg_metadata.len());
            for &idx in &fully_matched_row_group_indexes {
                new_access_plan.set(idx, self.access_plan.inner()[idx].clone());
                new_access_plan.mark_fully_matched(idx);
            }
            self.access_plan = new_access_plan;
        }
    }

    /// Prune remaining row groups to only those  within the specified range.
    ///
    /// Updates this set to mark row groups that should not be scanned
    ///
    /// # Panics
    /// if `groups.len() != self.len()`
    pub fn prune_by_range(&mut self, groups: &[RowGroupMetaData], range: &FileRange) {
        assert_eq!(groups.len(), self.access_plan.len());
        for (idx, metadata) in groups.iter().enumerate() {
            if !self.access_plan.should_scan(idx) {
                continue;
            }

            // Skip the row group if the first dictionary/data page are not
            // within the range.
            //
            // note don't use the location of metadata
            // <https://github.com/apache/datafusion/issues/5995>
            let col = metadata.column(0);
            let offset = col
                .dictionary_page_offset()
                .unwrap_or_else(|| col.data_page_offset());
            if !range.contains(offset) {
                self.access_plan.skip(idx);
            }
        }
    }
    /// Prune remaining row groups using min/max/null_count statistics and
    /// the [`PruningPredicate`] to determine if the predicate can not be true.
    ///
    /// Updates this set to mark row groups that should not be scanned
    ///
    /// This method has no file footer, so it cannot establish the sort order
    /// of string or binary min/max statistics. Such bounds are ignored. Use
    /// [`Self::prune_by_statistics_with_metadata`] when the full metadata is
    /// available. Null counts and other columns' statistics remain usable.
    ///
    /// # Panics
    /// if `groups.len() != self.len()`
    pub fn prune_by_statistics(
        &mut self,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
        groups: &[RowGroupMetaData],
        predicate: &PruningPredicate,
        metrics: &ParquetFileMetrics,
    ) {
        self.prune_by_statistics_inner(
            arrow_schema,
            parquet_schema,
            groups,
            None,
            predicate,
            metrics,
        );
    }

    /// Prune row groups using statistics and the comparison orders recorded
    /// in the Parquet file footer.
    ///
    /// Unlike [`Self::prune_by_statistics`], this method can use string and
    /// binary bounds when their ordering is known to match Arrow's ordering.
    ///
    /// # Panics
    /// if `metadata.num_row_groups() != self.len()`
    pub fn prune_by_statistics_with_metadata(
        &mut self,
        arrow_schema: &Schema,
        metadata: &ParquetMetaData,
        predicate: &PruningPredicate,
        metrics: &ParquetFileMetrics,
    ) {
        let file_metadata = metadata.file_metadata();
        self.prune_by_statistics_inner(
            arrow_schema,
            file_metadata.schema_descr(),
            metadata.row_groups(),
            file_metadata.column_orders().map(Vec::as_slice),
            predicate,
            metrics,
        );
    }

    fn prune_by_statistics_inner(
        &mut self,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
        groups: &[RowGroupMetaData],
        column_orders: Option<&[ColumnOrder]>,
        predicate: &PruningPredicate,
        metrics: &ParquetFileMetrics,
    ) {
        // scoped timer updates on drop
        let _timer_guard = metrics.statistics_eval_time.timer();

        assert_eq!(groups.len(), self.access_plan.len());
        // Indexes of row groups still to scan
        let row_group_indexes = self.access_plan.row_group_indexes();
        let row_group_metadatas = row_group_indexes
            .iter()
            .map(|&i| &groups[i])
            .collect::<Vec<_>>();

        let pruning_stats = RowGroupPruningStatistics {
            parquet_schema,
            column_orders,
            row_group_metadatas,
            arrow_schema,
            // Preserve the existing row-group pruning behavior. This path only
            // proves whether matching rows may exist, so it uses the
            // StatisticsConverter default for older parquet-rs files where a
            // missing null count can mean there are zero nulls.
            missing_null_counts_as_zero: true,
        };

        // try to prune the row groups in a single call
        match predicate.prune(&pruning_stats) {
            Ok(values) => {
                let mut fully_contained_candidates_original_idx: Vec<usize> = Vec::new();
                for (idx, &value) in row_group_indexes.iter().zip(values.iter()) {
                    if !value {
                        self.access_plan.skip(*idx);
                        metrics.row_groups_pruned_statistics.add_pruned(1);
                    } else {
                        metrics.row_groups_pruned_statistics.add_matched(1);
                        fully_contained_candidates_original_idx.push(*idx);
                    }
                }

                // Check if any of the matched row groups are fully contained by the predicate
                self.identify_fully_matched_row_groups(
                    &fully_contained_candidates_original_idx,
                    &pruning_stats,
                    groups,
                    predicate,
                    metrics,
                );
            }
            // stats filter array could not be built, so we can't prune
            Err(e) => {
                log::debug!("Error evaluating row group predicate values {e}");
                metrics.predicate_evaluation_errors.add(1);
            }
        }
    }

    /// Identifies row groups that are fully matched by the predicate.
    ///
    /// This optimization checks whether all rows in a row group satisfy the predicate
    /// by inverting the predicate and checking if it prunes the row group. If the
    /// inverted predicate prunes a row group, it means no rows match the inverted
    /// predicate, which implies all rows match the original predicate.
    ///
    /// Note: This optimization is relatively inexpensive for a limited number of row groups.
    fn identify_fully_matched_row_groups(
        &mut self,
        candidate_row_group_indices: &[usize],
        pruning_stats: &RowGroupPruningStatistics<'_>,
        groups: &[RowGroupMetaData],
        predicate: &PruningPredicate,
        metrics: &ParquetFileMetrics,
    ) {
        if candidate_row_group_indices.is_empty() {
            return;
        }
        // Some pruning rewrites preserve only whether rows can be TRUE, while
        // full-match inference must distinguish FALSE from UNKNOWN.
        if !predicate.can_be_inverted_for_full_match() {
            return;
        }
        let arrow_schema = pruning_stats.arrow_schema;

        let mut inverted_expr: Arc<dyn PhysicalExpr> =
            Arc::new(NotExpr::new(Arc::clone(predicate.orig_expr())));

        // Rows where the predicate evaluates to NULL do not pass the filter.
        // Include NULL checks in the inverted expression so a row group is only
        // considered fully matched when every referenced column is known non-null.
        // This is conservative for null-accepting predicates, but fully matched
        // row groups must not have false positives.
        let mut columns = collect_columns(predicate.orig_expr())
            .into_iter()
            .filter(|column| arrow_schema.field(column.index()).is_nullable())
            .collect::<Vec<_>>();
        columns.sort_by(|a, b| {
            a.index()
                .cmp(&b.index())
                .then_with(|| a.name().cmp(b.name()))
        });

        for column in columns {
            inverted_expr = Arc::new(BinaryExpr::new(
                inverted_expr,
                Operator::Or,
                Arc::new(IsNullExpr::new(Arc::new(column))),
            ));
        }

        // Simplify the inverted expression (e.g., NOT(c1 = 0) -> c1 != 0)
        // before building the pruning predicate
        let simplifier = PhysicalExprSimplifier::new(arrow_schema);
        let Ok(inverted_expr) = simplifier.simplify(inverted_expr) else {
            return;
        };

        let Ok(inverted_predicate) = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(predicate.schema()))
            .with_max_in_list_size(predicate.max_in_list_size())
            .try_build(inverted_expr)
        else {
            return;
        };
        // The inverse can introduce filter-only rewrites that were hidden
        // inside an unhandled expression in the forward predicate. In that
        // case pruning UNKNOWN cannot establish that every row matches.
        if !inverted_predicate.can_be_inverted_for_full_match() {
            return;
        }

        let inverted_pruning_stats = RowGroupPruningStatistics {
            parquet_schema: pruning_stats.parquet_schema,
            column_orders: pruning_stats.column_orders,
            row_group_metadatas: candidate_row_group_indices
                .iter()
                .map(|&i| &groups[i])
                .collect::<Vec<_>>(),
            arrow_schema,
            // Fully matched row groups require a stronger proof: every row
            // must pass the predicate. Missing null counts are unknown here;
            // treating them as zero can incorrectly mark nullable row groups as
            // fully matched and make limit pruning unsound.
            missing_null_counts_as_zero: false,
        };

        let Ok(inverted_values) = inverted_predicate.prune(&inverted_pruning_stats)
        else {
            return;
        };

        for (i, &original_row_group_idx) in candidate_row_group_indices.iter().enumerate()
        {
            // If the inverted predicate *also* prunes this row group (meaning inverted_values[i] is false),
            // it implies that *all* rows in this group satisfy the original predicate.
            if !inverted_values[i] {
                self.access_plan.mark_fully_matched(original_row_group_idx);
                metrics.row_groups_pruned_statistics.add_fully_matched(1);
            }
        }
    }

    /// Prune remaining row groups using loaded bloom filters and the
    /// [`PruningPredicate`].
    ///
    /// Updates this set with row groups that should not be scanned.
    /// `row_group_bloom_filters[idx]` contains the bloom filters for the
    /// parquet row group at index `idx`.
    ///
    /// # Panics
    /// if `row_group_bloom_filters` does not have the same number of row groups as this set
    pub fn prune_by_bloom_filters(
        &mut self,
        predicate: &PruningPredicate,
        metrics: &ParquetFileMetrics,
        row_group_bloom_filters: &[BloomFilterStatistics],
    ) {
        // scoped timer updates on drop
        let _timer_guard = metrics.bloom_filter_eval_time.timer();

        assert_eq!(row_group_bloom_filters.len(), self.access_plan.len());
        for (idx, stats) in row_group_bloom_filters.iter().enumerate() {
            if !self.access_plan.should_scan(idx) {
                continue;
            }

            // Can this group be pruned?
            let prune_group = match predicate.prune(stats) {
                Ok(values) => !values[0],
                Err(e) => {
                    log::debug!(
                        "Error evaluating row group predicate on bloom filter: {e}"
                    );
                    metrics.predicate_evaluation_errors.add(1);
                    false
                }
            };

            if prune_group {
                metrics.row_groups_pruned_bloom_filter.add_pruned(1);
                self.access_plan.skip(idx)
            } else {
                metrics.row_groups_pruned_bloom_filter.add_matched(1);
            }
        }
    }
}

/// Wraps a slice of [`RowGroupMetaData`] in a way that implements [`PruningStatistics`].
///
/// Visible to sibling modules so runtime row-group pruners (e.g. the dynamic
/// TopK pruner in `push_decoder.rs`) can reuse this adapter without
/// duplicating the statistics-to-`PruningStatistics` plumbing.
pub(crate) struct RowGroupPruningStatistics<'a> {
    pub(crate) parquet_schema: &'a SchemaDescriptor,
    pub(crate) column_orders: Option<&'a [ColumnOrder]>,
    pub(crate) row_group_metadatas: Vec<&'a RowGroupMetaData>,
    pub(crate) arrow_schema: &'a Schema,
    pub(crate) missing_null_counts_as_zero: bool,
}

impl<'a> RowGroupPruningStatistics<'a> {
    /// Return an iterator over the row group metadata
    fn metadata_iter(&self) -> impl Iterator<Item = &'a RowGroupMetaData> + '_ {
        self.row_group_metadatas.iter().copied()
    }

    fn statistics_converter(&self, column: &Column) -> Result<StatisticsConverter<'a>> {
        Ok(StatisticsConverter::try_new(
            &column.name,
            self.arrow_schema,
            self.parquet_schema,
        )?
        .with_missing_null_counts_as_zero(self.missing_null_counts_as_zero))
    }

    fn min_max_statistics_converter(
        &self,
        column: &Column,
    ) -> Option<StatisticsConverter<'a>> {
        let converter = self.statistics_converter(column).ok()?;
        let parquet_index = converter.parquet_column_index();
        if parquet_index.is_some_and(|index| {
            has_untrusted_min_max_order(self.parquet_schema, self.column_orders, index)
        }) {
            return None;
        }
        Some(converter)
    }

    fn mask_untrusted_byte_array_stats(
        &self,
        parquet_index: Option<usize>,
        values: ArrayRef,
    ) -> Option<ArrayRef> {
        if !has_untrusted_byte_array_stats(
            self.parquet_schema,
            parquet_index,
            self.metadata_iter(),
        ) {
            return Some(values);
        }
        // A file may mix legacy and modern row-group statistics. Keep the
        // modern bounds usable rather than discarding this entire column.
        let mask = BooleanArray::from_iter(self.metadata_iter().map(|group| {
            has_untrusted_byte_array_stats(self.parquet_schema, parquet_index, [group])
        }));
        nullif(values.as_ref(), &mask).ok()
    }
}

impl PruningStatistics for RowGroupPruningStatistics<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        let converter = self.min_max_statistics_converter(column)?;
        let values = converter.row_group_mins(self.metadata_iter()).ok()?;
        self.mask_untrusted_byte_array_stats(converter.parquet_column_index(), values)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        let converter = self.min_max_statistics_converter(column)?;
        let values = converter.row_group_maxes(self.metadata_iter()).ok()?;
        self.mask_untrusted_byte_array_stats(converter.parquet_column_index(), values)
    }

    fn num_containers(&self) -> usize {
        self.row_group_metadatas.len()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.statistics_converter(column)
            .and_then(|c| Ok(c.row_group_null_counts(self.metadata_iter())?))
            .ok()
            .map(|counts| Arc::new(counts) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        // Row counts are container-level — read directly from row group metadata.
        let counts: UInt64Array = self
            .metadata_iter()
            .map(|rg| Some(rg.num_rows() as u64))
            .collect();
        Some(Arc::new(counts) as ArrayRef)
    }

    fn contained(
        &self,
        _column: &Column,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Rem;

    use super::*;
    use crate::test_util::ExpectedPruning;

    use arrow::datatypes::DataType::Decimal128;
    use arrow::datatypes::{DataType, Field};
    use datafusion_expr::{cast, col, lit};
    use datafusion_physical_expr::planner::logical2physical;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use parquet::arrow::ArrowSchemaConverter;
    use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
    use parquet::basic::LogicalType;
    use parquet::data_type::{ByteArray, FixedLenByteArray};
    use parquet::file::metadata::ColumnChunkMetaData;
    use parquet::{
        basic::Type as PhysicalType, file::statistics::Statistics as ParquetStatistics,
        schema::types::SchemaDescPtr,
    };

    fn build_test_pruning_predicate(
        expr: Arc<dyn PhysicalExpr>,
        schema: Arc<Schema>,
    ) -> PruningPredicate {
        PruningPredicateBuilder::new()
            .with_file_schema(schema)
            .try_build(expr)
            .unwrap()
    }

    struct PrimitiveTypeField {
        name: &'static str,
        physical_ty: PhysicalType,
        logical_ty: Option<LogicalType>,
        precision: Option<i32>,
        scale: Option<i32>,
        byte_len: Option<i32>,
    }

    impl PrimitiveTypeField {
        fn new(name: &'static str, physical_ty: PhysicalType) -> Self {
            Self {
                name,
                physical_ty,
                logical_ty: None,
                precision: None,
                scale: None,
                byte_len: None,
            }
        }

        fn with_logical_type(mut self, logical_type: LogicalType) -> Self {
            self.logical_ty = Some(logical_type);
            self
        }

        fn with_precision(mut self, precision: i32) -> Self {
            self.precision = Some(precision);
            self
        }

        fn with_scale(mut self, scale: i32) -> Self {
            self.scale = Some(scale);
            self
        }

        fn with_byte_len(mut self, byte_len: i32) -> Self {
            self.byte_len = Some(byte_len);
            self
        }
    }

    /// Exercise a real join producer through the pruning snapshot boundary,
    /// and the independently owned bitmap used by a detached pruning expr.
    #[tokio::test]
    async fn pruning_pipeline_retains_join_producer_and_detached_bitmap() -> Result<()> {
        use arrow::array::Int64Array;
        use arrow::record_batch::RecordBatch;
        use datafusion_common::pruning::PrunableStatistics;
        use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
        use datafusion_common::{JoinType, NullEquality, Statistics};
        use datafusion_datasource::memory::MemorySourceConfig;
        use datafusion_execution::TaskContext;
        use datafusion_execution::config::SessionConfig;
        use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;
        use datafusion_physical_expr::PhysicalExprRef;
        use datafusion_physical_expr::expressions::{
            Column as PhysicalColumn, DynamicFilterPhysicalExpr, InListExpr, Literal,
            lit as physical_lit,
        };
        use datafusion_physical_plan::ExecutionPlan;
        use datafusion_physical_plan::common::collect;
        use datafusion_physical_plan::filter::FilterExecBuilder;
        use datafusion_physical_plan::joins::{
            HashJoinExec, HashTableLookupExpr, PartitionMode,
        };

        for struct_in_list in [true, false] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]));
            let batch = |keys: Vec<i64>, values: Vec<i64>| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(keys)),
                        Arc::new(Int64Array::from(values)),
                    ],
                )
            };
            let build = batch(vec![0, 1_000_000], vec![10, 20])?;
            let build_buffers = build
                .columns()
                .iter()
                .map(|column| column.to_data().buffers()[0].as_ptr() as usize)
                .collect::<Vec<_>>();
            let probe = batch(vec![0, 1_000_000, 99], vec![10, 20, 10])?;
            let left = MemorySourceConfig::try_new_exec(
                &[vec![build]],
                Arc::clone(&schema),
                None,
            )?;
            let right = MemorySourceConfig::try_new_exec(
                &[vec![probe.clone()]],
                Arc::clone(&schema),
                None,
            )?;
            let keys = (0..if struct_in_list { 2 } else { 1 })
                .map(|index| {
                    Arc::new(PhysicalColumn::new(schema.field(index).name(), index))
                        as PhysicalExprRef
                })
                .collect::<Vec<_>>();
            let on = keys
                .iter()
                .map(|key| (Arc::clone(key), Arc::clone(key)))
                .collect();
            let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                keys.clone(),
                physical_lit(true),
            ));
            let consumer: PhysicalExprRef = dynamic.clone();
            let right = Arc::new(FilterExecBuilder::new(consumer, right).build()?);
            let join = HashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?
            .with_dynamic_filter_expr(Arc::clone(&dynamic))?;
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool))
                .build_arc()?;
            let mut config = SessionConfig::new();
            if !struct_in_list {
                config
                    .options_mut()
                    .optimizer
                    .hash_join_inlist_pushdown_max_size = 0;
            }
            let task = Arc::new(
                TaskContext::default()
                    .with_runtime(runtime)
                    .with_session_config(config),
            );
            let batches = collect(join.execute(0, task)?).await?;
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
            let current = dynamic.current()?;
            let mut membership = None;
            current.apply(|node| {
                if node.downcast_ref::<InListExpr>().is_some()
                    || node.downcast_ref::<HashTableLookupExpr>().is_some()
                {
                    membership = Some(Arc::clone(node));
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            let membership = membership.expect("join-created membership expression");
            assert!(pool.reserved() > 0);

            if struct_in_list {
                let list = membership
                    .downcast_ref::<InListExpr>()
                    .expect("join IN-list");
                let literal = list.list()[0].downcast_ref::<Literal>().unwrap();
                let ScalarValue::Struct(values) = literal.value() else {
                    panic!("join must publish Struct IN literals");
                };
                let buffers = values
                    .columns()
                    .iter()
                    .map(|column| column.to_data().buffers()[0].as_ptr() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(
                    buffers, build_buffers,
                    "literal buffers are producer buffers"
                );
                let negated: PhysicalExprRef =
                    Arc::new(NotExpr::new(Arc::clone(&membership)));
                let snapshot: PhysicalExprRef =
                    Arc::new(DynamicFilterPhysicalExpr::new(keys, negated));
                let pruning = PruningPredicateBuilder::new()
                    .with_file_schema(Arc::clone(&schema))
                    .try_build(Arc::clone(&snapshot))?;
                assert!(
                    pruning
                        .orig_expr()
                        .downcast_ref::<InListExpr>()
                        .unwrap()
                        .negated(),
                    "exercise actual snapshot NOT(IN) simplification"
                );
                let producer_charge = pool.reserved();
                assert!(producer_charge > 0);
                drop(snapshot);
                drop(membership);
                drop(current);
                drop(batches);
                drop(dynamic);
                drop(join);
                assert_eq!(
                    pool.reserved(),
                    producer_charge,
                    "pruner must retain the entire original producer charge"
                );
                let actual = pruning
                    .orig_expr()
                    .evaluate(&probe)?
                    .into_array(probe.num_rows())?;
                assert_eq!(
                    actual.as_ref(),
                    &BooleanArray::from(vec![false, false, true])
                );
                let unknown = PrunableStatistics::new(
                    vec![Arc::new(Statistics::new_unknown(&schema))],
                    Arc::clone(&schema),
                );
                assert_eq!(pruning.prune(&unknown)?, vec![true]);
                drop(actual);
                drop(pruning);
                assert_eq!(pool.reserved(), 0, "pruner releases producer charge");
            } else {
                let bitmap_bytes = membership
                    .downcast_ref::<HashTableLookupExpr>()
                    .unwrap()
                    .pruning_bitmap()
                    .expect("sparse bitmap")
                    .size();
                let dynamic_expr: PhysicalExprRef = dynamic.clone();
                let pruning = PruningPredicateBuilder::new()
                    .with_file_schema(Arc::clone(&schema))
                    .try_build(dynamic_expr)?;
                let mut bitmap_expr = None;
                pruning.predicate_expr().apply(|node| {
                    if node.to_string().starts_with("KEY_RANGE_BITMAP(") {
                        bitmap_expr = Some(Arc::clone(node));
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                let bitmap_expr = bitmap_expr.expect("actual bitmap pruning expression");
                let remapped = bitmap_expr
                    .with_new_children(vec![physical_lit(0_i64), physical_lit(0_i64)])?;
                drop(pruning);
                drop(membership);
                drop(current);
                drop(batches);
                drop(dynamic);
                drop(join);
                assert_eq!(
                    pool.reserved(),
                    bitmap_bytes,
                    "detached pruning expression retains only the bitmap charge"
                );
                let actual = remapped.evaluate(&probe)?.into_array(probe.num_rows())?;
                assert_eq!(actual.as_ref(), &BooleanArray::from(vec![true, true, true]));
                let absent = remapped.with_new_children(vec![
                    physical_lit(500_000_i64),
                    physical_lit(500_000_i64),
                ])?;
                let actual = absent.evaluate(&probe)?.into_array(probe.num_rows())?;
                assert_eq!(
                    actual.as_ref(),
                    &BooleanArray::from(vec![false, false, false])
                );
                drop(absent);
                assert_eq!(pool.reserved(), 0, "detached bitmap charge releases last");
            }
        }
        Ok(())
    }

    #[test]
    fn remaining_row_group_count_reports_non_skipped_groups() {
        let mut filter = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(4));
        assert_eq!(filter.remaining_row_group_count(), 4);

        filter.access_plan.skip(1);
        assert_eq!(filter.remaining_row_group_count(), 3);

        filter.access_plan.skip(3);
        assert_eq!(filter.remaining_row_group_count(), 2);
    }

    #[test]
    fn row_group_pruning_predicate_simple_expr() {
        use datafusion_expr::{col, lit};
        // int > 1 => c1_max > 1
        let schema =
            Arc::new(Schema::new(vec![Field::new("c1", DataType::Int32, false)]));
        let expr = col("c1").gt(lit(15));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));

        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32);
        let schema_descr = get_test_schema_descr(vec![field]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(1),
                Some(10),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(11),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );

        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1]))
    }

    #[test]
    fn row_group_fully_matched_requires_known_non_null_predicate_columns() {
        use datafusion_expr::{col, lit};

        let schema = Arc::new(Schema::new(vec![Field::new("c1", DataType::Int32, true)]));
        let expr = logical2physical(&col("c1").gt(lit(15)), &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));

        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32);
        let schema_descr = get_test_schema_descr(vec![field]);

        // All three row groups have non-null values in the predicate range,
        // so none are pruned. Only the second row group can be proven fully
        // matched because it is the only one with a known zero null count.
        let rg_with_null = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(16),
                Some(20),
                None,
                Some(1),
                false,
            )],
        );
        let rg_without_null = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(16),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );
        let rg_unknown_null_count = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(16),
                Some(20),
                None,
                None,
                false,
            )],
        );

        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(3));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rg_with_null, rg_without_null, rg_unknown_null_count],
            &pruning_predicate,
            &metrics,
        );

        assert_eq!(row_groups.access_plan.row_group_indexes(), vec![0, 1, 2]);
        assert_eq!(row_groups.is_fully_matched(), &vec![false, true, false]);
    }

    #[test]
    fn null_in_list_under_not_does_not_prove_full_match() {
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_physical_expr::expressions::{
            col as physical_col, in_list, lit as physical_lit,
        };
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;

        let schema =
            Arc::new(Schema::new(vec![Field::new("c1", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![100, 1000]))],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .set_bloom_filter_enabled(false)
            .build();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(properties))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let bytes = Bytes::from(bytes);
        let metadata = Arc::clone(
            ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
                .unwrap()
                .metadata(),
        );
        let column = physical_col("c1", &schema).unwrap();
        let mut list = (0..21)
            .map(|value| physical_lit(ScalarValue::Int32(Some(value))))
            .collect::<Vec<_>>();
        list.push(physical_lit(ScalarValue::Int32(None)));
        let in_list = in_list(Arc::clone(&column), list, &false, &schema).unwrap();
        let not_in: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(in_list));
        let at_least_100: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&column),
            Operator::GtEq,
            physical_lit(ScalarValue::Int32(Some(100))),
        ));
        let thousand: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            column,
            Operator::Eq,
            physical_lit(ScalarValue::Int32(Some(1000))),
        ));
        // At c1=100, the first arm is UNKNOWN and the second is false.
        // At c1=1000, the second arm is true, so LIMIT must not discard it.
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(not_in, Operator::And, at_least_100)),
            Operator::Or,
            thousand,
        ));
        let pruning_predicate = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(&schema))
            .with_max_in_list_size(32)
            .try_build(Arc::clone(&predicate))
            .unwrap();
        assert!(pruning_predicate.can_be_inverted_for_full_match());
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics_with_metadata(
            &schema,
            &metadata,
            &pruning_predicate,
            &metrics,
        );
        let full_match_flags = row_groups.is_fully_matched().clone();
        row_groups.prune_by_limit(1, metadata.row_groups(), &metrics);
        let selected = row_groups.row_group_indexes().collect::<Vec<_>>();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .with_row_groups(selected.clone())
            .build()
            .unwrap();
        let matching_values = reader
            .flat_map(|batch| {
                let batch = batch.unwrap();
                let filter = predicate
                    .evaluate(&batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                let filter = filter.as_any().downcast_ref::<BooleanArray>().unwrap();
                let filtered =
                    arrow::compute::filter_record_batch(&batch, filter).unwrap();
                filtered
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            (full_match_flags, selected, matching_values),
            (vec![false, false], vec![0, 1], vec![1000]),
            "UNKNOWN from NOT IN with a NULL member must not prove a full match"
        );
    }

    /// A CASE arm can manufacture NULL even when every input column is non-null.
    /// The first row is UNKNOWN and the second is TRUE; LIMIT must keep the latter.
    #[tokio::test]
    async fn computed_null_case_does_not_prove_full_match() -> Result<()> {
        use crate::row_filter::build_row_filter;
        use crate::source::ParquetSource;
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_datasource::PartitionedFile;
        use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
        use datafusion_datasource::source::DataSourceExec;
        use datafusion_execution::TaskContext;
        use datafusion_execution::object_store::ObjectStoreUrl;
        use datafusion_physical_expr::expressions::{
            CaseExpr, TryCastExpr, col as physical_col, lit as physical_lit,
        };
        use datafusion_physical_plan::filter::FilterExec;
        use datafusion_physical_plan::limit::GlobalLimitExec;
        use datafusion_physical_plan::{ExecutionPlan, collect, displayable};
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("x", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![-200, 0])),
                Arc::new(Int32Array::from(vec![1, 1])),
            ],
        )?;
        let column = |name| physical_col(name, &schema).unwrap();
        let lit_i32 = |v| physical_lit(ScalarValue::Int32(Some(v)));
        let when = Arc::new(BinaryExpr::new(column("a"), Operator::Eq, lit_i32(1)));
        let cast = Arc::new(TryCastExpr::new(column("x"), DataType::Int8));
        let then = Arc::new(BinaryExpr::new(
            cast,
            Operator::Gt,
            physical_lit(ScalarValue::Int8(Some(0))),
        ));
        let case = Arc::new(CaseExpr::try_new(
            None,
            vec![(when, then)],
            Some(physical_lit(ScalarValue::Boolean(Some(false)))),
        )?);
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(case)),
            Operator::And,
            Arc::new(BinaryExpr::new(column("b"), Operator::Gt, lit_i32(0))),
        ));
        let evaluated = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
        assert_eq!(
            evaluated.as_ref(),
            &BooleanArray::from(vec![None, Some(true)])
        );

        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .set_bloom_filter_enabled(false)
            .build();
        let mut bytes = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        let bytes = Bytes::from(bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
        let metadata = Arc::clone(builder.metadata());
        let pruning_predicate =
            build_test_pruning_predicate(Arc::clone(&predicate), Arc::clone(&schema));
        let metrics = parquet_file_metrics();
        let mut groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        groups.prune_by_statistics_with_metadata(
            &schema,
            &metadata,
            &pruning_predicate,
            &metrics,
        );
        let flags = groups.is_fully_matched().clone();
        groups.prune_by_limit(1, metadata.row_groups(), &metrics);
        let selected = groups.row_group_indexes().collect::<Vec<_>>();
        let row_filter =
            build_row_filter(&predicate, &schema, &metadata, false, &metrics)?
                .expect("real parquet reader filter");
        let reader = builder
            .with_row_groups(selected.clone())
            .with_row_filter(row_filter)
            .build()?;
        let read_x = reader
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();

        // The same two rows can share a row group; max(x)=0 then masks the
        // NULL produced by TRY_CAST(-200 AS Int8) in the inverse rewrite.
        let mut mixed_bytes = Vec::new();
        let mut mixed_writer =
            ArrowWriter::try_new(&mut mixed_bytes, Arc::clone(&schema), None)?;
        mixed_writer.write(&batch)?;
        mixed_writer.close()?;
        let mixed_bytes = Bytes::from(mixed_bytes);
        let mixed_builder =
            ParquetRecordBatchReaderBuilder::try_new(mixed_bytes.clone())?;
        let mixed_metadata = Arc::clone(mixed_builder.metadata());
        assert_eq!(mixed_metadata.num_row_groups(), 1);
        let mut mixed_groups =
            RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(1));
        mixed_groups.prune_by_statistics_with_metadata(
            &schema,
            &mixed_metadata,
            &pruning_predicate,
            &metrics,
        );
        let mixed_flags = mixed_groups.is_fully_matched().clone();
        mixed_groups.prune_by_limit(1, mixed_metadata.row_groups(), &metrics);
        let mixed_selected = mixed_groups.row_group_indexes().collect::<Vec<_>>();
        let mixed_filter =
            build_row_filter(&predicate, &schema, &mixed_metadata, false, &metrics)?
                .expect("mixed reader filter");
        let mixed_reader = mixed_builder
            .with_row_groups(mixed_selected.clone())
            .with_row_filter(mixed_filter)
            .build()?;
        let mixed_read_x = mixed_reader
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "mixed CASE flags={mixed_flags:?}, selected={mixed_selected:?}, reader={mixed_read_x:?}"
        );
        // Additionally place a later matching row in a second row group. A
        // LIMIT of two cannot be satisfied by the first row group alone.
        let batch3 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1])),
                Arc::new(Int32Array::from(vec![-200, 0, 0])),
                Arc::new(Int32Array::from(vec![1, 1, 1])),
            ],
        )?;
        let properties3 = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .set_bloom_filter_enabled(false)
            .build();
        let mut bytes3 = Vec::new();
        let mut writer3 =
            ArrowWriter::try_new(&mut bytes3, Arc::clone(&schema), Some(properties3))?;
        writer3.write(&batch3)?;
        writer3.close()?;
        let bytes3 = Bytes::from(bytes3);
        let builder3 = ParquetRecordBatchReaderBuilder::try_new(bytes3.clone())?;
        let metadata3 = Arc::clone(builder3.metadata());
        assert_eq!(metadata3.num_row_groups(), 2);
        let mut groups3 = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        groups3.prune_by_statistics_with_metadata(
            &schema,
            &metadata3,
            &pruning_predicate,
            &metrics,
        );
        let flags3 = groups3.is_fully_matched().clone();
        let before_limit3 = groups3.row_group_indexes().collect::<Vec<_>>();
        groups3.prune_by_limit(2, metadata3.row_groups(), &metrics);
        let selected3 = groups3.row_group_indexes().collect::<Vec<_>>();
        let filter3 = build_row_filter(&predicate, &schema, &metadata3, false, &metrics)?
            .expect("two-group reader filter");
        let reader3 = builder3
            .with_row_groups(selected3.clone())
            .with_row_filter(filter3)
            .build()?;
        let read3 = reader3
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "LIMIT2 before={before_limit3:?}, flags={flags3:?}, selected={selected3:?}, reader={read3:?}"
        );

        // Exercise the real Parquet DataSourceExec with row-filter pushdown both
        // enabled and disabled. The external filter gives identical SQL semantics.
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(file.path(), &bytes)?;
        let path = object_store::path::Path::from_filesystem_path(file.path()).unwrap();
        let mut exec_results = Vec::new();
        for pushdown in [false, true] {
            let source = Arc::new(
                ParquetSource::new(Arc::clone(&schema))
                    .with_predicate(Arc::clone(&predicate))
                    .with_pushdown_filters(pushdown)
                    .with_enable_page_index(false)
                    .with_bloom_filter_on_read(false),
            );
            let config =
                FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
                    .with_file(PartitionedFile::new(path.to_string(), bytes.len() as u64))
                    .build();
            let scan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
            let plan: Arc<dyn ExecutionPlan> =
                Arc::new(FilterExec::try_new(Arc::clone(&predicate), scan)?);
            let batches = collect(plan, Arc::new(TaskContext::default())).await?;
            let values = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>();
            exec_results.push(values);
        }
        let file3 = tempfile::NamedTempFile::new()?;
        std::fs::write(file3.path(), &bytes3)?;
        let path3 = object_store::path::Path::from_filesystem_path(file3.path()).unwrap();
        let mut exec3 = Vec::new();
        for pushdown in [false, true] {
            for limit in [None, Some(2)] {
                let source = Arc::new(
                    ParquetSource::new(Arc::clone(&schema))
                        .with_predicate(Arc::clone(&predicate))
                        .with_pushdown_filters(pushdown)
                        .with_enable_page_index(false)
                        .with_bloom_filter_on_read(false),
                );
                let config = FileScanConfigBuilder::new(
                    ObjectStoreUrl::local_filesystem(),
                    source,
                )
                .with_file(PartitionedFile::new(path3.to_string(), bytes3.len() as u64))
                // A scan fetch is valid only when its row filter enforces the predicate.
                .with_limit(if pushdown { limit } else { None })
                .build();
                let scan: Arc<dyn ExecutionPlan> =
                    DataSourceExec::from_data_source(config);
                let filtered: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(
                    Arc::clone(&predicate),
                    Arc::clone(&scan),
                )?);
                let plan: Arc<dyn ExecutionPlan> = if limit.is_some() {
                    Arc::new(GlobalLimitExec::new(filtered, 0, limit))
                } else {
                    filtered
                };
                eprintln!(
                    "pushdown={pushdown} limit={limit:?} plan:\n{}",
                    displayable(plan.as_ref()).indent(true)
                );
                let batches = collect(plan, Arc::new(TaskContext::default())).await?;
                if pushdown && limit.is_none() {
                    let metrics = scan.metrics().expect("scan metrics");
                    let pruned = metrics
                        .sum_by_name("pushdown_rows_pruned")
                        .expect("row filter metric")
                        .as_usize();
                    eprintln!("actual reader row-filter pruned={pruned}");
                    assert!(pruned > 0, "configured row filter must actually run");
                }
                let mut values = batches
                    .iter()
                    .flat_map(|batch| {
                        batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .values()
                            .to_vec()
                    })
                    .collect::<Vec<_>>();
                values.sort_unstable();
                exec3.push(values);
            }
        }
        eprintln!("LIMIT2 exec pushdown-off/on [no-limit, limit]: {exec3:?}");
        assert_eq!(
            exec3,
            vec![vec![0, 0]; 4],
            "LIMIT must agree with the unbounded results with and without pushdown"
        );
        assert_eq!(
            (mixed_flags, mixed_selected, mixed_read_x),
            (vec![false], vec![0], vec![0])
        );
        assert_eq!(
            (before_limit3, flags3, selected3, read3),
            (vec![0, 1], vec![false, false], vec![0, 1], vec![0, 0])
        );
        assert_eq!(
            (flags, selected, read_x, exec_results),
            (
                vec![false, false],
                vec![0, 1],
                vec![0],
                vec![vec![0], vec![0]]
            ),
            "computed NULL must not prove a full match or discard the later true row"
        );
        Ok(())
    }

    #[test]
    fn safe_cast_case_computed_null_does_not_prove_full_match() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::compute::CastOptions;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_physical_expr::expressions::{
            CaseExpr, CastExpr, col as physical_col, lit as physical_lit,
        };
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("x", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1])),
                Arc::new(Int32Array::from(vec![-200, 0])),
                Arc::new(Int32Array::from(vec![1, 1])),
            ],
        )?;
        let column = |name| physical_col(name, &schema).unwrap();
        let i32_lit = |value| physical_lit(ScalarValue::Int32(Some(value)));
        let cast = Arc::new(CastExpr::new(
            column("x"),
            DataType::Int8,
            Some(CastOptions {
                safe: true,
                ..Default::default()
            }),
        ));
        assert!(
            !cast.nullable(&schema)?,
            "safe CAST currently reports non-null for non-null input"
        );
        let then: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            cast,
            Operator::Gt,
            physical_lit(ScalarValue::Int8(Some(0))),
        ));
        assert!(!then.nullable(&schema)?);
        let case = Arc::new(CaseExpr::try_new(
            None,
            vec![(
                Arc::new(BinaryExpr::new(column("a"), Operator::Eq, i32_lit(1))),
                then,
            )],
            Some(physical_lit(ScalarValue::Boolean(Some(false)))),
        )?);
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(NotExpr::new(case)),
            Operator::And,
            Arc::new(BinaryExpr::new(column("b"), Operator::Gt, i32_lit(0))),
        ));
        let evaluated = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
        assert_eq!(
            evaluated.as_ref(),
            &BooleanArray::from(vec![None, Some(true)])
        );
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), None)?;
        writer.write(&batch)?;
        writer.close()?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))?;
        let metadata = Arc::clone(reader.metadata());
        let pruning = build_test_pruning_predicate(predicate, Arc::clone(&schema));
        let mut groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(1));
        groups.prune_by_statistics_with_metadata(
            &schema,
            &metadata,
            &pruning,
            &parquet_file_metrics(),
        );
        assert_eq!(
            groups.is_fully_matched(),
            &vec![false],
            "safe CAST may create NULL despite nullable=false"
        );
        Ok(())
    }

    #[test]
    fn optimized_computed_null_cast_does_not_prove_full_match() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_physical_expr::expressions::{
            TryCastExpr, col as physical_col, lit as physical_lit,
        };
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("x", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1])),
                Arc::new(Int32Array::from(vec![-200, 0, 0])),
                Arc::new(Int32Array::from(vec![1, 1, 1])),
            ],
        )?;
        let column = |name| physical_col(name, &schema).unwrap();
        let int32 = |v| physical_lit(ScalarValue::Int32(Some(v)));
        // The SQL optimizer eliminates CASE to this non-CASE expression.
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(column("a"), Operator::NotEq, int32(1))),
                Operator::Or,
                Arc::new(BinaryExpr::new(
                    Arc::new(TryCastExpr::new(column("x"), DataType::Int8)),
                    Operator::LtEq,
                    physical_lit(ScalarValue::Int8(Some(0))),
                )),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(column("b"), Operator::Gt, int32(0))),
        ));
        let values = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
        assert_eq!(
            values.as_ref(),
            &BooleanArray::from(vec![None, Some(true), Some(true)])
        );
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut bytes = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        let metadata = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))?
            .metadata()
            .clone();
        let pruning = build_test_pruning_predicate(predicate, Arc::clone(&schema));
        assert!(!pruning.can_be_inverted_for_full_match());
        let metrics = parquet_file_metrics();
        let mut groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        groups.prune_by_statistics_with_metadata(&schema, &metadata, &pruning, &metrics);
        assert_eq!(groups.is_fully_matched(), &vec![false, false]);
        groups.prune_by_limit(2, metadata.row_groups(), &metrics);
        assert_eq!(groups.row_group_indexes().collect::<Vec<_>>(), vec![0, 1]);
        Ok(())
    }

    #[test]
    fn total_casts_can_prove_full_match_and_satisfy_limit() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::compute::CastOptions;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_physical_expr::expressions::{
            CastExpr, TryCastExpr, col as physical_col, lit as physical_lit,
        };
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )?;
        let mut bytes = Vec::new();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .build();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        let metadata = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))?
            .metadata()
            .clone();
        for safe_cast in [false, true] {
            for target in [DataType::Int32, DataType::Int64] {
                let column = physical_col("x", &schema)?;
                let cast: Arc<dyn PhysicalExpr> = if safe_cast {
                    Arc::new(CastExpr::new(
                        column,
                        target.clone(),
                        Some(CastOptions {
                            safe: true,
                            ..Default::default()
                        }),
                    ))
                } else {
                    Arc::new(TryCastExpr::new(column, target.clone()))
                };
                let zero = match target {
                    DataType::Int32 => ScalarValue::Int32(Some(0)),
                    DataType::Int64 => ScalarValue::Int64(Some(0)),
                    _ => unreachable!(),
                };
                let predicate = build_test_pruning_predicate(
                    Arc::new(BinaryExpr::new(cast, Operator::Gt, physical_lit(zero))),
                    Arc::clone(&schema),
                );
                assert!(
                    predicate.can_be_inverted_for_full_match(),
                    "safe_cast={safe_cast} target={target}"
                );
                let metrics = parquet_file_metrics();
                let mut groups =
                    RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
                groups.prune_by_statistics_with_metadata(
                    &schema, &metadata, &predicate, &metrics,
                );
                assert_eq!(
                    groups.is_fully_matched(),
                    &vec![true, true],
                    "safe_cast={safe_cast} target={target}"
                );
                groups.prune_by_limit(1, metadata.row_groups(), &metrics);
                assert_eq!(groups.row_group_indexes().collect::<Vec<_>>(), vec![0]);
                assert_eq!(metrics.limit_pruned_row_groups.pruned(), 1);
            }
        }
        Ok(())
    }

    #[test]
    fn non_nullable_case_can_still_prove_full_match_and_satisfy_limit() -> Result<()> {
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;
        use bytes::Bytes;
        use datafusion_physical_expr::expressions::{
            CaseExpr, col as physical_col, lit as physical_lit,
        };
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::properties::WriterProperties;
        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )?;
        let column = physical_col("b", &schema)?;
        let zero = physical_lit(ScalarValue::Int32(Some(0)));
        let when = Arc::new(BinaryExpr::new(
            Arc::clone(&column),
            Operator::Gt,
            Arc::clone(&zero),
        ));
        let then = Arc::new(BinaryExpr::new(column, Operator::LtEq, zero));
        let case = Arc::new(CaseExpr::try_new(
            None,
            vec![(when, then)],
            Some(physical_lit(ScalarValue::Boolean(Some(false)))),
        )?);
        let predicate = build_test_pruning_predicate(
            Arc::new(NotExpr::new(case)),
            Arc::clone(&schema),
        );
        assert!(predicate.can_be_inverted_for_full_match());
        let mut bytes = Vec::new();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .build();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(&schema), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        let metadata = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))?
            .metadata()
            .clone();
        let metrics = parquet_file_metrics();
        let mut groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        groups
            .prune_by_statistics_with_metadata(&schema, &metadata, &predicate, &metrics);
        assert_eq!(groups.is_fully_matched(), &vec![true, true]);
        groups.prune_by_limit(1, metadata.row_groups(), &metrics);
        assert_eq!(groups.row_group_indexes().collect::<Vec<_>>(), vec![0]);
        Ok(())
    }

    #[test]
    fn prune_by_limit_preserves_row_selection() {
        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32);
        let schema_descr = get_test_schema_descr(vec![field]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(None, None, None, Some(0), false)],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(None, None, None, Some(0), false)],
        );
        let groups = &[rgm1, rgm2];

        let selection =
            RowSelection::from(vec![RowSelector::skip(900), RowSelector::select(100)]);
        let mut access_plan = ParquetAccessPlan::new_all(2);
        access_plan.scan_selection(0, selection.clone());
        access_plan.mark_fully_matched(0);
        access_plan.mark_fully_matched(1);

        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
        row_groups.prune_by_limit(50, groups, &metrics);

        assert_eq!(row_groups.access_plan.row_group_indexes(), vec![0]);
        assert_eq!(
            row_groups.access_plan.inner(),
            &[RowGroupAccess::Selection(selection), RowGroupAccess::Skip]
        );
        assert_eq!(row_groups.is_fully_matched(), &vec![true, false]);
    }

    #[test]
    fn prune_by_limit_counts_only_selected_rows() {
        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32);
        let schema_descr = get_test_schema_descr(vec![field]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(None, None, None, Some(0), false)],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(None, None, None, Some(0), false)],
        );
        let groups = &[rgm1, rgm2];

        let selection =
            RowSelection::from(vec![RowSelector::select(10), RowSelector::skip(990)]);
        let mut access_plan = ParquetAccessPlan::new_all(2);
        access_plan.scan_selection(0, selection.clone());
        access_plan.mark_fully_matched(0);

        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
        row_groups.prune_by_limit(50, groups, &metrics);

        assert_eq!(row_groups.access_plan.row_group_indexes(), vec![0, 1]);
        assert_eq!(
            row_groups.access_plan.inner(),
            &[RowGroupAccess::Selection(selection), RowGroupAccess::Scan]
        );
        assert_eq!(row_groups.is_fully_matched(), &vec![true, false]);
    }

    #[test]
    fn row_group_pruning_predicate_missing_stats() {
        use datafusion_expr::{col, lit};
        // int > 1 => c1_max > 1
        let schema =
            Arc::new(Schema::new(vec![Field::new("c1", DataType::Int32, false)]));
        let expr = col("c1").gt(lit(15));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));

        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32);
        let schema_descr = get_test_schema_descr(vec![field]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(None, None, None, Some(0), false)],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::int32(
                Some(11),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );
        let metrics = parquet_file_metrics();
        // missing statistics for first row group mean that the result from the predicate expression
        // is null / undefined so the first row group can't be filtered out
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::None);
    }

    #[test]
    fn row_group_pruning_predicate_partial_expr() {
        use datafusion_expr::{col, lit};
        // test row group predicate with partially supported expression
        // (int > 1) and ((int % 2) = 0) => c1_max > 1 and true
        let schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ]));
        let expr = col("c1").gt(lit(15)).and(col("c2").rem(lit(2)).eq(lit(0)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));

        let schema_descr = get_test_schema_descr(vec![
            PrimitiveTypeField::new("c1", PhysicalType::INT32),
            PrimitiveTypeField::new("c2", PhysicalType::INT32),
        ]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(1), Some(10), None, Some(0), false),
                ParquetStatistics::int32(Some(1), Some(10), None, Some(0), false),
            ],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(11), Some(20), None, Some(0), false),
                ParquetStatistics::int32(Some(11), Some(20), None, Some(0), false),
            ],
        );

        let metrics = parquet_file_metrics();
        let groups = &[rgm1, rgm2];
        // the first row group is still filtered out because the predicate expression can be partially evaluated
        // when conditions are joined using AND
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            groups,
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1]));

        // if conditions in predicate are joined with OR and an unsupported expression is used
        // this bypasses the entire predicate expression and no row groups are filtered out
        let expr = col("c1").gt(lit(15)).or(col("c2").rem(lit(2)).eq(lit(0)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));

        // if conditions in predicate are joined with OR and an unsupported expression is used
        // this bypasses the entire predicate expression and no row groups are filtered out
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            groups,
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::None);
    }

    #[test]
    fn row_group_pruning_predicate_file_schema() {
        use datafusion_expr::{col, lit};
        // test row group predicate when file schema is different than table schema
        // c1 > 0
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Int32, false),
        ]));
        let expr = col("c1").gt(lit(0));
        let expr = logical2physical(&expr, &table_schema);
        let pruning_predicate =
            build_test_pruning_predicate(expr, Arc::clone(&table_schema));

        // Model a file schema's column order c2 then c1, which is the opposite
        // of the table schema
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("c2", DataType::Int32, false),
            Field::new("c1", DataType::Int32, false),
        ]));
        let schema_descr = get_test_schema_descr(vec![
            PrimitiveTypeField::new("c2", PhysicalType::INT32),
            PrimitiveTypeField::new("c1", PhysicalType::INT32),
        ]);
        // rg1 has c2 less than zero, c1 greater than zero
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(-10), Some(-1), None, Some(0), false), // c2
                ParquetStatistics::int32(Some(1), Some(10), None, Some(0), false),
            ],
        );
        // rg1 has c2 greater than zero, c1 less than zero
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(1), Some(10), None, Some(0), false),
                ParquetStatistics::int32(Some(-10), Some(-1), None, Some(0), false),
            ],
        );

        let metrics = parquet_file_metrics();
        let groups = &[rgm1, rgm2];
        // the first row group should be left because c1 is greater than zero
        // the second should be filtered out because c1 is less than zero
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &file_schema,
            &schema_descr,
            groups,
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![0]));
    }

    fn gen_row_group_meta_data_for_pruning_predicate() -> Vec<RowGroupMetaData> {
        let schema_descr = get_test_schema_descr(vec![
            PrimitiveTypeField::new("c1", PhysicalType::INT32),
            PrimitiveTypeField::new("c2", PhysicalType::BOOLEAN),
        ]);
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(1), Some(10), None, Some(0), false),
                ParquetStatistics::boolean(Some(false), Some(true), None, Some(0), false),
            ],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![
                ParquetStatistics::int32(Some(11), Some(20), None, Some(0), false),
                ParquetStatistics::boolean(Some(false), Some(true), None, Some(1), false),
            ],
        );
        vec![rgm1, rgm2]
    }

    #[test]
    fn row_group_pruning_predicate_null_expr() {
        use datafusion_expr::{col, lit};
        // int > 1 and IsNull(bool) => c1_max > 1 and bool_null_count > 0
        let schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Boolean, false),
        ]));
        let schema_descr = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let expr = col("c1").gt(lit(15)).and(col("c2").is_null());
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        let groups = gen_row_group_meta_data_for_pruning_predicate();

        let metrics = parquet_file_metrics();
        // First row group was filtered out because it contains no null value on "c2".
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(2));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &groups,
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1]));
    }

    #[test]
    fn row_group_pruning_predicate_eq_null_expr() {
        use datafusion_expr::{col, lit};
        // test row group predicate with an unknown (Null) expr
        //
        // int > 1 and bool = NULL => c1_max > 1 and null
        let schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Int32, false),
            Field::new("c2", DataType::Boolean, false),
        ]));
        let schema_descr = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let expr = col("c1")
            .gt(lit(15))
            .and(col("c2").eq(lit(ScalarValue::Boolean(None))));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        let groups = gen_row_group_meta_data_for_pruning_predicate();

        let metrics = parquet_file_metrics();
        // bool = NULL always evaluates to NULL (and thus will not
        // pass predicates. Ideally these should both be false
        let mut row_groups =
            RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(groups.len()));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &groups,
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1]));
    }

    #[test]
    fn row_group_pruning_predicate_decimal_type() {
        // For the decimal data type, parquet can use `INT32`, `INT64`, `BYTE_ARRAY`, `FIXED_LENGTH_BYTE_ARRAY` to
        // store the data.
        // In this case, construct four types of statistics to filtered with the decimal predication.

        // INT32: c1 > 5, the c1 is decimal(9,2)
        // The type of scalar value if decimal(9,2), don't need to do cast
        let schema =
            Arc::new(Schema::new(vec![Field::new("c1", Decimal128(9, 2), false)]));
        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32)
            .with_logical_type(LogicalType::decimal(2, 9))
            .with_scale(2)
            .with_precision(9);
        let schema_descr = get_test_schema_descr(vec![field]);
        let expr = col("c1").gt(lit(ScalarValue::Decimal128(Some(500), 9, 2)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            // [1.00, 6.00]
            // c1 > 5, this row group will be included in the results.
            vec![ParquetStatistics::int32(
                Some(100),
                Some(600),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            // [0.1, 0.2]
            // c1 > 5, this row group will not be included in the results.
            vec![ParquetStatistics::int32(
                Some(10),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );
        let rgm3 = get_row_group_meta_data(
            &schema_descr,
            // [1, None]
            // c1 > 5, this row group can not be filtered out, so will be included in the results.
            vec![ParquetStatistics::int32(
                Some(100),
                None,
                None,
                Some(0),
                false,
            )],
        );
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(3));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2, rgm3],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![0, 2]));
    }

    #[test]
    fn row_group_pruning_predicate_decimal_type2() {
        // INT32: c1 > 5, but parquet decimal type has different precision or scale to arrow decimal
        // The c1 type is decimal(9,0) in the parquet file, and the type of scalar is decimal(5,2).
        // We should convert all type to the coercion type, which is decimal(11,2)
        // The decimal of arrow is decimal(5,2), the decimal of parquet is decimal(9,0)
        let schema =
            Arc::new(Schema::new(vec![Field::new("c1", Decimal128(9, 0), false)]));

        let field = PrimitiveTypeField::new("c1", PhysicalType::INT32)
            .with_logical_type(LogicalType::decimal(0, 9))
            .with_scale(0)
            .with_precision(9);
        let schema_descr = get_test_schema_descr(vec![field]);
        let expr = cast(col("c1"), Decimal128(11, 2)).gt(cast(
            lit(ScalarValue::Decimal128(Some(500), 5, 2)),
            Decimal128(11, 2),
        ));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            // [100, 600]
            // c1 > 5, this row group will be included in the results.
            vec![ParquetStatistics::int32(
                Some(100),
                Some(600),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            // [10, 20]
            // c1 > 5, this row group will be included in the results.
            vec![ParquetStatistics::int32(
                Some(10),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );
        let rgm3 = get_row_group_meta_data(
            &schema_descr,
            // [0, 2]
            // c1 > 5, this row group will not be included in the results.
            vec![ParquetStatistics::int32(
                Some(0),
                Some(2),
                None,
                Some(0),
                false,
            )],
        );
        let rgm4 = get_row_group_meta_data(
            &schema_descr,
            // [None, 2]
            // c1 > 5, this row group will also not be included in the results
            // (the min value is unknown, but the max value is 2, so no values can be greater than 5)
            vec![ParquetStatistics::int32(
                None,
                Some(2),
                None,
                Some(0),
                false,
            )],
        );
        let rgm5 = get_row_group_meta_data(
            &schema_descr,
            // [2, None]
            // c1 > 5, this row group must be included
            // (the min value is 2, but the max value is unknown, so it may have values greater than 5)
            vec![ParquetStatistics::int32(
                Some(2),
                None,
                None,
                Some(0),
                false,
            )],
        );
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(5));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2, rgm3, rgm4, rgm5],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![0, 1, 4]));
    }
    #[test]
    fn row_group_pruning_predicate_decimal_type3() {
        // INT64: c1 < 5, the c1 is decimal(18,2)
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c1",
            Decimal128(18, 2),
            false,
        )]));
        let field = PrimitiveTypeField::new("c1", PhysicalType::INT64)
            .with_logical_type(LogicalType::decimal(2, 18))
            .with_scale(2)
            .with_precision(18);
        let schema_descr = get_test_schema_descr(vec![field]);
        let expr = col("c1").lt(lit(ScalarValue::Decimal128(Some(500), 18, 2)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            // [6.00, 8.00]
            vec![ParquetStatistics::int32(
                Some(600),
                Some(800),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            // [0.1, 0.2]
            vec![ParquetStatistics::int64(
                Some(10),
                Some(20),
                None,
                Some(0),
                false,
            )],
        );
        let rgm3 = get_row_group_meta_data(
            &schema_descr,
            // [0.1, 0.2]
            vec![ParquetStatistics::int64(None, None, None, Some(0), false)],
        );
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(3));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2, rgm3],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1, 2]));
    }
    #[test]
    fn row_group_pruning_predicate_decimal_type4() {
        // FIXED_LENGTH_BYTE_ARRAY: c1 = decimal128(100000, 28, 3), the c1 is decimal(18,2)
        // the type of parquet is decimal(18,2)
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c1",
            Decimal128(18, 2),
            false,
        )]));
        let field = PrimitiveTypeField::new("c1", PhysicalType::FIXED_LEN_BYTE_ARRAY)
            .with_logical_type(LogicalType::decimal(2, 18))
            .with_scale(2)
            .with_precision(18)
            .with_byte_len(16);
        let schema_descr = get_test_schema_descr(vec![field]);
        // cast the type of c1 to decimal(28,3)
        let left = cast(col("c1"), Decimal128(28, 3));
        let expr = left.eq(lit(ScalarValue::Decimal128(Some(100000), 28, 3)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        // we must use the big-endian when encode the i128 to bytes or vec[u8].
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::fixed_len_byte_array(
                // 5.00
                Some(FixedLenByteArray::from(ByteArray::from(
                    500i128.to_be_bytes().to_vec(),
                ))),
                // 80.00
                Some(FixedLenByteArray::from(ByteArray::from(
                    8000i128.to_be_bytes().to_vec(),
                ))),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::fixed_len_byte_array(
                // 5.00
                Some(FixedLenByteArray::from(ByteArray::from(
                    500i128.to_be_bytes().to_vec(),
                ))),
                // 200.00
                Some(FixedLenByteArray::from(ByteArray::from(
                    20000i128.to_be_bytes().to_vec(),
                ))),
                None,
                Some(0),
                false,
            )],
        );

        let rgm3 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::fixed_len_byte_array(
                None,
                None,
                None,
                Some(0),
                false,
            )],
        );
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(3));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2, rgm3],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1, 2]));
    }
    #[test]
    fn row_group_pruning_predicate_decimal_type5() {
        // BYTE_ARRAY: c1 = decimal128(100000, 28, 3), the c1 is decimal(18,2)
        // the type of parquet is decimal(18,2)
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c1",
            Decimal128(18, 2),
            false,
        )]));
        let field = PrimitiveTypeField::new("c1", PhysicalType::BYTE_ARRAY)
            .with_logical_type(LogicalType::decimal(2, 18))
            .with_scale(2)
            .with_precision(18)
            .with_byte_len(16);
        let schema_descr = get_test_schema_descr(vec![field]);
        // cast the type of c1 to decimal(28,3)
        let left = cast(col("c1"), Decimal128(28, 3));
        let expr = left.eq(lit(ScalarValue::Decimal128(Some(100000), 28, 3)));
        let expr = logical2physical(&expr, &schema);
        let pruning_predicate = build_test_pruning_predicate(expr, Arc::clone(&schema));
        // we must use the big-endian when encode the i128 to bytes or vec[u8].
        let rgm1 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::byte_array(
                // 5.00
                Some(ByteArray::from(500i128.to_be_bytes().to_vec())),
                // 80.00
                Some(ByteArray::from(8000i128.to_be_bytes().to_vec())),
                None,
                Some(0),
                false,
            )],
        );
        let rgm2 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::byte_array(
                // 5.00
                Some(ByteArray::from(500i128.to_be_bytes().to_vec())),
                // 200.00
                Some(ByteArray::from(20000i128.to_be_bytes().to_vec())),
                None,
                Some(0),
                false,
            )],
        );
        let rgm3 = get_row_group_meta_data(
            &schema_descr,
            vec![ParquetStatistics::byte_array(
                None,
                None,
                None,
                Some(0),
                false,
            )],
        );
        let metrics = parquet_file_metrics();
        let mut row_groups = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(3));
        row_groups.prune_by_statistics(
            &schema,
            &schema_descr,
            &[rgm1, rgm2, rgm3],
            &pruning_predicate,
            &metrics,
        );
        assert_pruned(row_groups, ExpectedPruning::Some(vec![1, 2]));
    }

    fn get_row_group_meta_data(
        schema_descr: &SchemaDescPtr,
        column_statistics: Vec<ParquetStatistics>,
    ) -> RowGroupMetaData {
        let mut columns = vec![];
        let number_row = 1000;
        for (i, s) in column_statistics.iter().enumerate() {
            let column = ColumnChunkMetaData::builder(schema_descr.column(i))
                .set_statistics(s.clone())
                .set_num_values(number_row)
                .build()
                .unwrap();
            columns.push(column);
        }
        RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(number_row)
            .set_total_byte_size(2000)
            .set_column_metadata(columns)
            .build()
            .unwrap()
    }

    fn get_test_schema_descr(fields: Vec<PrimitiveTypeField>) -> SchemaDescPtr {
        use parquet::schema::types::Type as SchemaType;
        let schema_fields = fields
            .iter()
            .map(|field| {
                let mut builder =
                    SchemaType::primitive_type_builder(field.name, field.physical_ty);
                // add logical type for the parquet field
                if let Some(logical_type) = &field.logical_ty {
                    builder = builder.with_logical_type(Some(logical_type.clone()));
                }
                if let Some(precision) = field.precision {
                    builder = builder.with_precision(precision);
                }
                if let Some(scale) = field.scale {
                    builder = builder.with_scale(scale);
                }
                if let Some(byte_len) = field.byte_len {
                    builder = builder.with_length(byte_len);
                }
                Arc::new(builder.build().unwrap())
            })
            .collect::<Vec<_>>();
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(schema_fields)
            .build()
            .unwrap();

        Arc::new(SchemaDescriptor::new(Arc::new(schema)))
    }

    fn parquet_file_metrics() -> ParquetFileMetrics {
        let metrics = Arc::new(ExecutionPlanMetricsSet::new());
        ParquetFileMetrics::new(0, "file.parquet", &metrics)
    }

    fn assert_pruned(row_groups: RowGroupAccessPlanFilter, expected: ExpectedPruning) {
        expected.assert(&row_groups);
    }
}
