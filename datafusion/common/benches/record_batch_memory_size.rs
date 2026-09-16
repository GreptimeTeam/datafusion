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

//! Compares [`get_record_batch_memory_size`] against the `ArrayData`-based walk
//! it replaced, on batch shapes that appear in observability workloads.

use std::hint::black_box;
use std::num::NonZero;
use std::sync::Arc;

use arrow::array::{
    ArrayData, ArrayRef, DictionaryArray, Float64Array, RecordBatch, StringArray,
    TimestampMillisecondArray, UInt32Array,
};
use arrow::datatypes::UInt32Type;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::HashSet;
use datafusion_common::utils::memory::{
    RecordBatchMemoryCounter, get_record_batch_memory_size,
};

/// The accounting walk as it was before the direct-buffer rewrite, so both
/// variants are measured in the same binary against the same fixtures.
fn baseline_count_array_data(
    array_data: &ArrayData,
    counted_buffers: &mut HashSet<NonZero<usize>>,
    total_size: &mut usize,
) {
    for buffer in array_data.buffers() {
        if counted_buffers.insert(buffer.data_ptr().addr()) {
            *total_size += buffer.capacity();
        }
    }

    if let Some(null_buffer) = array_data.nulls()
        && counted_buffers.insert(null_buffer.inner().inner().data_ptr().addr())
    {
        *total_size += null_buffer.inner().inner().capacity();
    }

    for child in array_data.child_data() {
        baseline_count_array_data(child, counted_buffers, total_size);
    }
}

fn baseline_count_batch(
    batch: &RecordBatch,
    counted_buffers: &mut HashSet<NonZero<usize>>,
) -> usize {
    let mut total_size = 0;
    for array in batch.columns() {
        baseline_count_array_data(&array.to_data(), counted_buffers, &mut total_size);
    }
    total_size
}

fn baseline_get_record_batch_memory_size(batch: &RecordBatch) -> usize {
    baseline_count_batch(batch, &mut HashSet::default())
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    RecordBatch::try_from_iter(columns).unwrap()
}

/// Timestamp + value + low-cardinality dictionary tags: the shape a PromQL scan
/// pushes through every operator.
fn promql_batch(rows: usize, tags: usize) -> RecordBatch {
    let tag_values: ArrayRef = Arc::new(StringArray::from(
        (0..64).map(|v| format!("value-{v:04}")).collect::<Vec<_>>(),
    ));

    let mut columns: Vec<(String, ArrayRef)> = vec![
        (
            "ts".to_string(),
            Arc::new(TimestampMillisecondArray::from(
                (0..rows).map(|r| r as i64 * 1000).collect::<Vec<_>>(),
            )),
        ),
        (
            "value".to_string(),
            Arc::new(Float64Array::from(
                (0..rows).map(|r| r as f64).collect::<Vec<_>>(),
            )),
        ),
    ];

    for tag in 0..tags {
        let keys = UInt32Array::from(
            (0..rows)
                .map(|r| ((r + tag) % 64) as u32)
                .collect::<Vec<_>>(),
        );
        columns.push((
            format!("tag_{tag}"),
            Arc::new(
                DictionaryArray::<UInt32Type>::try_new(keys, Arc::clone(&tag_values))
                    .unwrap(),
            ),
        ));
    }

    batch(
        columns
            .iter()
            .map(|(n, a)| (n.as_str(), Arc::clone(a)))
            .collect(),
    )
}

fn wide_primitive_batch(rows: usize, columns: usize) -> RecordBatch {
    let named: Vec<(String, ArrayRef)> = (0..columns)
        .map(|c| {
            (
                format!("f{c}"),
                Arc::new(Float64Array::from(
                    (0..rows).map(|r| (r + c) as f64).collect::<Vec<_>>(),
                )) as ArrayRef,
            )
        })
        .collect();
    batch(
        named
            .iter()
            .map(|(n, a)| (n.as_str(), Arc::clone(a)))
            .collect(),
    )
}

fn bench_single_batch(c: &mut Criterion) {
    let cases = vec![
        ("promql_8192x4tags", promql_batch(8192, 4)),
        ("promql_64x4tags", promql_batch(64, 4)),
        ("promql_8192x8tags", promql_batch(8192, 8)),
        ("wide_primitive_8192x20", wide_primitive_batch(8192, 20)),
        ("narrow_primitive_8192x2", wide_primitive_batch(8192, 2)),
    ];

    let mut group = c.benchmark_group("record_batch_memory_size");
    for (name, batch) in &cases {
        assert_eq!(
            get_record_batch_memory_size(batch),
            baseline_get_record_batch_memory_size(batch),
            "{name} disagrees with the baseline walk"
        );
        group.bench_function(BenchmarkId::new("baseline", name), |b| {
            b.iter(|| black_box(baseline_get_record_batch_memory_size(black_box(batch))))
        });
        group.bench_function(BenchmarkId::new("direct", name), |b| {
            b.iter(|| black_box(get_record_batch_memory_size(black_box(batch))))
        });
    }
    group.finish();
}

/// Many small zero-copy slices of one batch, as produced by operators that emit
/// a large batch in pieces.
fn bench_shared_slices(c: &mut Criterion) {
    let backing = promql_batch(8192, 4);
    let slices: Vec<RecordBatch> = (0..512).map(|i| backing.slice(i * 16, 16)).collect();

    let mut group = c.benchmark_group("record_batch_memory_size/shared_slices");
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let total: usize = black_box(&slices)
                .iter()
                .map(baseline_get_record_batch_memory_size)
                .sum();
            black_box(total)
        })
    });
    group.bench_function("direct", |b| {
        b.iter(|| {
            let total: usize = black_box(&slices)
                .iter()
                .map(get_record_batch_memory_size)
                .sum();
            black_box(total)
        })
    });
    group.finish();
}

/// One counter accumulating many batches, as the hash join build side does.
/// Distinct buffers push it past the inline capacity on the first batches.
fn bench_accumulating_counter(c: &mut Criterion) {
    let batches: Vec<RecordBatch> = (0..64).map(|_| promql_batch(1024, 4)).collect();

    let mut group = c.benchmark_group("record_batch_memory_size/accumulating_counter");
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let mut counted = HashSet::default();
            let total: usize = black_box(&batches)
                .iter()
                .map(|batch| baseline_count_batch(batch, &mut counted))
                .sum();
            black_box(total)
        })
    });
    group.bench_function("direct", |b| {
        b.iter(|| {
            let mut counter = RecordBatchMemoryCounter::new();
            for batch in black_box(&batches) {
                counter.count_batch(batch);
            }
            black_box(counter.memory_usage())
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_single_batch,
    bench_shared_slices,
    bench_accumulating_counter
);
criterion_main!(benches);
