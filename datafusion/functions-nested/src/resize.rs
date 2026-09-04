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

//! [`ScalarUDFImpl`] definitions for array_resize function.

use crate::utils::make_scalar_function;
use arrow::array::{
    Array, ArrayRef, Capacities, GenericListArray, Int64Array, MutableArrayData,
    NullBufferBuilder, OffsetSizeTrait, new_null_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::DataType;
use arrow::datatypes::Field;
use arrow::datatypes::{
    DataType::{LargeList, List},
    FieldRef,
};
use datafusion_common::cast::{as_int64_array, as_large_list_array, as_list_array};
use datafusion_common::utils::ListCoercion;
use datafusion_common::{Result, ScalarValue, exec_err, internal_datafusion_err};
use datafusion_expr::{
    ArrayFunctionArgument, ArrayFunctionSignature, ColumnarValue, Documentation,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion_macros::user_doc;
use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

make_udf_expr_and_func!(
    ArrayResize,
    array_resize,
    array size value,
    "returns an array with the specified size filled with the given value.",
    array_resize_udf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Resizes the list to contain size elements. Initializes new elements with value or empty if value is not set.",
    syntax_example = "array_resize(array, size, value)",
    sql_example = r#"```sql
> select array_resize([1, 2, 3], 5, 0);
+-------------------------------------+
| array_resize(List([1,2,3],5,0))     |
+-------------------------------------+
| [1, 2, 3, 0, 0]                     |
+-------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "size", description = "New size of given array."),
    argument(
        name = "value",
        description = "Defines new elements' value or empty if value is not set."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayResize {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayResize {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayResize {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::ArraySignature(ArrayFunctionSignature::Array {
                        arguments: vec![
                            ArrayFunctionArgument::Array,
                            ArrayFunctionArgument::Index,
                        ],
                        array_coercion: Some(ListCoercion::FixedSizedListToList),
                    }),
                    TypeSignature::ArraySignature(ArrayFunctionSignature::Array {
                        arguments: vec![
                            ArrayFunctionArgument::Array,
                            ArrayFunctionArgument::Index,
                            ArrayFunctionArgument::Element,
                        ],
                        array_coercion: Some(ListCoercion::FixedSizedListToList),
                    }),
                ],
                Volatility::Immutable,
            ),
            aliases: vec!["list_resize".to_string()],
        }
    }
}

impl ScalarUDFImpl for ArrayResize {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_resize"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match &arg_types[0] {
            List(field) => Ok(List(Arc::clone(field))),
            LargeList(field) => Ok(LargeList(Arc::clone(field))),
            DataType::Null => {
                Ok(List(Arc::new(Field::new_list_field(DataType::Int64, true))))
            }
            _ => exec_err!(
                "Not reachable, data_type should be List, LargeList or FixedSizeList"
            ),
        }
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_resize_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn array_resize_inner(arg: &[ArrayRef]) -> Result<ArrayRef> {
    if arg.len() < 2 || arg.len() > 3 {
        return exec_err!("array_resize needs two or three arguments");
    }

    let array = &arg[0];

    // Checks if entire array is null
    if array.logical_null_count() == array.len() {
        let return_type = match array.data_type() {
            List(field) => List(Arc::clone(field)),
            LargeList(field) => LargeList(Arc::clone(field)),
            _ => {
                return exec_err!(
                    "array_resize does not support type '{:?}'.",
                    array.data_type()
                );
            }
        };
        return Ok(new_null_array(&return_type, array.len()));
    }

    let new_len = as_int64_array(&arg[1])?;
    let new_element = if arg.len() == 3 {
        Some(Arc::clone(&arg[2]))
    } else {
        None
    };

    match &arg[0].data_type() {
        List(field) => {
            let array = as_list_array(&arg[0])?;
            general_list_resize::<i32>(array, new_len, field, new_element)
        }
        LargeList(field) => {
            let array = as_large_list_array(&arg[0])?;
            general_list_resize::<i64>(array, new_len, field, new_element)
        }
        array_type => exec_err!("array_resize does not support type '{array_type}'."),
    }
}

/// array_resize keep the original array and append the default element to the end
fn general_list_resize<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    count_array: &Int64Array,
    field: &FieldRef,
    default_element: Option<ArrayRef>,
) -> Result<ArrayRef> {
    let data_type = array.value_type();
    let ResizePreflight {
        counts,
        output_values_len,
    } = preflight_resize(array, count_array, &data_type)?;

    let values = array.values();
    let original_data = values.to_data();

    // create default element array
    let default_element = if let Some(default_element) = default_element {
        default_element
    } else {
        let null_scalar = ScalarValue::try_from(&data_type)?;
        null_scalar.to_array_of_size(original_data.len())?
    };
    let default_value_data = default_element.to_data();

    // create a mutable array to store the original data
    let capacity = Capacities::Array(output_values_len);
    let zero = O::from_usize(0).ok_or_else(|| {
        internal_datafusion_err!("array_resize: failed to convert zero offset")
    })?;
    let mut offsets = vec![zero];
    let mut mutable = MutableArrayData::with_capacities(
        vec![&original_data, &default_value_data],
        false,
        capacity,
    );

    let mut null_builder = NullBufferBuilder::new(array.len());

    for ((row_index, offset_window), count) in
        array.offsets().windows(2).enumerate().zip(counts)
    {
        let Some(count) = count else {
            null_builder.append_null();
            offsets.push(offsets[row_index]);
            continue;
        };
        null_builder.append_non_null();

        let start = offset_window[0];
        let end = offset_window[1];
        let start_usize = start.to_usize().ok_or_else(|| {
            internal_datafusion_err!(
                "array_resize: failed to convert source list offset to usize"
            )
        })?;
        let end_usize = end.to_usize().ok_or_else(|| {
            internal_datafusion_err!(
                "array_resize: failed to convert source list offset to usize"
            )
        })?;
        let current_len = (end - start).to_usize().ok_or_else(|| {
            internal_datafusion_err!(
                "array_resize: failed to convert source list length to usize"
            )
        })?;
        if count > current_len {
            let extra_count = count - current_len;
            mutable.extend(0, start_usize, end_usize);
            // append default element
            for _ in 0..extra_count {
                mutable.extend(1, row_index, row_index + 1);
            }
        } else {
            let count = O::from_usize(count).ok_or_else(|| {
                internal_datafusion_err!("array_resize: failed to convert size to offset")
            })?;
            let end = start + count;
            let end_usize = end.to_usize().ok_or_else(|| {
                internal_datafusion_err!(
                    "array_resize: failed to convert source list offset to usize"
                )
            })?;
            mutable.extend(0, start_usize, end_usize);
        };
        offsets.push(O::from_usize(mutable.len()).ok_or_else(|| {
            internal_datafusion_err!("array_resize: failed to convert size to offset")
        })?);
    }

    let data = mutable.freeze();

    Ok(Arc::new(GenericListArray::<O>::try_new(
        Arc::clone(field),
        OffsetBuffer::<O>::new(offsets.into()),
        arrow::array::make_array(data),
        null_builder.finish(),
    )?))
}

#[derive(Debug)]
struct ResizePreflight {
    counts: Vec<Option<usize>>,
    output_values_len: usize,
}

/// Validates target lengths before materializing default values or allocating output data.
fn preflight_resize<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    count_array: &Int64Array,
    value_type: &DataType,
) -> Result<ResizePreflight> {
    let mut counts = Vec::with_capacity(array.len());
    let mut output_values_len = 0;
    let max_values = max_resize_values(value_type);

    for row_index in 0..array.len() {
        if array.is_null(row_index) {
            counts.push(None);
            continue;
        }

        // Deliberately read null counts as the existing build loop does. For this
        // branch, their value buffer produces an empty, non-null list.
        let count = usize::try_from(count_array.value(row_index)).map_err(|_| {
            internal_datafusion_err!("array_resize: failed to convert size to usize")
        })?;
        output_values_len = checked_resize_total(output_values_len, count)?;

        if output_values_len > max_values || O::from_usize(output_values_len).is_none() {
            return exec_err!(
                "array_resize: resulting array of {output_values_len} elements exceeds the maximum array size"
            );
        }
        counts.push(Some(count));
    }

    Ok(ResizePreflight {
        counts,
        output_values_len,
    })
}

fn checked_resize_total(total: usize, count: usize) -> Result<usize> {
    total.checked_add(count).ok_or_else(|| {
        internal_datafusion_err!(
            "array_resize: resulting array element count overflow exceeds the maximum array size"
        )
    })
}

/// The most bytes an Arrow `MutableBuffer` can request after reserving its
/// maximum 64-byte allocation alignment padding.
const MAX_BUFFER_BYTES: usize = (isize::MAX as usize) - 63;

/// The fallback number of bytes assumed for each variable-width value.
///
/// It bounds eager preallocation, but cannot bound arbitrary payload bytes
/// copied while extending a variable-width array.
const VARIABLE_WIDTH_BYTES_PER_VALUE: usize = size_of::<u128>();

fn max_buffer_values(element_width: usize) -> usize {
    MAX_BUFFER_BYTES / element_width.max(1)
}

fn max_bitmap_values() -> usize {
    // `bit_util::ceil` uses `usize::div_ceil`, so all usize values are safe
    // inputs and require at most `usize::MAX / 8 + 1` bytes.
    usize::MAX
}

fn max_offset_values(offset_width: usize) -> usize {
    // List and binary buffers allocate one leading offset in addition to each value.
    max_buffer_values(offset_width).saturating_sub(1)
}

fn max_variable_width_values(offset_width: usize) -> usize {
    max_offset_values(offset_width).min(max_buffer_values(VARIABLE_WIDTH_BYTES_PER_VALUE))
}

fn max_run_end_values(run_end_type: &DataType) -> usize {
    match run_end_type {
        DataType::Int16 => i16::MAX as usize,
        DataType::Int32 => i32::MAX as usize,
        DataType::Int64 => i64::MAX as usize,
        _ => 0,
    }
}

/// Safe logical element-count limit whose Arrow `Capacities::Array` eager
/// preallocation cannot exceed `isize::MAX` after 64-byte rounding.
///
/// Variable-size leaves use a conservative 16-byte-per-value heuristic only
/// where payload bytes cannot be derived from the type. These limits do not
/// bound arbitrary copied payload bytes or the variable cardinality of nested
/// children while `extend` copies input data.
fn max_resize_values(value_type: &DataType) -> usize {
    match value_type {
        DataType::FixedSizeBinary(size) => {
            usize::try_from(*size).map(max_buffer_values).unwrap_or(0)
        }
        DataType::FixedSizeList(field, size) => usize::try_from(*size)
            .ok()
            .map(|size| {
                max_bitmap_values().min(match size {
                    0 => max_resize_values(field.data_type()),
                    size => max_resize_values(field.data_type()) / size,
                })
            })
            .unwrap_or(0),
        List(field) | DataType::Map(field, _) => {
            max_offset_values(size_of::<i32>()).min(max_resize_values(field.data_type()))
        }
        LargeList(field) => {
            max_offset_values(size_of::<i64>()).min(max_resize_values(field.data_type()))
        }
        DataType::ListView(field) => {
            max_buffer_values(size_of::<i32>()).min(max_resize_values(field.data_type()))
        }
        DataType::LargeListView(field) => {
            max_buffer_values(size_of::<i64>()).min(max_resize_values(field.data_type()))
        }
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| max_resize_values(field.data_type()))
            .min()
            .unwrap_or_else(max_bitmap_values)
            .min(max_bitmap_values()),
        DataType::Union(fields, mode) => {
            let direct_limit = match mode {
                arrow::datatypes::UnionMode::Sparse => max_buffer_values(size_of::<i8>()),
                arrow::datatypes::UnionMode::Dense => max_buffer_values(size_of::<i8>())
                    .min(max_buffer_values(size_of::<i32>()))
                    // Dense union offsets are written as the previous child length.
                    .min(i32::MAX as usize + 1),
            };
            fields
                .iter()
                .map(|(_, field)| max_resize_values(field.data_type()))
                .fold(direct_limit, usize::min)
        }
        DataType::RunEndEncoded(run_ends, values) => {
            max_run_end_values(run_ends.data_type())
                .min(max_resize_values(run_ends.data_type()))
                .min(max_resize_values(values.data_type()))
        }
        DataType::Dictionary(key_type, _) => key_type
            .primitive_width()
            .map(max_buffer_values)
            .unwrap_or(0),
        DataType::Utf8 | DataType::Binary => max_variable_width_values(size_of::<i32>()),
        DataType::LargeUtf8 | DataType::LargeBinary => {
            max_variable_width_values(size_of::<i64>())
        }
        DataType::BinaryView | DataType::Utf8View => max_buffer_values(size_of::<u128>()),
        DataType::Boolean => max_bitmap_values(),
        // Arrow does not allocate a data or validity buffer for Null arrays.
        DataType::Null => usize::MAX,
        _ => value_type
            .primitive_width()
            .map(max_buffer_values)
            .unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        array_resize_inner, checked_resize_total, max_bitmap_values, max_buffer_values,
        max_offset_values, max_resize_values, preflight_resize,
    };
    use arrow::array::{
        ArrayRef, AsArray, FixedSizeBinaryArray, Int64Array, LargeListArray, ListArray,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Int64Type};
    use std::mem::size_of;
    use std::sync::Arc;

    fn assert_limit_error(err: datafusion_common::DataFusionError, total: usize) {
        let err = err.to_string();
        assert!(err.contains(&total.to_string()), "unexpected error: {err}");
        assert!(
            err.contains("exceeds the maximum array size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn preflight_rejects_huge_list_size() {
        let array =
            ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1)])]);
        let counts = Int64Array::from(vec![i64::MAX]);

        assert_limit_error(
            preflight_resize::<i32>(&array, &counts, &DataType::Int64).unwrap_err(),
            i64::MAX as usize,
        );
    }

    #[test]
    fn preflight_rejects_huge_large_list_size() {
        let array =
            LargeListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![
                Some(1),
            ])]);
        let counts = Int64Array::from(vec![i64::MAX]);

        assert_limit_error(
            preflight_resize::<i64>(&array, &counts, &DataType::Int64).unwrap_err(),
            i64::MAX as usize,
        );
    }

    #[test]
    fn preflight_rejects_i32_offset_overflow() {
        let array =
            ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1)])]);
        let counts = Int64Array::from(vec![3_000_000_000]);

        assert_limit_error(
            preflight_resize::<i32>(&array, &counts, &DataType::Int64).unwrap_err(),
            3_000_000_000,
        );
    }

    #[test]
    fn preflight_uses_fixed_size_binary_width() {
        let values =
            FixedSizeBinaryArray::try_from_iter(vec![vec![0_u8; 32]].into_iter())
                .unwrap();
        let field = Arc::new(Field::new_list_field(DataType::FixedSizeBinary(32), true));
        let array = LargeListArray::new(
            field,
            OffsetBuffer::<i64>::new(vec![0, 1].into()),
            Arc::new(values),
            None,
        );
        let total = max_resize_values(&DataType::FixedSizeBinary(32)) + 1;
        let counts = Int64Array::from(vec![total as i64]);

        assert_limit_error(
            preflight_resize::<i64>(&array, &counts, &DataType::FixedSizeBinary(32))
                .unwrap_err(),
            total,
        );
    }

    #[test]
    fn preflight_rejects_old_exact_width_boundary() {
        let array =
            LargeListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![
                Some(1),
            ])]);
        let total = (isize::MAX as usize) / size_of::<i64>();
        let counts = Int64Array::from(vec![total as i64]);

        assert_limit_error(
            preflight_resize::<i64>(&array, &counts, &DataType::Int64).unwrap_err(),
            total,
        );
    }

    #[test]
    fn max_resize_values_uses_conservative_utf8_heuristic() {
        assert_eq!(
            max_resize_values(&DataType::Utf8),
            max_buffer_values(size_of::<u128>())
        );
    }

    #[test]
    fn max_resize_values_matches_arrow_eager_capacity_behavior() {
        assert_eq!(max_resize_values(&DataType::Boolean), max_bitmap_values());
        assert_eq!(max_resize_values(&DataType::Null), usize::MAX);
        assert_eq!(
            max_resize_values(&DataType::Dictionary(
                Box::new(DataType::Int16),
                Box::new(DataType::Utf8),
            )),
            max_buffer_values(size_of::<i16>())
        );
        assert_eq!(
            max_resize_values(&DataType::BinaryView),
            max_buffer_values(size_of::<u128>())
        );

        let dense_union = DataType::Union(
            [(0, Arc::new(Field::new("item", DataType::Int8, true)))]
                .into_iter()
                .collect(),
            arrow::datatypes::UnionMode::Dense,
        );
        assert_eq!(max_resize_values(&dense_union), i32::MAX as usize + 1,);

        let run_end_encoded = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::Int8, true)),
        );
        assert_eq!(max_resize_values(&run_end_encoded), i16::MAX as usize,);
    }

    #[test]
    fn max_resize_values_limits_fixed_size_list_child_capacity() {
        let value_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int8, true)),
            64,
        );
        let old_overflow_boundary = usize::MAX / 64 + 1;

        assert!(
            max_resize_values(&value_type) < old_overflow_boundary,
            "fixed-size-list limit must prevent Arrow's child capacity multiplication"
        );
    }

    #[test]
    fn max_resize_values_limits_nested_large_list_offsets() {
        let value_type =
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int8, true)));

        assert_eq!(
            max_resize_values(&value_type),
            max_offset_values(size_of::<i64>())
        );
    }

    #[test]
    fn preflight_accumulates_across_rows() {
        let array = LargeListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2)]),
        ]);
        let count = max_resize_values(&DataType::Int64) / 2 + 1;
        let total = count * 2;
        let counts = Int64Array::from(vec![count as i64, count as i64]);

        assert_limit_error(
            preflight_resize::<i64>(&array, &counts, &DataType::Int64).unwrap_err(),
            total,
        );
    }

    #[test]
    fn preflight_reports_cumulative_checked_add_overflow() {
        let err = checked_resize_total(usize::MAX, 1).unwrap_err().to_string();
        assert!(err.contains("array_resize"), "unexpected error: {err}");
        assert!(
            err.contains("exceeds the maximum array size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn preflight_skips_only_null_input_arrays() {
        let array = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            None,
            Some(vec![Some(1)]),
        ]);
        let counts = Int64Array::from(vec![Some(i64::MAX), None]);

        let preflight =
            preflight_resize::<i32>(&array, &counts, &DataType::Int64).unwrap();
        assert_eq!(preflight.counts, vec![None, Some(0)]);
        assert_eq!(preflight.output_values_len, 0);
    }

    #[test]
    fn preflight_rejects_negative_size_without_panicking() {
        let array =
            ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1)])]);
        let counts = Int64Array::from(vec![-1]);

        let err = preflight_resize::<i32>(&array, &counts, &DataType::Int64)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("array_resize: failed to convert size to usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn array_resize_resizes_list_with_row_specific_fills() {
        let array: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                Some(vec![Some(4), Some(5)]),
            ]));
        let counts: ArrayRef = Arc::new(Int64Array::from(vec![5, 1]));
        let fills: ArrayRef = Arc::new(Int64Array::from(vec![10, 20]));

        let result = array_resize_inner(&[array, counts, fills]).unwrap();
        let expected = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3), Some(10), Some(10)]),
            Some(vec![Some(4)]),
        ]);
        assert_eq!(result.as_list::<i32>(), &expected);
    }

    #[test]
    fn array_resize_resizes_large_list_with_row_specific_fills() {
        let array: ArrayRef =
            Arc::new(LargeListArray::from_iter_primitive::<Int64Type, _, _>(
                vec![
                    Some(vec![Some(1), Some(2), Some(3)]),
                    Some(vec![Some(4), Some(5)]),
                ],
            ));
        let counts: ArrayRef = Arc::new(Int64Array::from(vec![5, 1]));
        let fills: ArrayRef = Arc::new(Int64Array::from(vec![10, 20]));

        let result = array_resize_inner(&[array, counts, fills]).unwrap();
        let expected = LargeListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3), Some(10), Some(10)]),
            Some(vec![Some(4)]),
        ]);
        assert_eq!(result.as_list::<i64>(), &expected);
    }

    #[test]
    #[ignore = "the huge resize witness must run process-isolated"]
    fn array_resize_huge_size_returns_controlled_error() {
        let control_array: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1)]),
            ]));
        let control_count: ArrayRef = Arc::new(Int64Array::from(vec![3]));
        let control_fill: ArrayRef = Arc::new(Int64Array::from(vec![0]));
        let control =
            array_resize_inner(&[control_array, control_count, control_fill]).unwrap();
        let expected =
            ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![
                Some(1),
                Some(0),
                Some(0),
            ])]);
        assert_eq!(control.as_list::<i32>(), &expected);

        let array: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1)]),
            ]));
        let count: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX]));
        let fill: ArrayRef = Arc::new(Int64Array::from(vec![0]));
        let err = array_resize_inner(&[array, count, fill]).unwrap_err();
        assert_limit_error(err, i64::MAX as usize);
    }
}
