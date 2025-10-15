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

use arrow::array::*;
use arrow::datatypes::DataType;

use datafusion_common::{internal_err, plan_err, Result, ScalarValue};

pub(crate) fn collection_select_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    match (left.data_type(), right.data_type()) {
        (DataType::Struct(struct_type), DataType::Utf8 | DataType::Utf8View) => {
            // Extract field name from scalar
            let field_name = match &right {
                ScalarValue::Utf8(Some(s)) | ScalarValue::Utf8View(Some(s)) => s.as_str(),
                _ => {
                    return Some(internal_err!(
                        "Expected non-null string for struct field access"
                    ));
                }
            };
            let struct_array = match left.as_any().downcast_ref::<StructArray>() {
                Some(struct_array) => struct_array,
                None => return Some(internal_err!("Failed to downcast to StructArray"))
            };

            // Find the field index by name
            let field_idx = struct_type
                .iter()
                .position(|f| f.name() == field_name);

            match field_idx {
                Some(idx) => {
                    Some(Ok(Arc::clone(struct_array.column(idx))))
                }
                None => {
                    // Create a null array with the same length as the struct array
                    Some(Ok(new_null_array(&DataType::Null, struct_array.len())))
                }
            }
        },
        (_other, DataType::Utf8 | DataType::Utf8View) => {
            Some(Ok(new_null_array(&DataType::Null, left.len())))
        }
        (
            DataType::List(_list_type),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64,
        ) => {
            let index = match &right {
                ScalarValue::Int8(Some(v)) => *v as i64,
                ScalarValue::Int16(Some(v)) => *v as i64,
                ScalarValue::Int32(Some(v)) => *v as i64,
                ScalarValue::Int64(Some(v)) => *v as i64,
                ScalarValue::UInt8(Some(v)) => *v as i64,
                ScalarValue::UInt16(Some(v)) => *v as i64,
                ScalarValue::UInt32(Some(v)) => *v as i64,
                ScalarValue::UInt64(Some(v)) => *v as i64,
                _ => {
                    return Some(internal_err!(
                        "Expected non-null integer for list index access"
                    ));
                }
            };

            let list_array = match left.as_any().downcast_ref::<ListArray>() {
                Some(list_array) => list_array,
                None => return Some(internal_err!("Failed to downcast to ListArray"))
            };

            // Collect the values to build the result array
            let mut scalars = Vec::with_capacity(list_array.len());

            for i in 0..list_array.len() {
                if list_array.is_null(i) {
                    scalars.push(ScalarValue::Null);
                    continue;
                }

                let list = list_array.value(i);
                let list_len = list.len();

                // Handle negative indexing (Python-style) for signed integers
                let actual_index = if let ScalarValue::Int8(_) | ScalarValue::Int16(_)
                    | ScalarValue::Int32(_) | ScalarValue::Int64(_) = &right {
                    if index < 0 {
                        let signed_index = index;
                        if (-signed_index) as usize > list_len {
                            list_len // Out of bounds, will be caught below
                        } else {
                            (list_len as i64 + signed_index) as usize
                        }
                    } else {
                        index as usize
                    }
                } else {
                    index as usize
                };

                if actual_index < list_len && !list.is_null(actual_index) {
                    match ScalarValue::try_from_array(&list, actual_index) {
                        Ok(item) => scalars.push(item),
                        Err(e) => return Some(Err(e)),
                    }

                } else {
                    scalars.push(ScalarValue::Null);
                }
            }
            Some(ScalarValue::iter_to_array(scalars))
        },
        (
        _other,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64,
        ) => {
            Some(Ok(new_null_array(&DataType::Null, left.len())))
        }
        (other1, other2) => Some(plan_err!("Data type {}, {} not supported for binary operation collection_select_dyn_scalar", other1, other2))
    }
}
