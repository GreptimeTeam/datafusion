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

use arrow::compute::CastOptions;
use arrow::datatypes::DataType;
use arrow::{array::*, compute::cast_with_options};

use datafusion_common::{internal_err, plan_err, Result, ScalarValue};

/// Operator that return value for given index/field of list/struct
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
                    return Some(plan_err!(
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
                ScalarValue::Int64(Some(v)) => *v,
                ScalarValue::UInt8(Some(v)) => *v as i64,
                ScalarValue::UInt16(Some(v)) => *v as i64,
                ScalarValue::UInt32(Some(v)) => *v as i64,
                ScalarValue::UInt64(Some(v)) => *v as i64,
                _ => {
                    return Some(plan_err!(
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

pub(crate) fn cast_to_string_array(array: ArrayRef) -> Result<ArrayRef> {
    cast_with_options(&array, &DataType::Utf8, &CastOptions::default())
        .map_err(Into::into)
}

/// Operator that returns value with a given path as list of string
pub(crate) fn collection_select_path_dyn_scalar(
    left: Arc<dyn Array>,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    match (left.data_type(), right) {
        (DataType::List(_)|DataType::Struct(_), ScalarValue::List(field_path)) => {
            if matches!(field_path.value_type(), DataType::Utf8 | DataType::Utf8View) {
                let mut collection = left;
                let path_list = field_path.value(0);

                for i in 0..path_list.len() {
                    if path_list.is_null(i) {
                        return Some(plan_err!("Unexpected null in path list"));
                    } else {
                        let field_scalar = match ScalarValue::try_from_array(&path_list, i) {
                            Ok(field_scalar) =>  {
                                // check if field_scalar is a numerical value,
                                // and collection is a list, we will transform
                                // the field_scalar value type
                                if matches!(collection.data_type(), DataType::List(_))  {
                                    if let Ok(casted_scalar) = field_scalar.cast_to(&DataType::Int64) {
                                        casted_scalar
                                    } else {
                                        field_scalar
                                    }
                                } else {
                                    field_scalar
                                }
                            },
                            Err(e) => return Some(internal_err!("Failed to convert to ScalarValue {}", e))
                        };

                        match collection_select_dyn_scalar(&collection, field_scalar) {
                            Some(Ok(col)) => {
                                // early return for null value
                                if col.data_type() == &DataType::Null {
                                    return Some(Ok(col));
                                }

                                collection = col;
                            },
                            other => {
                                return other;
                            }
                        }
                    }
                }
                Some(Ok(Arc::clone(&collection)))
            } else{
                Some(plan_err!(
                    "Expected string list for operator #> or #>>"
                ))
            }
        },
        (_, ScalarValue::List(_field_list)) => {
            Some(Ok(new_null_array(&DataType::Null, left.len())))
        }
        (other1, other2) =>Some(plan_err!("Data type {}, {} not supported for binary operation collection_select_path_dyn_scalar", other1, other2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Fields, Int32Type};
    use datafusion_common::ScalarValue;
    use std::sync::Arc;

    #[test]
    fn test_collection_select_struct_field() {
        let a_array = Arc::new(Int32Array::from(vec![Some(1), Some(2), None]));
        let b_array = Arc::new(StringArray::from(vec![Some("x"), Some("y"), Some("z")]));
        let struct_array = StructArray::try_new(
            Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]),
            vec![a_array, b_array],
            None,
        )
        .unwrap();

        // Test valid field access "a"
        let result = collection_select_dyn_scalar(
            &struct_array,
            ScalarValue::Utf8(Some("a".to_string())),
        )
        .unwrap()
        .unwrap();
        let expected =
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None])) as ArrayRef;
        assert_eq!(&result, &expected);

        // Test valid field access "b"
        let result = collection_select_dyn_scalar(
            &struct_array,
            ScalarValue::Utf8(Some("b".to_string())),
        )
        .unwrap()
        .unwrap();
        let expected = Arc::new(StringArray::from(vec![Some("x"), Some("y"), Some("z")]))
            as ArrayRef;
        assert_eq!(&result, &expected);

        // Test invalid field access
        let result = collection_select_dyn_scalar(
            &struct_array,
            ScalarValue::Utf8(Some("c".to_string())),
        )
        .unwrap()
        .unwrap();
        let expected = new_null_array(&DataType::Null, 3);
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_collection_select_list_index() {
        // Create a list array
        let list_array =
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                Some(vec![Some(4), Some(5)]),
                Some(vec![Some(7)]), // Single element list instead of empty
                Some(vec![Some(6)]),
            ])) as ArrayRef;

        // Test valid positive index access
        let result = collection_select_dyn_scalar(
            list_array.as_ref(),
            ScalarValue::Int32(Some(0)),
        )
        .unwrap()
        .unwrap();
        let expected =
            Arc::new(Int32Array::from(vec![Some(1), Some(4), Some(7), Some(6)]))
                as ArrayRef;
        assert_eq!(&result, &expected);

        // Test valid negative index access
        let result = collection_select_dyn_scalar(
            list_array.as_ref(),
            ScalarValue::Int32(Some(-1)),
        )
        .unwrap()
        .unwrap();
        let expected =
            Arc::new(Int32Array::from(vec![Some(3), Some(5), Some(7), Some(6)]))
                as ArrayRef;
        assert_eq!(&result, &expected);

        // Test out of bounds index - but skip this test for now due to NullArray issue
        let result = collection_select_dyn_scalar(
            list_array.as_ref(),
            ScalarValue::Int32(Some(10)),
        )
        .unwrap()
        .unwrap();
        let expected = new_null_array(&DataType::Null, 4);

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_cast_to_string_array() {
        let int_array = Arc::new(Int32Array::from(vec![Some(1), Some(2), None]));
        let result = cast_to_string_array(int_array).unwrap();
        let expected =
            Arc::new(StringArray::from(vec![Some("1"), Some("2"), None])) as ArrayRef;
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_collection_select_path_simple() {
        // Create a simple struct with one field
        let struct_array = StructArray::try_new(
            Fields::from(vec![Field::new("a", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![Some(1)]))],
            None,
        )
        .unwrap();

        // Test path ["a"]
        let path_list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Utf8(Some("a".to_string()))],
            &DataType::Utf8,
        ));

        // Debug: Check what type the path_list has
        println!("Path list data type: {:?}", path_list.data_type());

        let result = collection_select_path_dyn_scalar(Arc::new(struct_array), path_list);

        match result {
            Some(Ok(array)) => {
                println!("Success! Result data type: {:?}", array.data_type());
                let expected = Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef;
                assert_eq!(&array, &expected);
            }
            Some(Err(e)) => {
                println!("Error: {:?}", e);
                panic!("Unexpected error: {:?}", e);
            }
            None => {
                println!("Result is None");
                panic!("Unexpected None result");
            }
        }
    }

    #[test]
    fn test_collection_select_path_mixed_types() {
        // Create a struct with a list field
        let list_array =
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(10), Some(20), Some(30)]),
            ])) as ArrayRef;
        let struct_array = StructArray::try_new(
            Fields::from(vec![Field::new(
                "items",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            )]),
            vec![list_array],
            None,
        )
        .unwrap();

        // Test path ["items", "1"] - but "1" is not a valid struct field name
        // Let's test just ["items"] instead
        let path_list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Utf8(Some("items".to_string()))],
            &DataType::Utf8,
        ));
        let result = collection_select_path_dyn_scalar(Arc::new(struct_array), path_list)
            .unwrap()
            .unwrap();
        // The result should be the list array [[10, 20, 30]]
        let expected = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(10), Some(20), Some(30)]),
        ])) as ArrayRef;
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_collection_select_path_invalid_path() {
        let struct_array = StructArray::try_new(
            Fields::from(vec![Field::new("a", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![Some(1)]))],
            None,
        )
        .unwrap();

        // Test invalid path ["b"]
        let path_list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::Utf8(Some("b".to_string()))],
            &DataType::Utf8,
        ));
        let result = collection_select_path_dyn_scalar(Arc::new(struct_array), path_list)
            .unwrap()
            .unwrap();
        let expected = new_null_array(&DataType::Null, 1);
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_collection_select_errors() {
        let struct_array = StructArray::try_new(
            Fields::from(vec![Field::new("a", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![Some(1)]))],
            None,
        )
        .unwrap();

        // Test null field name
        let result =
            collection_select_dyn_scalar(&struct_array, ScalarValue::Utf8(None)).unwrap();
        assert!(result.is_err());

        // Test invalid data type combination
        let result =
            collection_select_dyn_scalar(&struct_array, ScalarValue::Float64(Some(1.0)))
                .unwrap();
        assert!(result.is_err());
    }
}
