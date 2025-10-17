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
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use datafusion_common::{Result, ScalarValue};

/// Test if the fist value contains the second, array version
///
/// Rule of containment:
/// https://www.postgresql.org/docs/18/datatype-json.html#JSON-CONTAINMENT
pub(crate) fn collection_contains_dyn(
    left: Arc<dyn Array>,
    right: Arc<dyn Array>,
) -> Result<ArrayRef> {
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(format!(
            "Arrays must have the same length: {} != {}",
            left.len(),
            right.len()
        ))
        .into());
    }

    let nulls = NullBuffer::union(left.nulls(), right.nulls());
    let mut results = BooleanBufferBuilder::new(left.len());

    for i in 0..left.len() {
        if nulls.as_ref().map_or(false, |n| n.is_null(i)) {
            results.append(false);
            continue;
        }

        let left_value = ScalarValue::try_from_array(&left, i)?;
        let right_value = ScalarValue::try_from_array(&right, i)?;

        let contains = jsonb_contains_scalar(&left_value, &right_value)?;
        results.append(contains);
    }

    let data = unsafe {
        ArrayDataBuilder::new(DataType::Boolean)
            .len(left.len())
            .buffers(vec![results.into()])
            .nulls(nulls)
            .build_unchecked()
    };
    Ok(Arc::new(BooleanArray::from(data)))
}

/// Test if left JSONB scalar value contains right JSONB scalar value following PostgreSQL rules
fn jsonb_contains_scalar(left: &ScalarValue, right: &ScalarValue) -> Result<bool> {
    match (left, right) {
        // Scalar values contain only identical values
        (ScalarValue::Utf8(Some(l)), ScalarValue::Utf8(Some(r))) => Ok(l == r),
        (ScalarValue::Utf8View(Some(l)), ScalarValue::Utf8View(Some(r))) => Ok(l == r),
        (ScalarValue::Int8(Some(l)), ScalarValue::Int8(Some(r))) => Ok(l == r),
        (ScalarValue::Int16(Some(l)), ScalarValue::Int16(Some(r))) => Ok(l == r),
        (ScalarValue::Int32(Some(l)), ScalarValue::Int32(Some(r))) => Ok(l == r),
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => Ok(l == r),
        (ScalarValue::UInt8(Some(l)), ScalarValue::UInt8(Some(r))) => Ok(l == r),
        (ScalarValue::UInt16(Some(l)), ScalarValue::UInt16(Some(r))) => Ok(l == r),
        (ScalarValue::UInt32(Some(l)), ScalarValue::UInt32(Some(r))) => Ok(l == r),
        (ScalarValue::UInt64(Some(l)), ScalarValue::UInt64(Some(r))) => Ok(l == r),
        (ScalarValue::Float32(Some(l)), ScalarValue::Float32(Some(r))) => Ok(l == r),
        (ScalarValue::Float64(Some(l)), ScalarValue::Float64(Some(r))) => Ok(l == r),
        (ScalarValue::Boolean(Some(l)), ScalarValue::Boolean(Some(r))) => Ok(l == r),

        // Arrays: order and duplicates don't matter
        (ScalarValue::List(l_array), ScalarValue::List(r_array)) => {
            let left_values = extract_scalar_values_from_list(l_array)?;
            let right_values = extract_scalar_values_from_list(r_array)?;

            // Special case: array can contain primitive value
            if right_values.len() == 1 && !is_nested_scalar(&right_values[0]) {
                return array_contains_primitive_scalar(&left_values, &right_values[0]);
            }

            // For arrays, check if all elements in right array are contained in left array
            array_contains_array_scalar(&left_values, &right_values)
        }

        // Mixed types: array can contain primitive
        (ScalarValue::List(l_array), right) if !is_nested_scalar(right) => {
            let left_values = extract_scalar_values_from_list(l_array)?;
            array_contains_primitive_scalar(&left_values, right)
        }

        // Structs: right must be subset of left
        (ScalarValue::Struct(l_struct), ScalarValue::Struct(r_struct)) => {
            struct_contains_struct_scalar(l_struct, r_struct)
        }

        _ => Ok(false),
    }
}

/// Extract scalar values from a list array
fn extract_scalar_values_from_list(
    list_array: &Arc<ListArray>,
) -> Result<Vec<ScalarValue>> {
    let mut values = Vec::new();
    for i in 0..list_array.value(0).len() {
        values.push(ScalarValue::try_from_array(&list_array.value(0), i)?);
    }
    Ok(values)
}

/// Check if array contains a primitive value
fn array_contains_primitive_scalar(
    array: &[ScalarValue],
    primitive: &ScalarValue,
) -> Result<bool> {
    for element in array {
        if jsonb_contains_scalar(element, primitive)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if left array contains all elements of right array
fn array_contains_array_scalar(
    left: &[ScalarValue],
    right: &[ScalarValue],
) -> Result<bool> {
    for right_element in right {
        let mut found = false;

        for left_element in left {
            if jsonb_contains_scalar(left_element, right_element)? {
                found = true;
                break;
            }
        }

        if !found {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check if left struct contains right struct (right is subset of left)
fn struct_contains_struct_scalar(
    left: &Arc<StructArray>,
    right: &Arc<StructArray>,
) -> Result<bool> {
    // Get field names from right struct data type
    let DataType::Struct(right_fields) = right.data_type() else {
        return Ok(false);
    };

    // For each field in right struct, check if it exists and matches in left struct
    for right_field in right_fields {
        let field_name = right_field.name();

        // Find matching field in left struct
        if let Some(left_field) = left.column_by_name(field_name) {
            let right_field = right.column_by_name(field_name).unwrap();
            let left_value = ScalarValue::try_from_array(left_field.as_ref(), 0)?;
            let right_value = ScalarValue::try_from_array(right_field.as_ref(), 0)?;

            if !jsonb_contains_scalar(&left_value, &right_value)? {
                return Ok(false);
            }
        } else {
            // Field exists in right but not in left
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check if scalar value is nested (list, struct)
fn is_nested_scalar(scalar: &ScalarValue) -> bool {
    matches!(scalar, ScalarValue::List(_) | ScalarValue::Struct(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, ListArray, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Int32Type};
    use std::sync::Arc;

    #[test]
    fn test_scalar_contains() -> Result<()> {
        // Test scalar values contain only identical values
        let left = Arc::new(StringArray::from(vec!["hello", "world"]));
        let right = Arc::new(StringArray::from(vec!["hello", "foo"]));
        let result = collection_contains_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_array_contains_primitive() -> Result<()> {
        // Test array contains primitive value
        let left = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4), Some(5)]),
        ]));
        let right = Arc::new(Int32Array::from(vec![2, 6]));
        let result = collection_contains_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_array_contains_array() -> Result<()> {
        // Test array contains array (order and duplicates don't matter)
        let left = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4), Some(5)]),
        ]));
        let right = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(2), Some(1)]), // order doesn't matter
            Some(vec![Some(6)]),          // not contained
        ]));
        let result = collection_contains_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_struct_contains_subset() -> Result<()> {
        // Test {"foo":{"bar":"baz"}} contains {"foo":{}}
        // This should return true because right struct is a subset of left struct
        
        // Create left struct: {"foo":{"bar":"baz"}}
        let inner_left_fields = Fields::from(vec![Field::new("bar", DataType::Utf8, true)]);
        let inner_left_array = Arc::new(StringArray::from(vec!["baz"])) as ArrayRef;
        let inner_left_struct = StructArray::new(
            inner_left_fields.clone(),
            vec![inner_left_array],
            None,
        );
        
        let outer_left_fields = Fields::from(vec![Field::new("foo", DataType::Struct(inner_left_fields), true)]);
        let left = Arc::new(StructArray::new(
            outer_left_fields,
            vec![Arc::new(inner_left_struct) as ArrayRef],
            None,
        ));

        // Create right struct: {"foo":{}}
        let inner_right_fields = Fields::empty(); // empty struct
        let inner_right_struct = StructArray::try_new_with_length(
            inner_right_fields.clone(),
            vec![],
            None,
            1,
        )?;
        
        let outer_right_fields = Fields::from(vec![Field::new("foo", DataType::Struct(inner_right_fields), true)]);
        let right = Arc::new(StructArray::new(
            outer_right_fields,
            vec![Arc::new(inner_right_struct) as ArrayRef],
            None,
        ));

        let result = collection_contains_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }
}
