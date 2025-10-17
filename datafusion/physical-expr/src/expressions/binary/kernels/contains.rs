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
        if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
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

/// Test if collection (list or struct) contains string key, PostgreSQL `?` operator
///
/// Rule for `?` operator:
/// https://www.postgresql.org/docs/18/functions-json.html#FUNCTIONS-JSONB-OP-TABLE
/// - For arrays: tests if the string exists as an array element
/// - For objects: tests if the string exists as a top-level key
pub(crate) fn collection_contains_string_dyn(
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
        if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
            results.append(false);
            continue;
        }

        let left_value = ScalarValue::try_from_array(&left, i)?;
        let right_value = ScalarValue::try_from_array(&right, i)?;

        let contains = collection_contains_string_scalar(&left_value, &right_value)?;
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

/// Test if collection scalar contains string scalar, PostgreSQL `?` operator
fn collection_contains_string_scalar(
    left: &ScalarValue,
    right: &ScalarValue,
) -> Result<bool> {
    match (left, right) {
        // Struct: test if string exists as top-level key
        (ScalarValue::Struct(struct_array), ScalarValue::Utf8(Some(field_name))) => {
            // Check if field exists in struct
            let exists = struct_array.column_by_name(field_name).is_some();
            Ok(exists)
        }

        // List: test if string exists as array element
        (ScalarValue::List(list_array), ScalarValue::Utf8(Some(search_string))) => {
            let list_values = extract_scalar_values_from_list(list_array)?;

            // Search through list elements for the string
            for element in &list_values {
                if let ScalarValue::Utf8(Some(s)) = element {
                    if s == search_string {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }

        _ => Ok(false),
    }
}

/// Test if collection contains ANY of the strings in the array, PostgreSQL `?|` operator
///
/// Rule for `?|` operator:
/// https://www.postgresql.org/docs/18/functions-json.html#FUNCTIONS-JSONB-OP-TABLE
/// - For arrays: tests if ANY of the strings in the right array exist as array elements
/// - For objects: tests if ANY of the strings in the right array exist as top-level keys
pub(crate) fn collection_contains_any_string_dyn(
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
        if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
            results.append(false);
            continue;
        }

        let left_value = ScalarValue::try_from_array(&left, i)?;
        let right_value = ScalarValue::try_from_array(&right, i)?;

        let contains = collection_contains_any_string_scalar(&left_value, &right_value)?;
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

/// Test if collection scalar contains ANY of the strings in the array scalar, PostgreSQL `?|` operator
fn collection_contains_any_string_scalar(
    left: &ScalarValue,
    right: &ScalarValue,
) -> Result<bool> {
    match (left, right) {
        // Struct: test if ANY of the strings exist as top-level keys
        (ScalarValue::Struct(struct_array), ScalarValue::List(string_list)) => {
            let search_strings = extract_scalar_values_from_list(string_list)?;

            for search_string in &search_strings {
                if let ScalarValue::Utf8(Some(field_name)) = search_string {
                    if struct_array.column_by_name(field_name).is_some() {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }

        // List: test if ANY of the strings exist as array elements
        (ScalarValue::List(list_array), ScalarValue::List(string_list)) => {
            let list_values = extract_scalar_values_from_list(list_array)?;
            let search_strings = extract_scalar_values_from_list(string_list)?;

            for search_string in &search_strings {
                if let ScalarValue::Utf8(Some(s)) = search_string {
                    for element in &list_values {
                        if let ScalarValue::Utf8(Some(e)) = element {
                            if e == s {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
            Ok(false)
        }

        _ => Ok(false),
    }
}

/// Test if collection contains ALL of the strings in the array, PostgreSQL `?&` operator
///
/// Rule for `?&` operator:
/// https://www.postgresql.org/docs/18/functions-json.html#FUNCTIONS-JSONB-OP-TABLE
/// - For arrays: tests if ALL of the strings in the right array exist as array elements
/// - For objects: tests if ALL of the strings in the right array exist as top-level keys
pub(crate) fn collection_contains_all_strings_dyn(
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
        if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
            results.append(false);
            continue;
        }

        let left_value = ScalarValue::try_from_array(&left, i)?;
        let right_value = ScalarValue::try_from_array(&right, i)?;

        let contains = collection_contains_all_strings_scalar(&left_value, &right_value)?;
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

/// Test if collection scalar contains ALL of the strings in the array scalar, PostgreSQL `?&` operator
fn collection_contains_all_strings_scalar(
    left: &ScalarValue,
    right: &ScalarValue,
) -> Result<bool> {
    match (left, right) {
        // Struct: test if ALL of the strings exist as top-level keys
        (ScalarValue::Struct(struct_array), ScalarValue::List(string_list)) => {
            let search_strings = extract_scalar_values_from_list(string_list)?;

            for search_string in &search_strings {
                if let ScalarValue::Utf8(Some(field_name)) = search_string {
                    if struct_array.column_by_name(field_name).is_none() {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }

        // List: test if ALL of the strings exist as array elements
        (ScalarValue::List(list_array), ScalarValue::List(string_list)) => {
            let list_values = extract_scalar_values_from_list(list_array)?;
            let search_strings = extract_scalar_values_from_list(string_list)?;

            for search_string in &search_strings {
                if let ScalarValue::Utf8(Some(s)) = search_string {
                    let mut found = false;
                    for element in &list_values {
                        if let ScalarValue::Utf8(Some(e)) = element {
                            if e == s {
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }

        _ => Ok(false),
    }
}

/// Scalar version of collection_contains_dyn - array contains scalar
pub(crate) fn collection_contains_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let mut results = BooleanBufferBuilder::new(left.len());

    for i in 0..left.len() {
        if left.is_null(i) {
            results.append(false);
            continue;
        }

        let left_value = match ScalarValue::try_from_array(left, i) {
            Ok(value) => value,
            Err(e) => return Some(Err(e)),
        };

        let contains = match jsonb_contains_scalar(&left_value, &right) {
            Ok(contains) => contains,
            Err(e) => return Some(Err(e)),
        };
        results.append(contains);
    }

    let data = unsafe {
        ArrayDataBuilder::new(DataType::Boolean)
            .len(left.len())
            .buffers(vec![results.into()])
            .nulls(left.nulls().cloned())
            .build_unchecked()
    };
    Some(Ok(Arc::new(BooleanArray::from(data))))
}

/// Scalar version of collection_contains_string_dyn - array contains string scalar
pub(crate) fn collection_contains_string_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let mut results = BooleanBufferBuilder::new(left.len());

    for i in 0..left.len() {
        if left.is_null(i) {
            results.append(false);
            continue;
        }

        let left_value = match ScalarValue::try_from_array(left, i) {
            Ok(value) => value,
            Err(e) => return Some(Err(e)),
        };

        let contains = match collection_contains_string_scalar(&left_value, &right) {
            Ok(contains) => contains,
            Err(e) => return Some(Err(e)),
        };
        results.append(contains);
    }

    let data = unsafe {
        ArrayDataBuilder::new(DataType::Boolean)
            .len(left.len())
            .buffers(vec![results.into()])
            .nulls(left.nulls().cloned())
            .build_unchecked()
    };
    Some(Ok(Arc::new(BooleanArray::from(data))))
}

/// Scalar version of collection_contains_any_dyn - array contains ANY of the strings in scalar list
pub(crate) fn collection_contains_any_string_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let mut results = BooleanBufferBuilder::new(left.len());

    for i in 0..left.len() {
        if left.is_null(i) {
            results.append(false);
            continue;
        }

        let left_value = match ScalarValue::try_from_array(left, i) {
            Ok(value) => value,
            Err(e) => return Some(Err(e)),
        };

        let contains = match collection_contains_any_string_scalar(&left_value, &right) {
            Ok(contains) => contains,
            Err(e) => return Some(Err(e)),
        };
        results.append(contains);
    }

    let data = unsafe {
        ArrayDataBuilder::new(DataType::Boolean)
            .len(left.len())
            .buffers(vec![results.into()])
            .nulls(left.nulls().cloned())
            .build_unchecked()
    };
    Some(Ok(Arc::new(BooleanArray::from(data))))
}

/// Scalar version of collection_contains_all_dyn - array contains ALL of the strings in scalar list
pub(crate) fn collection_contains_all_strings_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let mut results = BooleanBufferBuilder::new(left.len());

    for i in 0..left.len() {
        if left.is_null(i) {
            results.append(false);
            continue;
        }

        let left_value = match ScalarValue::try_from_array(left, i) {
            Ok(value) => value,
            Err(e) => return Some(Err(e)),
        };

        let contains = match collection_contains_all_strings_scalar(&left_value, &right) {
            Ok(contains) => contains,
            Err(e) => return Some(Err(e)),
        };
        results.append(contains);
    }

    let data = unsafe {
        ArrayDataBuilder::new(DataType::Boolean)
            .len(left.len())
            .buffers(vec![results.into()])
            .nulls(left.nulls().cloned())
            .build_unchecked()
    };
    Some(Ok(Arc::new(BooleanArray::from(data))))
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
        let inner_left_fields =
            Fields::from(vec![Field::new("bar", DataType::Utf8, true)]);
        let inner_left_array = Arc::new(StringArray::from(vec!["baz"])) as ArrayRef;
        let inner_left_struct =
            StructArray::new(inner_left_fields.clone(), vec![inner_left_array], None);

        let outer_left_fields = Fields::from(vec![Field::new(
            "foo",
            DataType::Struct(inner_left_fields),
            true,
        )]);
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

        let outer_right_fields = Fields::from(vec![Field::new(
            "foo",
            DataType::Struct(inner_right_fields),
            true,
        )]);
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

    #[test]
    fn test_scalar_contains_scalar() -> Result<()> {
        // Test scalar version of collection_contains_dyn
        let left = Arc::new(StringArray::from(vec!["hello", "world"])) as ArrayRef;
        let right = ScalarValue::Utf8(Some("hello".to_string()));
        let result = collection_contains_dyn_scalar(left.as_ref(), right)
            .unwrap()
            .unwrap();
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_scalar_contains_string_scalar() -> Result<()> {
        // Test scalar version of collection_contains_string_dyn
        // Create a struct array with fields
        let fields = Fields::from(vec![
            Field::new("foo", DataType::Int32, true),
            Field::new("bar", DataType::Utf8, true),
        ]);

        let foo_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bar_array = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;

        let left = Arc::new(StructArray::new(fields, vec![foo_array, bar_array], None))
            as ArrayRef;
        let right = ScalarValue::Utf8(Some("foo".to_string()));

        let result = collection_contains_string_dyn_scalar(left.as_ref(), right)
            .unwrap()
            .unwrap();
        let expected = BooleanArray::from(vec![true, true]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_collection_contains_any_struct() -> Result<()> {
        // Test struct contains ANY of the strings
        let fields = Fields::from(vec![
            Field::new("foo", DataType::Int32, true),
            Field::new("bar", DataType::Utf8, true),
        ]);

        let foo_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bar_array = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;

        let left = Arc::new(StructArray::new(fields, vec![foo_array, bar_array], None));

        // Create right array with list of strings using ListBuilder
        let mut builder = ListBuilder::new(StringBuilder::new());

        // First row: ["foo", "baz"]
        builder.values().append_value("foo");
        builder.values().append_value("baz");
        builder.append(true);

        // Second row: ["qux", "quux"]
        builder.values().append_value("qux");
        builder.values().append_value("quux");
        builder.append(true);

        let right = Arc::new(builder.finish());

        let result = collection_contains_any_string_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_collection_contains_any_list() -> Result<()> {
        // Test list contains ANY of the strings
        let mut left_builder = ListBuilder::new(StringBuilder::new());

        // First row: ["a", "b", "c"]
        left_builder.values().append_value("a");
        left_builder.values().append_value("b");
        left_builder.values().append_value("c");
        left_builder.append(true);

        // Second row: ["x", "y"]
        left_builder.values().append_value("x");
        left_builder.values().append_value("y");
        left_builder.append(true);

        let left = Arc::new(left_builder.finish());

        // Create right array with list of strings using ListBuilder
        let mut right_builder = ListBuilder::new(StringBuilder::new());

        // First row: ["b", "d"] - contains "b"
        right_builder.values().append_value("b");
        right_builder.values().append_value("d");
        right_builder.append(true);

        // Second row: ["z", "w"] - contains neither
        right_builder.values().append_value("z");
        right_builder.values().append_value("w");
        right_builder.append(true);

        let right = Arc::new(right_builder.finish());

        let result = collection_contains_any_string_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_collection_contains_all_struct() -> Result<()> {
        // Test struct contains ALL of the strings
        let fields = Fields::from(vec![
            Field::new("foo", DataType::Int32, true),
            Field::new("bar", DataType::Utf8, true),
            Field::new("baz", DataType::Float64, true),
        ]);

        let foo_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bar_array = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let baz_array = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;

        let left = Arc::new(StructArray::new(
            fields,
            vec![foo_array, bar_array, baz_array],
            None,
        ));

        // Create right array with list of strings using ListBuilder
        let mut builder = ListBuilder::new(StringBuilder::new());

        // First row: ["foo", "bar"] - contains both
        builder.values().append_value("foo");
        builder.values().append_value("bar");
        builder.append(true);

        // Second row: ["foo", "qux"] - missing "qux"
        builder.values().append_value("foo");
        builder.values().append_value("qux");
        builder.append(true);

        let right = Arc::new(builder.finish());

        let result = collection_contains_all_strings_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_collection_contains_all_list() -> Result<()> {
        // Test list contains ALL of the strings
        let mut left_builder = ListBuilder::new(StringBuilder::new());

        // First row: ["a", "b", "c"]
        left_builder.values().append_value("a");
        left_builder.values().append_value("b");
        left_builder.values().append_value("c");
        left_builder.append(true);

        // Second row: ["x", "y"]
        left_builder.values().append_value("x");
        left_builder.values().append_value("y");
        left_builder.append(true);

        let left = Arc::new(left_builder.finish());

        // Create right array with list of strings using ListBuilder
        let mut right_builder = ListBuilder::new(StringBuilder::new());

        // First row: ["a", "b"] - contains both
        right_builder.values().append_value("a");
        right_builder.values().append_value("b");
        right_builder.append(true);

        // Second row: ["x", "z"] - missing "z"
        right_builder.values().append_value("x");
        right_builder.values().append_value("z");
        right_builder.append(true);

        let right = Arc::new(right_builder.finish());

        let result = collection_contains_all_strings_dyn(left, right)?;
        let expected = BooleanArray::from(vec![true, false]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_scalar_contains_any_scalar() -> Result<()> {
        // Test scalar version of collection_contains_any_dyn
        let fields = Fields::from(vec![
            Field::new("foo", DataType::Int32, true),
            Field::new("bar", DataType::Utf8, true),
        ]);

        let foo_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bar_array = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;

        let left = Arc::new(StructArray::new(fields, vec![foo_array, bar_array], None))
            as ArrayRef;

        // Create scalar list of strings using ListBuilder
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("foo");
        builder.values().append_value("baz");
        builder.append(true);
        let string_list = Arc::new(builder.finish());
        let right = ScalarValue::List(string_list);

        let result = collection_contains_any_string_dyn_scalar(left.as_ref(), right)
            .unwrap()
            .unwrap();
        let expected = BooleanArray::from(vec![true, true]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn test_scalar_contains_all_scalar() -> Result<()> {
        // Test scalar version of collection_contains_all_dyn
        let fields = Fields::from(vec![
            Field::new("foo", DataType::Int32, true),
            Field::new("bar", DataType::Utf8, true),
            Field::new("baz", DataType::Float64, true),
        ]);

        let foo_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bar_array = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let baz_array = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;

        let left = Arc::new(StructArray::new(
            fields,
            vec![foo_array, bar_array, baz_array],
            None,
        )) as ArrayRef;

        // Create scalar list of strings using ListBuilder
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("foo");
        builder.values().append_value("bar");
        builder.append(true);
        let string_list = Arc::new(builder.finish());
        let right = ScalarValue::List(string_list);

        let result = collection_contains_all_strings_dyn_scalar(left.as_ref(), right)
            .unwrap()
            .unwrap();
        let expected = BooleanArray::from(vec![true, true]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }
}
