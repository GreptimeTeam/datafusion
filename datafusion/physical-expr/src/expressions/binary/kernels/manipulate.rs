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
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use datafusion_common::{plan_err, Result, ScalarValue};

/// Implements postgres like `||` operator to concat two values
/// According to definition in postgres:
///
/// - Concatenating two arrays generates an array containing all the elements of
///   each input.
/// - Concatenating two objects generates an object containing the union of
///   their keys, taking the second object's value when there are duplicate
///   keys.
/// - All other cases are treated by converting a non-array input into a
///   single-element array, and then proceeding as for two arrays.
/// - Does not operate recursively: only the top-level array or object structure
///   is merged.
pub(crate) fn collection_concat_dyn(
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
    // base
    match (left.data_type(), right.data_type()) {
        (DataType::List(list_type_left), DataType::List(list_type_right)) => {
            if list_type_left.data_type() != list_type_right.data_type() {
                return plan_err!(
                    "Cannot concat two lists of different data types: {}, {}",
                    list_type_left.data_type(),
                    list_type_right.data_type()
                );
            }

            let left_list = left.as_any().downcast_ref::<ListArray>().unwrap();
            let right_list = right.as_any().downcast_ref::<ListArray>().unwrap();

            // Create new offsets by concatenating the elements from both lists
            let mut new_offsets = Vec::with_capacity(left_list.len() + 1);
            new_offsets.push(0);

            let mut new_values = Vec::with_capacity(left_list.len() + right_list.len());

            for i in 0..left_list.len() {
                let left_start = left_list.value_offsets()[i] as usize;
                let left_end = left_list.value_offsets()[i + 1] as usize;
                let right_start = right_list.value_offsets()[i] as usize;
                let right_end = right_list.value_offsets()[i + 1] as usize;

                // Add left elements
                for j in left_start..left_end {
                    new_values.push(left_list.values().slice(j, 1));
                }
                // Add right elements
                for j in right_start..right_end {
                    new_values.push(right_list.values().slice(j, 1));
                }

                let new_offset = new_values.len() as i32;
                new_offsets.push(new_offset);
            }

            // Convert Vec<ArrayRef> to &[&dyn Array] for concat
            let value_refs: Vec<&dyn Array> =
                new_values.iter().map(|a| a.as_ref()).collect();
            let concatenated_values = arrow::compute::concat(&value_refs)?;

            // Create new ListArray
            Ok(Arc::new(ListArray::try_new(
                Arc::clone(list_type_left),
                OffsetBuffer::new(new_offsets.into()),
                concatenated_values,
                nulls,
            )?))
        }
        (DataType::Struct(left_fields), DataType::Struct(right_fields)) => {
            let left_struct = left.as_any().downcast_ref::<StructArray>().unwrap();
            let right_struct = right.as_any().downcast_ref::<StructArray>().unwrap();

            // Create a union of fields, preferring fields from the right when duplicates exist
            let mut merged_fields = Vec::new();
            let mut merged_columns = Vec::new();

            // First, add all fields from left struct that don't exist in right struct
            for (i, left_field) in left_fields.iter().enumerate() {
                if !right_fields.iter().any(|f| f.name() == left_field.name()) {
                    merged_fields.push(Arc::clone(left_field));
                    merged_columns.push(Arc::clone(left_struct.column(i)));
                }
            }

            // Then add all fields from right struct (this handles duplicates by taking the right value)
            for (i, right_field) in right_fields.iter().enumerate() {
                merged_fields.push(Arc::clone(right_field));
                merged_columns.push(Arc::clone(right_struct.column(i)));
            }

            // Create the merged struct array
            Ok(Arc::new(StructArray::try_new(
                merged_fields.into(),
                merged_columns,
                nulls,
            )?))
        }
        (other1, other2) => {
            // TODO: we will support more data type by creating list of items
            // from both side.
            plan_err!("Unsupported data types {}, {} for concat operation collection_concat_dyn", other1, other2)
        }
    }
}

/// delete key from left collection
/// it can be deleting column(s) from struct , or deleting be index for list
pub(crate) fn collection_delete_key_dyn_scalar(
    left: &dyn Array,
    right: ScalarValue,
) -> Option<Result<ArrayRef>> {
    match (left.data_type(), right.data_type()) {
        (DataType::Struct(_), DataType::Utf8 | DataType::Utf8View) => {
            let struct_array = left.as_any().downcast_ref::<StructArray>().unwrap();
            match right {
                ScalarValue::Utf8(Some(key))
                | ScalarValue::LargeUtf8(Some(key))
                | ScalarValue::Utf8View(Some(key)) => {
                    Some(struct_delete_keys(struct_array, &[key]))
                }
                _ => Some(Ok(Arc::new(struct_array.clone()))),
            }
        }
        (DataType::Struct(_), DataType::List(list_type)) => {
            if matches!(list_type.data_type(), DataType::Utf8 | DataType::Utf8View) {
                let struct_array = left.as_any().downcast_ref::<StructArray>().unwrap();
                if let ScalarValue::List(keys_array) = right {
                    let keys_to_delete: Vec<String> = if keys_array.is_null(0) {
                        vec![]
                    } else {
                        let list = keys_array.value(0);
                        let string_array = list.as_any().downcast_ref::<StringArray>()?;
                        string_array
                            .into_iter()
                            .filter_map(|s| s.map(|s| s.to_string()))
                            .collect()
                    };

                    Some(struct_delete_keys(struct_array, &keys_to_delete))
                } else {
                    Some(Ok(Arc::new(struct_array.clone())))
                }
            } else {
                Some(plan_err!(
                    "List for struct deletion must contain string keys"
                ))
            }
        }
        (
            DataType::List(_),
            DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8,
        ) => {
            let list_array = left.as_any().downcast_ref::<ListArray>().unwrap();
            let index_to_delete = match right {
                ScalarValue::Int8(Some(i)) => i as i32,
                ScalarValue::Int16(Some(i)) => i as i32,
                ScalarValue::Int32(Some(i)) => i,
                ScalarValue::Int64(Some(i)) => i as i32,
                ScalarValue::UInt8(Some(i)) => i as i32,
                ScalarValue::UInt16(Some(i)) => i as i32,
                ScalarValue::UInt32(Some(i)) => i as i32,
                ScalarValue::UInt64(Some(i)) => i as i32,
                _ => return Some(plan_err!("Invalid index to delete {}", right)),
            };

            Some(list_delete_index(list_array, index_to_delete))
        }
        (_other1, _other2) => None,
    }
}

fn struct_delete_keys(left: &StructArray, keys_to_delete: &[String]) -> Result<ArrayRef> {
    let fields = left.fields();
    let mut remaining_fields = Vec::new();
    let mut remaining_columns = Vec::new();

    // Filter out the fields that should be deleted
    for (i, field) in fields.iter().enumerate() {
        if !keys_to_delete.contains(field.name()) {
            remaining_fields.push(Arc::clone(field));
            remaining_columns.push(Arc::clone(left.column(i)));
        }
    }

    if remaining_fields.is_empty() {
        Ok(Arc::new(StructArray::new_empty_fields(
            left.len(),
            left.nulls().cloned(),
        )))
    } else {
        // Create the new struct array with remaining fields
        Ok(Arc::new(StructArray::try_new(
            remaining_fields.into(),
            remaining_columns,
            left.nulls().cloned(),
        )?))
    }
}

fn list_delete_index(list_array: &ListArray, index_to_delete: i32) -> Result<ArrayRef> {
    let offsets = list_array.value_offsets();
    let values = list_array.values();

    // Calculate new offsets and which values to keep
    let mut new_offsets = Vec::with_capacity(list_array.len() + 1);
    let mut indices_to_keep = Vec::new();

    new_offsets.push(0i32);

    for i in 0..list_array.len() {
        if list_array.is_null(i) {
            // Null list - no change in length
            new_offsets.push(*new_offsets.last().unwrap());
            continue;
        }

        let start = offsets[i];
        let end = offsets[i + 1];
        let list_len = end - start;

        // Calculate actual index to delete
        let actual_index = if index_to_delete < 0 {
            if -index_to_delete > list_len {
                None // Out of bounds
            } else {
                Some(list_len + index_to_delete)
            }
        } else if index_to_delete >= list_len {
            None // Out of bounds
        } else {
            Some(index_to_delete)
        };

        // Add indices to keep
        for j in 0..list_len {
            if Some(j) != actual_index {
                indices_to_keep.push(start + j);
            }
        }

        // Update offset
        let new_len = if actual_index.is_some() {
            list_len - 1
        } else {
            list_len
        };
        new_offsets.push(*new_offsets.last().unwrap() + new_len);
    }

    // Use take kernel to extract the values we want to keep
    let indices_array = Int32Array::from(indices_to_keep);
    let new_values = arrow::compute::take(values.as_ref(), &indices_array, None)?;

    // Get the field from the original list type
    let field = match list_array.data_type() {
        DataType::List(field) => Arc::clone(field),
        _ => unreachable!(),
    };

    // Create new ListArray with computed offsets
    let new_list_array = ListArray::try_new(
        field,
        OffsetBuffer::new(new_offsets.into()),
        new_values,
        list_array.nulls().cloned(),
    )?;

    Ok(Arc::new(new_list_array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Fields};

    #[test]
    fn test_list_concat() {
        // Test case 1: Concatenate two simple lists
        let left_values = Int32Array::from(vec![1, 2, 3]);
        let left_offsets = OffsetBuffer::new(vec![0, 2, 3].into());
        let left_list = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            left_offsets,
            Arc::new(left_values),
            None,
        )
        .unwrap();
        assert_eq!(left_list.len(), 2);

        let right_values = Int32Array::from(vec![4, 5, 6]);
        let right_offsets = OffsetBuffer::new(vec![0, 1, 3].into());
        let right_list = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            right_offsets,
            Arc::new(right_values),
            None,
        )
        .unwrap();
        assert_eq!(right_list.len(), 2);

        let result =
            collection_concat_dyn(Arc::new(left_list), Arc::new(right_list)).unwrap();
        let result_list = result.as_any().downcast_ref::<ListArray>().unwrap();

        // Verify the concatenated list structure
        assert_eq!(result_list.len(), 2);

        // First row: [1, 2] + [4] = [1, 2, 4]
        let first_row = result_list.value(0);
        let first_row_int = first_row.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_row_int.values(), &[1, 2, 4]);

        // Second row: [3] + [5, 6] = [3, 5, 6]
        let second_row = result_list.value(1);
        let second_row_int = second_row.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_row_int.values(), &[3, 5, 6]);

        // Test case 2: Lists with nulls
        let left_values_with_nulls = Int32Array::from(vec![Some(1), None, Some(3)]);
        let left_offsets_with_nulls = OffsetBuffer::new(vec![0, 1, 3].into());
        let left_list_with_nulls = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            left_offsets_with_nulls,
            Arc::new(left_values_with_nulls),
            None,
        )
        .unwrap();

        let right_values_with_nulls = Int32Array::from(vec![Some(4), None]);
        let right_offsets_with_nulls = OffsetBuffer::new(vec![0, 2, 2].into());
        let right_list_with_nulls = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            right_offsets_with_nulls,
            Arc::new(right_values_with_nulls),
            None,
        )
        .unwrap();

        let result_with_nulls = collection_concat_dyn(
            Arc::new(left_list_with_nulls),
            Arc::new(right_list_with_nulls),
        )
        .unwrap();
        let result_list_with_nulls = result_with_nulls
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        assert_eq!(result_list_with_nulls.len(), 2);

        // First row: [1] + [4, null] = [1, 4, null]
        let first_row_nulls = result_list_with_nulls.value(0);
        let first_row_nulls_int = first_row_nulls
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(first_row_nulls_int.value(0), 1);
        assert_eq!(first_row_nulls_int.value(1), 4);
        assert!(first_row_nulls_int.is_null(2));

        // Second row: [null, 3] + [] = [null, 3]
        let second_row_nulls = result_list_with_nulls.value(1);
        let second_row_nulls_int = second_row_nulls
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(second_row_nulls_int.is_null(0));
        assert_eq!(second_row_nulls_int.value(1), 3);

        // Test case 3: Error case - different data types
        let string_values = StringArray::from(vec!["a", "b"]);
        let string_offsets = OffsetBuffer::new(vec![0, 2].into());
        let string_list = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            string_offsets,
            Arc::new(string_values),
            None,
        )
        .unwrap();

        // Create a new right list for this test case
        let right_values_for_error = Int32Array::from(vec![4, 5, 6]);
        let right_offsets_for_error = OffsetBuffer::new(vec![0, 1, 3].into());
        let right_list_for_error = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            right_offsets_for_error,
            Arc::new(right_values_for_error),
            None,
        )
        .unwrap();

        let result_err =
            collection_concat_dyn(Arc::new(string_list), Arc::new(right_list_for_error));
        assert!(result_err.is_err());

        // Test case 4: Error case - different lengths
        let short_values = Int32Array::from(vec![7, 8]);
        let short_offsets = OffsetBuffer::new(vec![0, 2].into());
        let short_list = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            short_offsets,
            Arc::new(short_values),
            None,
        )
        .unwrap();

        // Create another new right list for this test case
        let right_values_for_len_error = Int32Array::from(vec![4, 5, 6]);
        let right_offsets_for_len_error = OffsetBuffer::new(vec![0, 1, 3].into());
        let right_list_for_len_error = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            right_offsets_for_len_error,
            Arc::new(right_values_for_len_error),
            None,
        )
        .unwrap();

        let result_len_err = collection_concat_dyn(
            Arc::new(short_list),
            Arc::new(right_list_for_len_error),
        );
        assert!(result_len_err.is_err());

        // Test case 5: Struct concatenation
        let left_int_values = Int32Array::from(vec![1, 2, 3]);
        let left_str_values = StringArray::from(vec!["a", "b", "c"]);
        let left_struct = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
            ]),
            vec![Arc::new(left_int_values), Arc::new(left_str_values)],
            None,
        )
        .unwrap();

        let right_int_values = Int32Array::from(vec![10, 20, 30]);
        let right_bool_values = BooleanArray::from(vec![Some(true), Some(false), None]);
        let right_struct = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true), // Duplicate field
                Field::new("active", DataType::Boolean, true),
            ]),
            vec![Arc::new(right_int_values), Arc::new(right_bool_values)],
            None,
        )
        .unwrap();

        let result_struct =
            collection_concat_dyn(Arc::new(left_struct), Arc::new(right_struct)).unwrap();
        let result_struct_array = result_struct
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        // Verify the merged struct has 3 fields (name from left, id and active from right)
        assert_eq!(result_struct_array.num_columns(), 3);
        assert_eq!(
            result_struct_array.column_names(),
            vec!["name", "id", "active"]
        );

        // Verify values - should use right struct's id values
        let id_column = result_struct_array.column_by_name("id").unwrap();
        let id_array = id_column.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_array.values(), &[10, 20, 30]);

        let name_column = result_struct_array.column_by_name("name").unwrap();
        let name_array = name_column.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_array.value(0), "a");
        assert_eq!(name_array.value(1), "b");
        assert_eq!(name_array.value(2), "c");

        let active_column = result_struct_array.column_by_name("active").unwrap();
        let active_array = active_column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(active_array.value(0));
        assert!(!active_array.value(1));
        assert!(active_array.is_null(2));
    }

    #[test]
    fn test_struct_concat() {
        // Test case 1: Basic struct concatenation with duplicate fields
        let left_int_values = Int32Array::from(vec![1, 2, 3]);
        let left_str_values = StringArray::from(vec!["a", "b", "c"]);
        let left_struct = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
            ]),
            vec![Arc::new(left_int_values), Arc::new(left_str_values)],
            None,
        )
        .unwrap();

        let right_int_values = Int32Array::from(vec![10, 20, 30]);
        let right_bool_values = BooleanArray::from(vec![Some(true), Some(false), None]);
        let right_struct = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true), // Duplicate field
                Field::new("active", DataType::Boolean, true),
            ]),
            vec![Arc::new(right_int_values), Arc::new(right_bool_values)],
            None,
        )
        .unwrap();

        let result_struct =
            collection_concat_dyn(Arc::new(left_struct), Arc::new(right_struct)).unwrap();
        let result_struct_array = result_struct
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        // Verify the merged struct has 3 fields (name from left, id and active from right)
        assert_eq!(result_struct_array.num_columns(), 3);

        // Verify values - should use right struct's id values (right overrides left)
        let id_column = result_struct_array.column_by_name("id").unwrap();
        let id_array = id_column.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_array.values(), &[10, 20, 30]);

        let name_column = result_struct_array.column_by_name("name").unwrap();
        let name_array = name_column.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_array.value(0), "a");
        assert_eq!(name_array.value(1), "b");
        assert_eq!(name_array.value(2), "c");

        let active_column = result_struct_array.column_by_name("active").unwrap();
        let active_array = active_column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(active_array.value(0));
        assert!(!active_array.value(1));
        assert!(active_array.is_null(2));
    }

    #[test]
    fn test_struct_delete_keys() {
        // Test case 1: Delete single key from struct
        let int_values = Int32Array::from(vec![1, 2, 3]);
        let str_values = StringArray::from(vec!["a", "b", "c"]);
        let bool_values = BooleanArray::from(vec![Some(true), Some(false), None]);
        let struct_array = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("active", DataType::Boolean, true),
            ]),
            vec![
                Arc::new(int_values),
                Arc::new(str_values),
                Arc::new(bool_values),
            ],
            None,
        )
        .unwrap();

        // Delete "name" field
        let result = struct_delete_keys(&struct_array, &["name".to_string()]).unwrap();
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify the result has 2 fields (id and active)
        assert_eq!(result_struct.num_columns(), 2);
        assert_eq!(result_struct.column_names(), vec!["id", "active"]);

        // Verify values are preserved
        let id_column = result_struct.column_by_name("id").unwrap();
        let id_array = id_column.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_array.values(), &[1, 2, 3]);

        let active_column = result_struct.column_by_name("active").unwrap();
        let active_array = active_column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(active_array.value(0));
        assert!(!active_array.value(1));
        assert!(active_array.is_null(2));

        // Test case 2: Delete multiple keys
        let result2 =
            struct_delete_keys(&struct_array, &["id".to_string(), "active".to_string()])
                .unwrap();
        let result_struct2 = result2.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify the result has only "name" field
        assert_eq!(result_struct2.num_columns(), 1);
        assert_eq!(result_struct2.column_names(), vec!["name"]);

        let name_column = result_struct2.column_by_name("name").unwrap();
        let name_array = name_column.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_array.value(0), "a");
        assert_eq!(name_array.value(1), "b");
        assert_eq!(name_array.value(2), "c");

        // Test case 3: Delete non-existent key (should be no-op)
        let result3 =
            struct_delete_keys(&struct_array, &["nonexistent".to_string()]).unwrap();
        let result_struct3 = result3.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify all fields are preserved
        assert_eq!(result_struct3.num_columns(), 3);
        assert_eq!(result_struct3.column_names(), vec!["id", "name", "active"]);

        // Test case 4: Delete all keys (should result in empty struct)
        let result4 = struct_delete_keys(
            &struct_array,
            &["id".to_string(), "name".to_string(), "active".to_string()],
        )
        .unwrap();
        let result_struct4 = result4.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify empty struct
        assert_eq!(result_struct4.num_columns(), 0);
    }

    #[test]
    fn test_collection_delete_key_dyn_scalar() {
        // Test case 1: Delete single key using string scalar
        let int_values = Int32Array::from(vec![1, 2, 3]);
        let str_values = StringArray::from(vec!["a", "b", "c"]);
        let bool_values = BooleanArray::from(vec![Some(true), Some(false), None]);
        let struct_array = StructArray::try_new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
                Field::new("active", DataType::Boolean, true),
            ]),
            vec![
                Arc::new(int_values),
                Arc::new(str_values),
                Arc::new(bool_values),
            ],
            None,
        )
        .unwrap();

        // Delete "name" field using string scalar
        let result = collection_delete_key_dyn_scalar(
            &struct_array,
            ScalarValue::Utf8(Some("name".to_string())),
        )
        .unwrap()
        .unwrap();
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify the result has 2 fields (id and active)
        assert_eq!(result_struct.num_columns(), 2);
        assert_eq!(result_struct.column_names(), vec!["id", "active"]);

        // Test case 2: Delete multiple keys using list scalar
        let mut keys_array_builder = ListBuilder::new(StringBuilder::new());
        keys_array_builder.values().append_option(Some("id"));
        keys_array_builder.values().append_option(Some("active"));
        keys_array_builder.append(true);
        let keys_array = keys_array_builder.finish();
        let keys_scalar = ScalarValue::List(Arc::new(keys_array));

        let result2 = collection_delete_key_dyn_scalar(&struct_array, keys_scalar)
            .unwrap()
            .unwrap();
        let result_struct2 = result2.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify the result has only "name" field
        assert_eq!(result_struct2.num_columns(), 1);
        assert_eq!(result_struct2.column_names(), vec!["name"]);

        // Test case 3: Null scalar (should return original struct)
        let result3 =
            collection_delete_key_dyn_scalar(&struct_array, ScalarValue::Utf8(None))
                .unwrap()
                .unwrap();
        let result_struct3 = result3.as_any().downcast_ref::<StructArray>().unwrap();

        // Verify all fields are preserved
        assert_eq!(result_struct3.num_columns(), 3);
        assert_eq!(result_struct3.column_names(), vec!["id", "name", "active"]);
    }

    #[test]
    fn test_list_delete_index() {
        // Test case 1: Delete item at index 1 from list
        let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let offsets = OffsetBuffer::new(vec![0, 3, 6].into());
        let list_array = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets,
            Arc::new(values),
            None,
        )
        .unwrap();

        // Delete index 1 from list
        let result =
            collection_delete_key_dyn_scalar(&list_array, ScalarValue::Int32(Some(1)))
                .unwrap()
                .unwrap();
        let result_list = result.as_any().downcast_ref::<ListArray>().unwrap();

        // Verify the result has 2 rows
        assert_eq!(result_list.len(), 2);

        // First row: [1, 2, 3] with index 1 deleted -> [1, 3]
        let first_row = result_list.value(0);
        let first_row_int = first_row.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_row_int.values(), &[1, 3]);

        // Second row: [4, 5, 6] with index 1 deleted -> [4, 6]
        let second_row = result_list.value(1);
        let second_row_int = second_row.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_row_int.values(), &[4, 6]);

        // Test case 2: Delete negative index (-1 means last element)
        let result2 =
            collection_delete_key_dyn_scalar(&list_array, ScalarValue::Int32(Some(-1)))
                .unwrap()
                .unwrap();
        let result_list2 = result2.as_any().downcast_ref::<ListArray>().unwrap();

        // First row: [1, 2, 3] with last element deleted -> [1, 2]
        let first_row2 = result_list2.value(0);
        let first_row_int2 = first_row2.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_row_int2.values(), &[1, 2]);

        // Second row: [4, 5, 6] with last element deleted -> [4, 5]
        let second_row2 = result_list2.value(1);
        let second_row_int2 = second_row2.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_row_int2.values(), &[4, 5]);

        // Test case 3: Delete index out of bounds (should be no-op)
        let result3 =
            collection_delete_key_dyn_scalar(&list_array, ScalarValue::Int32(Some(10)))
                .unwrap()
                .unwrap();
        let result_list3 = result3.as_any().downcast_ref::<ListArray>().unwrap();

        // Verify lists are unchanged
        let first_row3 = result_list3.value(0);
        let first_row_int3 = first_row3.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_row_int3.values(), &[1, 2, 3]);

        let second_row3 = result_list3.value(1);
        let second_row_int3 = second_row3.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_row_int3.values(), &[4, 5, 6]);

        // Test case 4: Delete from list with nulls
        let values_with_nulls = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
        let offsets_with_nulls = OffsetBuffer::new(vec![0, 2, 4].into());
        let list_array_with_nulls = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            offsets_with_nulls,
            Arc::new(values_with_nulls),
            None,
        )
        .unwrap();

        // Delete index 0 from list
        let result4 = collection_delete_key_dyn_scalar(
            &list_array_with_nulls,
            ScalarValue::Int32(Some(0)),
        )
        .unwrap()
        .unwrap();
        let result_list4 = result4.as_any().downcast_ref::<ListArray>().unwrap();

        // First row: [1, null] with index 0 deleted -> [null]
        let first_row4 = result_list4.value(0);
        let first_row_int4 = first_row4.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_row_int4.len(), 1);
        assert!(first_row_int4.is_null(0));

        // Second row: [3, 4] with index 0 deleted -> [4]
        let second_row4 = result_list4.value(1);
        let second_row_int4 = second_row4.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_row_int4.values(), &[4]);
    }
}
