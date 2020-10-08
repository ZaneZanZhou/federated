# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple ClientData based on in-memory tensor slices."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation import client_data


class FromTensorSlicesClientData(client_data.ClientData):
  """ClientData based on `tf.data.Dataset.from_tensor_slices`."""

  def __init__(self, tensor_slices_dict):
    """Constructs the object from a dictionary of client data.

    Note: All clients are required to have non-empty data.

    Args:
      tensor_slices_dict: A dictionary keyed by client_id, where values are
        structures suitable for passing to `tf.data.Dataset.from_tensor_slices`.

    Raises:
      ValueError: If a client with no data is found.
      TypeError: If `tensor_slices_dict` is not a dictionary, or its values are
        not either strictly lists or strictly dictionaries.
      TypeError: If flattened values in tensor_slices_dict convert to different
        TensorFlow data types.
    """
    py_typecheck.check_type(tensor_slices_dict, dict)
    values = list(tensor_slices_dict.values())
    example_value = values[0]
    # The values must be lists or dictionaries.
    py_typecheck.check_type(example_value, (list, dict))
    # The values must all be the same.
    for value in values:
      py_typecheck.check_type(value, type(example_value))

    def check_types_match(tensors, expected_dtypes):
      for tensor, expected_dtype in zip(tensors, expected_dtypes):
        if tensor.dtype is not expected_dtype:
          raise TypeError(
              'The input tensor_slices_dict must have entries that convert '
              'to identical TensorFlow data types, but found two different '
              'entries with values of %s and %s' %
              (expected_dtype, tensor.dtype))

    if isinstance(example_value, dict):

      # This is needed to keep text data that was loosely specified in a list
      # together in a common object (a tf.RaggedTensor), for correct flattening.
      def convert_any_lists_of_strings_or_bytes_to_ragged_tensors(value):
        for key, entries in value.items():
          if isinstance(entries, list) and isinstance(entries[0], (bytes, str)):
            value[key] = tf.ragged.constant(entries)
        return value

      example_value = convert_any_lists_of_strings_or_bytes_to_ragged_tensors(
          example_value)
      self._example_value_structure = example_value
      self._dtypes = [
          tf.constant(x).dtype for x in tf.nest.flatten(example_value)
      ]

      for v in values:
        v = convert_any_lists_of_strings_or_bytes_to_ragged_tensors(v)
        check_types_match([tf.constant(x) for x in tf.nest.flatten(v)],
                          self._dtypes)
    else:
      self._example_value_structure = None
      self._dtypes = [tf.constant(example_value).dtype]
      for v in values:
        check_types_match([tf.constant(v)], self._dtypes)

    self._tensor_slices_dict = tensor_slices_dict
    example_dataset = self.create_tf_dataset_for_client(self.client_ids[0])
    self._element_type_structure = example_dataset.element_spec

    self._dataset_computation = None

  @property
  def client_ids(self):
    return list(self._tensor_slices_dict.keys())

  def create_tf_dataset_for_client(self, client_id):
    tensor_slices = self._tensor_slices_dict[client_id]
    if tensor_slices:
      return tf.data.Dataset.from_tensor_slices(tensor_slices)
    else:
      raise ValueError('No data found for client {}'.format(client_id))

  @property
  def element_type_structure(self):
    return self._element_type_structure

  @property
  def dataset_computation(self):
    if self._dataset_computation is None:

      @computations.tf_computation(tf.string)
      def construct_dataset(client_id):
        keys = tf.constant(list(self._tensor_slices_dict.keys()))

        client_id_valid = tf.math.reduce_any(tf.math.equal(client_id, keys))
        assert_op = tf.Assert(client_id_valid,
                              ['No data found for client ', client_id])
        with tf.control_dependencies([assert_op]):
          # Serialize and flatten (if necessary) the contents of the input dict.
          serialized_flat_values = [[] for _ in range(len(self._dtypes))]
          for v in self._tensor_slices_dict.values():
            flat_values = tf.nest.flatten(v) if isinstance(v, dict) else [v]
            for i, x in enumerate(flat_values):
              serialized_flat_values[i].append(
                  tf.io.serialize_tensor(tf.constant(x)))
          # Put the data into a TF hash table.
          hash_tables = []
          for i in range(len(self._dtypes)):
            hash_tables.append(
                tf.lookup.StaticHashTable(
                    initializer=tf.lookup.KeyValueTensorInitializer(
                        keys=keys, values=serialized_flat_values[i]),
                    # Note: This default_value should never be encountered, as
                    # we do a check above that the client_id is in the set of
                    # keys.
                    default_value='unknown_value'))
          # Recover data relating to the given client_id from the hash table.
          tensor_slices_list = [
              tf.io.parse_tensor(table.lookup(client_id), out_type=dtype)
              for table, dtype in zip(hash_tables, self._dtypes)
          ]
          # If necessary, unflatten the values back into the desired structure.
          if self._example_value_structure is not None:
            tensor_slices = tf.nest.pack_sequence_as(
                self._example_value_structure, tensor_slices_list)
            for k, v in self._example_value_structure.items():
              tensor_slices[k] = tf.stack(tensor_slices[k])
              tensor_slices[k].set_shape(np.array(v).shape)
          else:
            tensor_slices = tensor_slices_list[0]

          return tf.data.Dataset.from_tensor_slices(tensor_slices)

      self._dataset_computation = construct_dataset

    return self._dataset_computation
