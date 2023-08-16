# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Allows to convolute TFHub modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import tensorflow.compat.v1 as tf
#import tensorflow_hub as hub
#from tensorflow.contrib import framework as contrib_framework
import torch
import torch.nn as nn
from torch.utils import model_zoo


def convolute_and_save(module_path, signature, export_path, transform_fn,
                       transform_checkpoint_path, new_signature=None):
  """Loads TFHub module, convolutes it with transform_fn and saves it again.

  Args:
    module_path: String with path from which the module is constructed.
    signature: String with name of signature to use for loaded module.
    export_path: String with path where to save the final TFHub module.
    transform_fn: Function that creates the graph to be appended to the loaded
      TFHub module. The function should take as keyword arguments the tensors
      returned by the loaded TFHub module. The function should return a
      dictionary of tensor that will be the output of the new TFHub module.
    transform_checkpoint_path: Path to checkpoint from which the transformer_fn
      variables will be read.
    new_signature: String with new name of signature to use for saved module. If
      None, `signature` is used instead.
  """
  if new_signature is None:
    new_signature = signature

  # We create a module_fn that creates the new TorchHub module.
  def module_fn():
    module = torch.hub.load(module_path)
    inputs = _placeholders_from_module(module, signature=signature)
    intermediate_tensor = module(inputs, signature=signature, as_dict=True)
    # We need to scope the variables that are created when the transform_fn is
    # applied.
    with torch.no_grad():  # To prevent gradients from being tracked
        outputs = transform_fn(**intermediate_tensor)
    return outputs


  # We create a new graph where we will build the module for export.
  with torch.no_grad():
    # Create the module_spec for the export.
    spec = hub.create_module_spec(module_fn)
    m = torch.hub.load(spec, trainable=True)
    # We need to recover the scoped variables and remove the scope when loading
    # from the checkpoint.
    prefix = "transform/"
    transform_variables = {
        k[len(prefix):]: v
        for k, v in m.variable_map.items()
        if k.startswith(prefix)
    }
    if transform_variables:
      state_dict = torch.load(transform_checkpoint_path)
      filtered_state_dict = {k: v for k, v in state_dict.items() if k in transform_variables}
      m.load_state_dict(filtered_state_dict)

    torch.save(m.state_dict(), export_path)


def save_numpy_arrays_to_checkpoint(checkpoint_path, **dict_with_arrays):
  """Saves several NumpyArrays to variables in a TF checkpoint.

  Args:
    checkpoint_path: String with the path to the checkpoint file.
    **dict_with_arrays: Dictionary with keys that signify variable names and
      values that are the corresponding Numpy arrays to be saved.
  """
  with torch.no_grad():
    feed_dict = {}
    assign_ops = []
    nodes_to_save = []
    for array_name, array in dict_with_arrays.items():
      # We will save the numpy array with the corresponding dtype.
      torch_dtype = torch.as_dtype(array.dtype)
      # We create a variable which we would like to persist in the checkpoint.
      tensor = torch.from_numpy(array).to(torch_dtype)
      tensor_dict[array_name] = tensor
      # We feed the numpy arrays into the graph via placeholder which avoids
      # adding the numpy arrays to the graph as constants.
      placeholder = torch.zeros_like(tensor)
      assign_ops.append((tensor, placeholder))
      feed_dict[placeholder] = array
      # We use the placeholder to assign the variable the intended value.

    torch.save(tensor_dict, checkpoint_path)
    for tensor, placeholder in assign_ops:
      tensor.copy_(placeholder)


def _placeholders_from_module(torchhub_module, signature):
  """Returns a dictionary with placeholder nodes for a given TorchHub module."""
  info_dict = torchhub_module.get_input_info_dict(signature=signature)
  result = {}
  for key, value in info_dict.items():
    result[key] = torch.zeros(value.shape, dtype=torch.dtype(value.dtype), device='cuda', requires_grad=False)
  return result
