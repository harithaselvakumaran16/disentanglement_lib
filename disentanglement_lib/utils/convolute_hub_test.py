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

"""Tests for convolute_hub.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.utils import convolute_hub
import numpy as np
#import tensorflow.compat.v1 as tf
#import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.testing import torch_testing
from absl.testing import absltest


class ConvoluteHubTest(torch_testing.TestCase):

  def test_convolute(self):

    # Create variables to use.
    random_state = np.random.RandomState(0)
    data = random_state.normal(size=(5, 10))
    variable1 = random_state.normal(size=(10, 6))
    variable2 = random_state.normal(size=(6, 2))

    # Save variables to checkpoint.
    checkpoint_path = os.path.join(self.get_temp_dir(), "checkpoint.ckpt")
    convolute_hub.save_numpy_arrays_to_checkpoint(
        checkpoint_path, variable1=variable1, variable2=variable2)

    # Save a TFHub module that we will convolute.
    module_path = os.path.join(self.get_temp_dir(), "module_path")
    def module_fn():
      tensor = torch.tensor(dtype=torch.float64, shape = (None,10))
      variable1 = nn.Parameter(torch.randn(10, 6, dtype=torch.float64))
      output = torch.matmul(tensor, variable1)
      torch.hub.add_signature(
          name="multiplication1",
          inputs={"tensor": tensor},
          outputs={"tensor": output})
    spec = torch.hub.create_module_spec(module_fn)
    torch.jit.save(spec, module_path))

    # Function used for the convolution.
    def _operation2(tensor):
      variable2 = torch.empty((6, 2), dtype=torch.float64)
      return dict(tensor=torch.matmul(tensor, variable2))

    # Save the convolution as a new TFHub module
    module_path_new = os.path.join(self.get_temp_dir(), "module_path_new")
    convolute_hub.convolute_and_save(module_path, "multiplication1",
                                     module_path_new, _operation2,
                                     checkpoint_path, "convoluted")

    # Check the first signature.
    with torch.hub.eval_function_for_module(module_path_new) as f:
      module_result = f(dict(tensor=data), signature="convoluted", as_dict=True)
    real_result = data.dot(variable1).dot(variable2)
    self.assertAllClose(real_result, module_result["tensor"])


if __name__ == "__main__":
  absltest.main()
