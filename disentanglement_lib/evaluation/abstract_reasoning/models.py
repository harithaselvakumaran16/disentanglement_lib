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

"""Keras models to perform abstract reasoning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.abstract_reasoning import relational_layers
import gin
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


@gin.configurable
class TwoStageModel(object):
  """Two stage model for abstract reasoning tasks.

  This class implements a flexible variation of the Wild Relation Networks model
  of Barrett et al., 2018 (https://arxiv.org/abs/1807.04225). There are two
  stages:

  1. Embedding: This embeds the patches of the PGM each indepently into a lower
    dimensional embedding (e.g., via CNN). It is also supported to take one-hot
    embeddings or integer embeddings of the ground-truth factors (as baselines).

  2. Reasoning: This performs reasoning on the embeddings of the patches of the
    PGM and returns the solution.
  """

  def __init__(self,
               embedding_model_class=gin.REQUIRED,
               reasoning_model_class=gin.REQUIRED,
               optimizer_fn=None):
    """Constructs a TwoStageModel.

    Args:
      embedding_model_class: Either `values`, `onehot`, or a class that has a
        __call__ function that takes as input a two-tuple of
        (batch_size, num_nodes, height, width, num_channels) tensors and returns
        two (batch_size, num_nodes, num_embedding_dims) tensors for both the
        context panels and the answer panels.
      reasoning_model_class: Class that has a __call__ function that takes as
        input a two-tuple of (batch_size, num_nodes, num_embedding_dims) tensors
        and returns the solution in a (batch_size,) tensor.
      optimizer_fn: Function that creates a tf.train optimizer.
    """
    if optimizer_fn is None:
      optimizer_fn = optim.Adam
    self.optimizer_fn = optimizer_fn
    self.embedding_model_class = embedding_model_class
    self.reasoning_model_class = reasoning_model_class

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model_fn."""
    del params
    is_training = (mode == "train")
    update_ops = []

    # First, embed the context and answer panels.
    if self.embedding_model_class == "values":
      # Use the integer values of the ground-truth factors.
      context_embeddings = features["context_factor_values"]
      answer_embeddings = features["answers_factor_values"]
    elif self.embedding_model_class == "onehot":
      # Use one-hot embeddings of the ground-truth factors.
      context_embeddings = features["context_factors_onehot"]
      answer_embeddings = features["answers_factors_onehot"]
    else:
      embedding_model = self.embedding_model_class()
      context_embeddings, answer_embeddings = embedding_model(
          [
              features["context"],
              features["answers"],
          ],
          training=is_training,
      )
      embedding_model.summary(print_fn=tf.logging.info)
      update_ops += embedding_model.updates

    # Apply the reasoning model.
    reasoning_model = self.reasoning_model_class()
    logits = reasoning_model([context_embeddings, answer_embeddings],
                             training=is_training)
    #reasoning_model.summary(print_fn=tf.logging.info)
    summary(reasoning_model, input_size=(num_channels, height, width))
    update_ops += reasoning_model.updates

    loss_vec = F.cross_entropy(logits), labels
    loss_mean = torch.mean(loss_vec)

    if mode == "eval":
      def metric_fn(labels, logits):
        predictions = torch.argmax(logits, dim= 1)
        accuracy = (predictions == labels).float().mean()
        return {"accuracy": accuracy.item()}

      return contrib_tpu.TPUEstimatorSpec(
          mode=mode, loss=loss_mean, eval_metrics=(metric_fn, [labels, logits]))

    if mode == "train":
      optimizer = self.optimizer_fn()  # Initialize optimizer
      optimizer.zero_grad()  # Clear gradients
      loss_mean.backward()  # Backpropagation to compute gradients
      optimizer.step()  # Update model's parameters
      return {"loss": loss_mean}

    raise NotImplementedError("Unsupported mode.")


@gin.configurable
class BaselineCNNEmbedder(nn.Module):
  """Baseline implementation where a CNN is learned from scratch."""

  def __init__(self,
               num_latent=gin.REQUIRED,
               name="BaselineCNNEmbedder",
               **kwargs):
    """Constructs a BaselineCNNEmbedder.

    Args:
      num_latent: Integer with the number of latent dimensions.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to torch.nn.Module.
    """
    super(BaselineCNNEmbedder, self).__init__(name=name, **kwargs)
                 self.name=name
                 self.embedding_layer = nn.Sequential(
                   nn.Conv2d(3,32,kernel_size=(4,4),stride=2, padding=1),
                   nn.ReLU(),
                   nn.Conv2d(32,32,kernel_size=(4,4),stride=2, padding=1),
                   nn.ReLU(),
                   nn.Conv2d(32,64,kernel_size=(4,4),stride=2, padding=1),
                   nn.ReLU(),
                   nn.Conv2d(64,64,kernel_size=(4,4),stride=2, padding=1),
                   nn.ReLU(),
                   nn.Flatten()
                 )
                 
  def call(self, inputs, **kwargs):
    context, answers = inputs
    context_embedding = self.embedding_layer(context, **kwargs)
    answers_embedding = self.embedding_layer(answers, **kwargs)
    return context_embedding, answers_embedding


@gin.configurable
class HubEmbedding(nn.Module):
  """Embed images using a pre-trained TFHub model.

  Compatible with the representation of a disentanglement_lib model.
  """

  def __init__(self, hub_path=gin.REQUIRED, name="HubEmbedding", **kwargs):
    """Constructs a HubEmbedding.

    Args:
      hub_path: Path to the PytorchHub module.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to nn.Module.
    """
    super(HubEmbedding, self).__init__(name=name, **kwargs)
    self.name=name

    def _embedder(x):
      embedder_module = torch.hub.load(hub_path, 'default', source='local') 
      embedder_module = hub.Module(hub_path)
      return embedder_module(dict(images=x), signature="representation")

    self.embedding_layer = self._embedder()

def call(self, inputs, **kwargs):
    context, answers = inputs
    context_embedding = self.embedding_layer(context, **kwargs)
    answers_embedding = self.embedding_layer(answers, **kwargs)
    return context_embedding, answers_embedding


@gin.configurable
class OptimizedWildRelNet(nn.Module):
  """Optimized implementation of the reasoning module in the WildRelNet model.

  Based on https://arxiv.org/pdf/1807.04225.pdf.
  """

  def __init__(self,
               edge_mlp=gin.REQUIRED,
               graph_mlp=gin.REQUIRED,
               dropout_in_last_graph_layer=gin.REQUIRED,
               name="OptimizedWildRelNet",
               **kwargs):
    """Constructs a OptimizedWildRelNet.

    Args:
      edge_mlp: List with number of latent nodes in different layers of the edge
        MLP.
      graph_mlp: List with number of latent nodes in different layers of the
        graph MLP.
      dropout_in_last_graph_layer: Dropout fraction to be applied in the last
        layer of the graph MLP.
      name: String with the name of the model.
      **kwargs: Other keyword arguments passed to tf.keras.Model.
    """
    super(OptimizedWildRelNet, self).__init__(name=name, **kwargs)
    self.name=name

    # Create the EdgeMLP.
    edge_layers = []
    for num_units in edge_mlp:
      edge_layers += [
        nn.Linear(num_units, activation=get_activation())
      ]
    self.edge_layer = nn.Sequential(*edge_layers)

    # Create the GraphMLP.
    graph_layers = []
    for num_units in graph_mlp:
      graph_layers += [edge_layers += [
        nn.Linear(num_units, activation=get_activation())
      ]
      
    if dropout_in_last_graph_layer:
      graph_layers += [
        nn.Dropout(1. - dropout_in_last_graph_layer)
      ]
      
    graph_layers += [
       nn.Linear(1, kernel_initalizer=get_kernel_initializer())
    ]

    # Create the auxiliary layers.
    self.graph_layer = nn.Sequential(*graph_layers)
    self.stacking_layer = relational_layers.StackAnswers()

    # Create the WildRelationNet.
    self.wildrelnet = nn.Sequential(
            relational_layers.AddPositionalEncoding(),
            relational_layers.RelationalLayer(self.edge_layer, lambda x: torch.sum(x, dim=-2)),
            lambda x: torch.sum(x, dim=-2),
            self.graph_layer,
            lambda x: torch.sum(x, dim=-1)
        )


  def call(self, inputs, **kwargs):
    context_embeddings, answer_embeddings = inputs
    # The stacking layer `stacks` each answer panel embedding onto the context
    # panels separately.
    stacked_answers = self.stacking_layer(
        [context_embeddings, answer_embeddings])
    # Apply the relational neural network.
    return self.wildrelnet(stacked_answers, **kwargs)


@gin.configurable("activation")
def get_activation(activation=nn.ReLU(inplace=True)):
    if activation == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    return activation

@gin.configurable("kernel_initializer")
def get_kernel_initializer(kernel_initializer="lecun_normal"):
    if kernel_initializer == "lecun_normal":
        return init.normal_
