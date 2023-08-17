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

"""Library of commonly used architectures and reconstruction losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import tensorflow.compat.v1 as tf
import gin.torch
import torch

@gin.configurable("encoder", whitelist=["num_latent", "encoder_fn"])
def make_gaussian_encoder(input_tensor,
                          is_training=True,
                          num_latent=gin.REQUIRED,
                          encoder_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a Gaussian encoder configurable with gin.

  This is a separate function so that several different models (such as
  BetaVAE and FactorVAE) can call this function while the gin binding always
  stays 'encoder.(...)'. This makes it easier to configure models and parse
  the results files.

  Args:
    input_tensor: Tensor with image that should be encoded.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    num_latent: Integer with dimensionality of latent space.
    encoder_fn: Function that that takes the arguments (input_tensor,
      num_latent, is_training) and returns the tuple (means, log_vars) with the
      encoder means and log variances.

  Returns:
    Tuple (means, log_vars) with the encoder means and log variances.
  """
  with torch.nn.Module("encoder"):
    return encoder_fn(
        input_tensor=input_tensor,
        num_latent=num_latent,
        is_training=is_training)


@gin.configurable("decoder", whitelist=["decoder_fn"])
def make_decoder(latent_tensor,
                 output_shape,
                 is_training=True,
                 decoder_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a decoder configurable with gin.

  This is a separate function so that several different models (such as
  BetaVAE and FactorVAE) can call this function while the gin binding always
  stays 'decoder.(...)'. This makes it easier to configure models and parse
  the results files.

  Args:
    latent_tensor: Tensor latent space embeddings to decode from.
    output_shape: Tuple with the output shape of the observations to be
      generated.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    decoder_fn: Function that that takes the arguments (input_tensor,
      output_shape, is_training) and returns the decoded observations.

  Returns:
    Tensor of decoded observations.
  """
  with torch.nn.Module("decoder"):
    return decoder_fn(
        latent_tensor=latent_tensor,
        output_shape=output_shape,
        is_training=is_training)


@gin.configurable("discriminator", whitelist=["discriminator_fn"])
def make_discriminator(input_tensor,
                       is_training=False,
                       discriminator_fn=gin.REQUIRED):
  """Gin wrapper to create and apply a discriminator configurable with gin.

  This is a separate function so that several different models (such as
  FactorVAE) can potentially call this function while the gin binding always
  stays 'discriminator.(...)'. This makes it easier to configure models and
  parse the results files.

  Args:
    input_tensor: Tensor on which the discriminator operates.
    is_training: Boolean that indicates whether we are training (usually
      required for batch normalization).
    discriminator_fn: Function that that takes the arguments
    (input_tensor, is_training) and returns tuple of (logits, clipped_probs).

  Returns:
    Tuple of (logits, clipped_probs) tensors.
  """
  with torch.nn.Module("discriminator"):
    logits, probs = discriminator_fn(input_tensor, is_training=is_training)
    clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
  return logits, clipped


@gin.configurable("fc_encoder", whitelist=[])
def fc_encoder(input_tensor, num_latent, is_training=True):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl).

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  flattened = input_tensor.view(input_tensor.size(0), -1)
  e1 = torch.nn.functional.relu(torch.nn.Linear(flattened.size(1), 1200)(flattened))
  e2 = torch.nn.functional.relu(torch.nn.Linear(1200, 1200)(e1))
  means = torch.nn.Linear(1200, num_latent)(e2)
  log_var = torch.nn.Linear(1200, num_latent)(e2)
  return means, log_var



@gin.configurable("conv_encoder", whitelist=[])
def conv_encoder(input_tensor, num_latent, is_training=True):
  """Convolutional encoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  e1 = torch.nn.functional.relu(torch.nn.Conv2d(
    in_channels=input_tensor.size(1),
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding="same"
  )(input_tensor))

  e2 = torch.nn.functional.relu(torch.nn.Conv2d(
    in_channels=e1.size(1),
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding="same"
  )(e1))

  e3 = torch.nn.functional.relu(torch.nn.Conv2d(
    in_channels=e2.size(1),
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding="same"
  )(e2))

  e4 = torch.nn.functional.relu(torch.nn.Conv2d(
    in_channels=e3.size(1),
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding="same"
  )(e3))

  flat_e4 = e4.view(e4.size(0), -1)
  e5 = torch.nn.functional.relu(torch.nn.Linear(flat_e4.size(1), 256)(flat_e4))
  means = torch.nn.Linear(e5.size(1), num_latent)(e5)
  log_var = torch.nn.Linear(e5.size(1), num_latent)(e5)
  return means, log_var


@gin.configurable("fc_decoder", whitelist=[])
def fc_decoder(latent_tensor, output_shape, is_training=True):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor to connect decoder to.
    output_shape: Shape of the data.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
    intensities.
  """
  del is_training
  d1 = torch.nn.functional.tanh(torch.nn.Linear(latent_tensor.size(1), 1200)(latent_tensor))
  d2 = torch.nn.functional.tanh(torch.nn.Linear(d1.size(1), 1200)(d1))
  d3 = torch.nn.functional.tanh(torch.nn.Linear(d2.size(1), 1200)(d2))
  d4 = torch.nn.Linear(d3.size(1), np.prod(output_shape))(d3)
  output = d4.view(-1, *output_shape)


@gin.configurable("deconv_decoder", whitelist=[])
def deconv_decoder(latent_tensor, output_shape, is_training=True):
  """Convolutional decoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
    output_shape: Shape of the data.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  del is_training
  d1 = torch.nn.functional.relu(torch.nn.Linear(latent_tensor.size(1), 256)(latent_tensor))
  d2 = torch.nn.functional.relu(torch.nn.Linear(d1.size(1), 1024)(d1))
  d2_reshaped = d2.view(-1, 64, 4, 4)
  d3 = torch.nn.functional.relu(torch.nn.ConvTranspose2d(d2_reshaped.size(1), 64, kernel_size=4, stride=2, padding=1)(d2_reshaped))
  d4 = torch.nn.functional.relu(torch.nn.ConvTranspose2d(d3.size(1), 32, kernel_size=4, stride=2, padding=1)(d3))
  d5 = torch.nn.functional.relu(torch.nn.ConvTranspose2d(d4.size(1), 32, kernel_size=4, stride=2, padding=1)(d4))
  d6 = torch.nn.ConvTranspose2d(d5.size(1), output_shape[2], kernel_size=4, stride=2, padding=1)(d5)
  output = d6.view(-1, *output_shape)
  return output



@gin.configurable("fc_discriminator", whitelist=[])
def fc_discriminator(input_tensor, is_training=True):
  """Fully connected discriminator used in FactorVAE paper for all datasets.

  Based on Appendix A page 11 "Disentangling by Factorizing"
  (https://arxiv.org/pdf/1802.05983.pdf)

  Args:
    input_tensor: Input tensor of shape (None, num_latents) to build
      discriminator on.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    logits: Output tensor of shape (batch_size, 2) with logits from
      discriminator.
    probs: Output tensor of shape (batch_size, 2) with probabilities from
      discriminator.
  """
  del is_training
  flattened = input_tensor.view(input_tensor.size(0), -1)
  d1 = torch.nn.functional.leaky_relu(torch.nn.Linear(flattened.size(1), 1000)(flattened))
  d2 = torch.nn.functional.leaky_relu(torch.nn.Linear(d1.size(1), 1000)(d1))
  d3 = torch.nn.functional.leaky_relu(torch.nn.Linear(d2.size(1), 1000)(d2))
  d4 = torch.nn.functional.leaky_relu(torch.nn.Linear(d3.size(1), 1000)(d3))
  d5 = torch.nn.functional.leaky_relu(torch.nn.Linear(d4.size(1), 1000)(d4))
  d6 = torch.nn.functional.leaky_relu(torch.nn.Linear(d5.size(1), 1000)(d5))
  logits = torch.nn.Linear(d6.size(1), 2)(d6)
  probs = torch.nn.functional.softmax(logits, dim=1)
  return logits, probs



@gin.configurable("test_encoder", whitelist=["num_latent"])
def test_encoder(input_tensor, num_latent, is_training):
  """Simple encoder for testing.

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training
  flattened = input_tensor.view(input_tensor.size(0), -1)
  means = torch.nn.Linear(flattened.size(1), num_latent)(flattened)
  log_var = torch.nn.Linear(flattened.size(1), num_latent)(flattened)
  return means, log_var



@gin.configurable("test_decoder", whitelist=[])
def test_decoder(latent_tensor, output_shape, is_training=False):
  """Simple decoder for testing.

  Args:
    latent_tensor: Input tensor to connect decoder to.
    output_shape: Output shape.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  del is_training
  output = torch.nn.Linear(latent_tensor.size(1), np.prod(output_shape))(latent_tensor)
  output = output.view(-1, *output_shape)
  return output
