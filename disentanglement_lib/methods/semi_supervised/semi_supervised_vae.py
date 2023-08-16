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

"""Library of losses for semi-supervised disentanglement learning.

Implementation of semi-supervised VAE based models for unsupervised learning of
disentangled representations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import vae
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

import gin.torch
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimatorSpec


class BaseS2VAE(vae.BaseVAE):
  """Abstract base class of a basic semi-supervised Gaussian encoder model."""

  def __init__(self, factor_sizes):
    self.factor_sizes = factor_sizes

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function.

    Args:
      features: Batch of images [batch_size, 64, 64, 3].
      labels: Tuple with batch of features [batch_size, 64, 64, 3] and the
        labels [batch_size, labels_size].
      mode: Mode for the TPUEstimator.
      params: Dict with parameters.

    Returns:
      TPU estimator.
    """

    is_training = (mode == "train")
    labelled_features = labels[0]
    labels = labels[1].float()
    data_shape = features.get_shape().as_list()[1:]
    with torch.no_grad()::
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = per_sample_loss.mean()
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    gamma_annealed = StepLR(optimizer, step_size=train.get_global_step(), gamma=self.gamma_sup)
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    regularizer = self.unsupervised_regularizer(
        kl_loss, z_mean, z_logvar, z_sampled) + gamma_annealed * supervised_loss
    loss = reconstruction_loss+ regularizer
    elbo = reconstruction_loss+ kl_loss
     if mode == "train":
      optimizer = optimizers.make_vae_optimizer()
      update_ops = []
      train_op = optimizer.minimize(
          loss=loss)
      train_op = torch.optim.optimizer.step([train_op, *update_ops])
      writer = SummaryWriter()  # Initialize a summary writer for logging
      writer.add_scalar("reconstruction_loss", reconstruction_loss.item())

    for step in range(num_steps):
      
      if step % 100 == 0:
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("reconstruction_loss", reconstruction_loss.item(), step)
        writer.add_scalar("supervised_loss", supervised_loss.item(), step)
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == "eval":
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                       "supervised_loss"),
                        [reconstruction_loss, regularizer, supervised_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


def sample_from_latent_distribution(z_mean, z_logvar):
  """Sample from the encoder distribution with reparametrization trick."""
  std = torch.exp(0.5 * z_logvar)
  eps = torch.randn_like(std)
  return z_mean + eps * std


def compute_gaussian_kl(z_mean, z_logvar):
  """Compute KL divergence between input Gaussian and Standard Normal."""
  kl_loss = 0.5 * torch.sum(torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1, dim=1)
  return torch.mean(kl_loss)


def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn


@gin.configurable("annealer", blacklist=["gamma", "step"])
def make_annealer(gamma,
                  step,
                  iteration_threshold=gin.REQUIRED,
                  anneal_fn=gin.REQUIRED):
  """Wrapper that creates annealing function."""
  return anneal_fn(gamma, step, iteration_threshold)


@gin.configurable("fixed", blacklist=["gamma", "step"])
def fixed_annealer(gamma, step, iteration_threshold):
  """No annealing."""
  del step, iteration_threshold
  return gamma


@gin.configurable("anneal", blacklist=["gamma", "step"])
def annealed_annealer(gamma, step, iteration_threshold):
  """Linear annealing."""
  iteration_threshold = 10000  # You can adjust this value as needed
    return torch.min(gamma * 1., gamma * 1. * torch.tensor(step) / iteration_threshold)



@gin.configurable("fine_tune", blacklist=["gamma", "step"])
def fine_tune_annealer(gamma, step, iteration_threshold):
  """Fine tuning.

  This annealer returns zero if step < iteration_threshold and gamma otherwise.

  Args:
    gamma: Weight of supervised loss.
    step: Current step of training.
    iteration_threshold: When to return gamma instead of zero.

  Returns:
    Either gamma or zero.
  """
  return gamma * torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.max(torch.tensor(0.0, dtype=torch.float32), torch.tensor(step - iteration_threshold, dtype=torch.float32)))


@gin.configurable("supervised_loss", blacklist=["representation", "labels"])
def make_supervised_loss(representation, labels,
                         factor_sizes=None, loss_fn=gin.REQUIRED):
  """Wrapper that creates supervised loss."""
  with tf.variable_scope("supervised_loss"):
    loss = loss_fn(representation, labels, factor_sizes)
  return loss


def normalize_labels(labels, factors_num_values):
  """Normalize the labels in [0, 1].

  Args:
    labels: Numpy array of shape (num_labelled_samples, num_factors) of Float32.
    factors_num_values: Numpy array of shape (num_factors,) containing the
      number of distinct values each factor can take.

  Returns:
    labels normalized in [0, 1].
  """
  factors_num_values_reshaped = np.repeat(
      np.expand_dims(np.float32(factors_num_values), axis=0),
      labels.shape[0],
      axis=0)
  return labels / factors_num_values_reshaped


@gin.configurable("l2", blacklist=["representation", "labels"])
def supervised_regularizer_l2(representation, labels,
                              factor_sizes=None,
                              learn_scale=True):
  """Implements a supervised l2 regularizer.

  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. The number of factors of variation must be smaller or equal to the
  number of latent codes. The representation can be scaled with a learned
  scaling to match the labels or the labels are normalized in [0,1] and the
  representation is projected in the same interval using a sigmoid.

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation (unused).
    learn_scale: Boolean indicating whether the scale should be learned or not.

  Returns:
    L2 loss between the representation and the labels.
  """
  number_latents = representation.shape[1].value
  number_factors_of_variations = labels.shape[1].value
  assert number_latents >= number_factors_of_variations, "Not enough latents."
  if learn_scale:
    b = nn.Parameter(torch.Tensor([1.]))
        return 2. * torch.nn.functional.mse_loss(
            representation[:, :number_factors_of_variations] * b - labels, torch.zeros_like(labels))
  else:
    normalized_labels = normalize_labels(labels, factor_sizes)
    sigmoid_input = representation[:, :number_factors_of_variations].unsqueeze(1)
    sigmoid_output = torch.sigmoid(sigmoid_input)
    return 2. * torch.nn.functional.mse_loss(sigmoid_output, normalized_labels)


@gin.configurable("xent", blacklist=["representation", "labels"])
def supervised_regularizer_xent(representation, labels,
                                factor_sizes=None):
  """Implements a supervised cross_entropy regularizer.

  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. If the number of factors of variation is larger than the latent
  code dimension it raise an exception. Labels are in [0, 1].

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation.

  Returns:
    Xent loss between the representation and the labels.
  """
  number_latents = representation.size[1]
  number_factors_of_variations = labels.size[1]
  assert number_latents >= number_factors_of_variations, "Not enough latents."
  logits = representation[:, :number_factors_of_variations]
  normalized_labels = normalize_labels(labels, factor_sizes)
  loss = F.binary_cross_entropy_with_logits(logits, normalized_labels, reduction='sum')
  return loss


@gin.configurable("cov", blacklist=["representation", "labels"])
def supervised_regularizer_cov(representation, labels,
                               factor_sizes=None):
  """Implements a supervised regularizer using a covariance.

  Penalize the deviation from the identity of the covariance between
  representation and factors of varations.
  If the number of latent dimension is greater than the number of factor of
  variations it only uses the first dimensions of the latent code to
  regularize. Labels are in [0, 1].

  Args:
    representation: Representation of labelled samples.
    labels: Labels for the labelled samples.
    factor_sizes: Cardinality of each factor of variation (unused).


  Returns:
    Loss between the representation and the labels.
  """
  del factor_sizes
  number_latents = representation.shape[1].value
  number_factors_of_variations = labels.shape[1].value
  num_diagonals = torch.min(number_latents, number_factors_of_variations)
  expectation_representation = torch.mean(representation, axis=0)
  expectation_labels = torch.mean(labels, axis=0)
  representation_centered = representation - expectation_representation
  labels_centered = labels - expectation_labels
  covariance = torch.mean(
    representation_centered.unsqueeze(2) * labels_centered.unsqueeze(1),
    dim=0)
    num_diagonals = covariance.size(0)
  l2_loss = torch.norm(torch.diag(covariance, 0), p=2)
  return 2. * l2_loss



@gin.configurable("embed", blacklist=["representation", "labels",
                                      "factor_sizes"])
def supervised_regularizer_embed(representation, labels,
                                 factor_sizes, sigma=gin.REQUIRED,
                                 use_order=False):
  """Embed factors in 1d and compute softmax with the representation.

  Assume a factor of variation indexed by j can take k values. We embed each
  value into k real numbers e_1, ..., e_k. Call e_label(r_j) the embedding of an
  observed label for the factor j. Then, for a dimension r_j of the
  representation, the loss is computed as
  exp(-((r_j - e_label(r_j))*sigma)^2)/sum_{i=1}^k exp(-(r_j - e_i)).
  We compute this term for each factor of variation j and each point. Finally,
  we add these terms into a single number.

  Args:
    representation: Computed representation, tensor of shape (batch_size,
      num_latents)
    labels: Observed values for the factors of variation, tensor of shape
      (batch_size, num_factors).
    factor_sizes: Cardinality of each factor of variation.
    sigma: Temperature for the softmax. Set to "learn" if to be learned.
    use_order: Boolean indicating whether to use the ordering information in the
      factors of variations or not.

  Returns:
    Supervised loss based on the softmax between embedded labels and
    representation.
  """
  number_factors_of_variations = labels.shape[1].value
  supervised_representation = representation[:, :number_factors_of_variations]
  loss = []
  for i in range(number_factors_of_variations):
    with torch.no_grad():
      if use_order:
        bias = torch.nn.Parameter(torch.zeros([]))
        slope = torch.nn.Parameter(torch.zeros([]))
        embedding = torch.arange(factor_sizes[i], dtype=torch.float32) * slope + bias
      else:
        embedding = torch.nn.Parameter(torch.zeros(factor_sizes[i]))
      if sigma == "learn":
        sigma_value = torch.nn.Parameter(torch.ones([1]))
      else:
        sigma_value = sigma
    logits = -torch.square(
        (supervised_representation[:, i].unsqueeze(1) - embedding) * sigma_value)

    one_hot_labels = torch.nn.functional.one_hot(
        labels[:, i].to(torch.int64), num_classes=factor_sizes[i])

    loss.append(torch.nn.functional.cross_entropy(logits, one_hot_labels))

  loss = torch.sum(torch.stack(loss))
  return loss

@gin.configurable("s2_vae")
class S2BetaVAE(BaseS2VAE):
  """Semi-supervised BetaVAE model."""

  def __init__(self, factor_sizes, beta=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a semi-supervised beta-VAE model.

    Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl) with additional supervision.

    Args:
      factor_sizes: Size of each factor of variation.
      beta: Hyperparameter for the unsupervised regularizer.
      gamma_sup: Hyperparameter for the supervised regularizer.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    self.beta = beta
    self.gamma_sup = gamma_sup
    super(S2BetaVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    """Standard betaVAE regularizer."""
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss


@gin.configurable("supervised")
class SupervisedVAE(BaseS2VAE):
  """Fully supervised method build on top of VAE to have visualizations."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function.

    Args:
      features: Batch of images [batch_size, 64, 64, 3].
      labels: Tuple with batch of features [batch_size, 64, 64, 3] and the
        labels [batch_size, labels_size].
      mode: Mode for the TPUEstimator.
      params: Dict with parameters.

    Returns:
      TPU Estimator.
    """

    is_training = (mode == "train")
    labelled_features = labels[0]
    labels = labels[1].float()
    data_shape = features.get_shape().as_list()[1:]
    with torch.no_grad():
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
reconstructions = self.decode(
    z_sampled.detach(), data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = torch.mean(per_sample_loss)
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    regularizer = supervised_loss
    loss = torch.add(reconstruction_loss, regularizer, name="loss")
    if mode == "train":
      optimizer = optimizers.make_vae_optimizer()
      update_ops = []
      train_op = optimizer.minimize(
          loss=loss)
      train_op = torch.optim.optimizer.step([train_op, *update_ops])
      writer = SummaryWriter()  # Initialize a summary writer for logging
      writer.add_scalar("reconstruction_loss", reconstruction_loss.item())

    for step in range(num_steps):
      
      if step % 100 == 0:
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("reconstruction_loss", reconstruction_loss.item(), step)
        writer.add_scalar("supervised_loss", supervised_loss.item(), step)
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == "eval":
      return TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                       "supervised_loss"),
                        [reconstruction_loss, regularizer, supervised_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")


def mine(x, z, name_net="estimator_network"):
  """Computes I(X, Z).

  Uses the algorithm in "Mutual Information Neural Estimation"
  (https://arxiv.org/pdf/1801.04062.pdf).

  Args:
    x: Samples from x [batch_size, size_x].
    z: Samples from z [batch_size, size_z].
    name_net: Scope for the variables forming the network.

  Returns:
    Estimate of the mutual information and the update op for the optimizer.
  """
  z_shuffled = vae.shuffle_codes(z)

  concat_x_x = tf.concat([x, x], axis=0)
  concat_z_z_shuffled = tf.stop_gradient(tf.concat([z, z_shuffled], axis=0))

  with torch.no_grad():
    d1_x = nn.Linear(concat_x_x.size(1), 20)(concat_x_x)
    d1_z = nn.Linear(concat_z_z_shuffled.size(1), 20)(concat_z_z_shuffled)
    d1 = torch.nn.functional.elu(d1_x + d1_z)
    d2 = nn.Linear(d1.size(1), 1)(d1)

  batch_size = tf.shape(x)[0]
  pred_x_z = d2[:batch_size]
  pred_x_z_shuffled = d2[batch_size:]
  batch_size = x.size(0)
  pred_x_z = d2[:batch_size]
  pred_x_z_shuffled = d2[batch_size:]
  loss = -(
    torch.mean(pred_x_z) + torch.log(torch.tensor(batch_size, dtype=torch.float32)) -
    torch.logsumexp(pred_x_z_shuffled, dim=0)
)

all_variables = []
for name, param in model.named_parameters():
    if "estimator_network" in name:
        all_variables.append(param)
mine_op = optim.Adam(all_variables, lr=0.01)
mine_op.zero_grad()
loss.backward()
mine_op.step()

return -loss.item(), mine_op


@gin.configurable("s2_mine_vae")
class MineVAE(BaseS2VAE):
  """MineVAE model."""

  def __init__(self, factor_sizes, gamma_sup=gin.REQUIRED, beta=gin.REQUIRED):
    """Creates a semi-supervised MineVAE model.

    Regularize mutual information using mine.

    Args:
      factor_sizes: Size of each factor of variation.
      gamma_sup: Hyperparameter for the supervised regularizer.
      beta: Hyperparameter for the unsupervised regularizer.
    """
    self.gamma_sup = gamma_sup
    self.beta = beta
    super(MineVAE, self).__init__(factor_sizes)

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    labelled_features = labels[0]
    labels = labels[1].float()
    data_shape = features.get_shape().as_list()[1:]
    with torch.no_grad():
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)

    supervised_loss = []
    mine_ops = []

    for l in range(labels.get_shape().as_list()[1]):
      for r in range(z_mean.get_shape().as_list()[1]):
        label_for_mi = labels[:, l].flatten()
        representation_for_mi = z_mean_labelled[:, r].flatten()
        mi_lr, op_lr = mine(representation_for_mi, label_for_mi,
                            "estimator_network_%d_%d" % (l, r))
        if l != r:
          supervised_loss = supervised_loss + [tf.math.square(mi_lr)]
        mine_ops = mine_ops + [op_lr]
    supervised_loss = torch.sum(torch.stack(supervised_loss))
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = torch.mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    standard_vae_loss = torch.add(
        reconstruction_loss, self.beta * kl_loss, name="VAE_loss")
    gamma_annealed = make_annealer(self.gamma_sup, global_step())
    s2_mine_vae_loss = torch.add(
        standard_vae_loss, gamma_annealed * supervised_loss,
        name="s2_factor_VAE_loss")
    if mode == "train":
    optimizer = optimizers.make_vae_optimizer()
    update_ops = []  # In PyTorch, update ops are not explicitly needed
    train_op = optimizer.minimize(loss=loss)
    train_op = torch.optim.optimizer.step([train_op, *update_ops])
    writer = SummaryWriter()  # Initialize a summary writer for logging
    writer.add_scalar("reconstruction_loss", reconstruction_loss.item())

    logging_hook = ...  # PyTorch does not have an equivalent LoggingTensorHook
    return TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook])
elif mode == tf.estimator.ModeKeys.EVAL:
    return TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                     "supervised_loss"),
                      [reconstruction_loss, regularizer, supervised_loss]))
else:
    raise NotImplementedError('Eval mode not supported')



@gin.configurable("s2_factor_vae")
class S2FactorVAE(BaseS2VAE):
  """FactorVAE model."""

  def __init__(self, factor_sizes, gamma=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a semi-supervised FactorVAE model.

    Implementing Eq. 2 of "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983).

    Args:
      factor_sizes: Size of each factor of variation.
      gamma: Hyperparameter for the unsupervised regularizer.
      gamma_sup: Hyperparameter for the supervised regularizer.
    """
    self.gamma = gamma
    self.gamma_sup = gamma_sup
    super(S2FactorVAE, self).__init__(factor_sizes)

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == "train")
    labelled_features = labels[0]
    labels = labels[1].float()
    data_shape = features.get_shape().as_list()[1:]
    with torch.no_grad():
      z_mean, z_logvar = self.gaussian_encoder(
          features, is_training=is_training)
      z_mean_labelled, _ = self.gaussian_encoder(
          labelled_features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    z_shuffle = vae.shuffle_codes(z_sampled)
    with torch.no_grad():
      logits_z, probs_z = architectures.make_discriminator(
          z_sampled, is_training=is_training)
      _, probs_z_shuffle = architectures.make_discriminator(
          z_shuffle, is_training=is_training)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = torch.mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    standard_vae_loss = torch.add(reconstruction_loss, kl_loss, name="VAE_loss")
    # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
    tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
    tc_loss = torch.mean(tc_loss_per_sample, axis=0)
    regularizer = kl_loss + self.gamma * tc_loss
    gamma_annealed = make_annealer(self.gamma_sup,self.global_step())
    supervised_loss = make_supervised_loss(z_mean_labelled, labels,
                                           self.factor_sizes)
    s2_factor_vae_loss = torch.add(
        standard_vae_loss,
        self.gamma * tc_loss + gamma_annealed * supervised_loss,
        name="s2_factor_VAE_loss")
    discr_loss = torch.add(
        0.5 * torch.mean(tf.log(probs_z[:, 0])),
        0.5 * torch.mean(tf.log(probs_z_shuffle[:, 1])),
        name="discriminator_loss")
    if mode == "train":
      optimizer_vae = optimizers.make_vae_optimizer()
      optimizer_discriminator = optimizers.make_discriminator_optimizer()
      all_variables = []
      encoder_vars = [var for var in all_variables if "encoder" in var.name]
      decoder_vars = [var for var in all_variables if "decoder" in var.name]
      discriminator_vars = [var for var in all_variables \
                            if "discriminator" in var.name]
      update_ops = []
      train_op_vae = optimizer_vae.minimize(
          loss=s2_factor_vae_loss,
          global_step=tf.train.get_global_step(),
          var_list=encoder_vars + decoder_vars)
      train_op_discr = optimizer_discriminator.minimize(
          loss=-discr_loss,
          global_step=tf.train.get_global_step(),
          var_list=discriminator_vars)
      train_op = torch.optim.Optimizer([train_op_vae, train_op_discr])
      torch_summary.scalar("reconstruction_loss", reconstruction_loss)

      class LoggingHook:
        def __init__(self):
          self.iteration = 0

        def __call__(self, loss):
          if self.iteration % 50 == 0:
            print(f"Iteration {self.iteration}, loss: {loss:.4f}")
          self.iteration += 1

      logging_hook = LoggingHook()

      if mode == "train":
        return TPUEstimatorSpec(
        mode=mode,
        loss=s2_factor_vae_loss,
        train_op=train_op,
        training_hooks=[logging_hook])
      elif mode == "eval":
        return TPUEstimatorSpec(
        mode=mode,
        loss=s2_factor_vae_loss,
        eval_metrics=(make_metric_fn("reconstruction_loss", "regularizer",
                                     "kl_loss", "supervised_loss"), [
                                         reconstruction_loss, regularizer,
                                         kl_loss, supervised_loss
                                     ]))
      else:
        raise NotImplementedError("Eval mode not supported.")



@gin.configurable("s2_dip_vae")
class S2DIPVAE(BaseS2VAE):
  """Semi-supervised DIPVAE model."""

  def __init__(self,
               factor_sizes,
               lambda_od=gin.REQUIRED,
               lambda_d_factor=gin.REQUIRED,
               gamma_sup=gin.REQUIRED,
               dip_type="i"):
    """Creates a DIP-VAE model.

    Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
    Concepts from Unlabeled Observations"
    (https://openreview.net/pdf?id=H1kG7GZAW).

    Args:
      factor_sizes: Size of each factor of variation.
      lambda_od: Hyperparameter for off diagonal values of covariance matrix.
      lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
        lambda_d = lambda_d_factor*lambda_od.
      gamma_sup: Hyperparameter for the supervised regularizer.
      dip_type: "i" or "ii".
    """
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type
    self.gamma_sup = gamma_sup
    super(S2DIPVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    cov_z_mean = vae.compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":  # Eq 6 page 4
      # mu = z_mean is [batch_size, num_latent]
      # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
      cov_dip_regularizer = vae.regularize_diag_off_diag_dip(
          cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
      cov_enc = torch.diag_embed(torch.exp(z_logvar))
      expectation_cov_enc = torch.mean(cov_enc, axis=0)
      cov_z = expectation_cov_enc + cov_z_mean
      cov_dip_regularizer = vae.regularize_diag_off_diag_dip(
          cov_z, self.lambda_od, lambda_d)
    else:
      raise NotImplementedError("DIP variant not supported.")
    return kl_loss + cov_dip_regularizer


@gin.configurable("s2_beta_tc_vae")
class S2BetaTCVAE(BaseS2VAE):
  """Semi-supervised BetaTCVAE model."""

  def __init__(self, factor_sizes, beta=gin.REQUIRED, gamma_sup=gin.REQUIRED):
    """Creates a beta-TC-VAE model.

    Based on Equation 5 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 5 can be written as ELBO + (1 - beta) * TC.

    Args:
      factor_sizes: Size of each factor of variation.
      beta: Hyperparameter total correlation.
      gamma_sup: Hyperparameter for the supervised regularizer.
    """
    self.beta = beta
    self.gamma_sup = gamma_sup
    super(S2BetaTCVAE, self).__init__(factor_sizes)

  def unsupervised_regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * vae.total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss
