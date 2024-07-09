# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

from typing import Union, Optional, Callable

import brainstate as bst
import brainunit as bu

from .._base import HHTypedNeuron, IonChannel, State4Integral

__all__ = [
  'SingleCompartment',
]


class SingleCompartment(HHTypedNeuron):
  r"""
  Base class to model conductance-based neuron group.

  The standard formulation for a conductance-based model is given as

  .. math::

      C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

  where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
  reversal potential, :math:`M` is the activation variable, and :math:`N` is the
  inactivation variable.

  :math:`M` and :math:`N` have the dynamics of

  .. math::

      {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

  where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
  :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
  Equivalently, the above equation can be written as:

  .. math::

      \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

  where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.


  Parameters
  ----------
  size : int, sequence of int
    The network size of this neuron group.
  name : optional, str
    The neuron group name.
  """
  __module__ = 'dendritex.neurons'

  def __init__(
      self,
      size: bst.typing.Size,
      C: Union[bst.typing.ArrayLike, Callable] = 1. * bu.uF / bu.cm ** 2,
      A: Union[bst.typing.ArrayLike, Callable] = 1e-3 * bu.cm ** 2,
      V_th: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      V_initializer: Union[bst.typing.ArrayLike, Callable] = bst.init.Uniform(-70 * bu.mV, -60. * bu.mV),
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      **ion_channels
  ):
    super().__init__(size, mode=mode, name=name, **ion_channels)

    # parameters for neurons
    assert self.n_compartment == 1, (f'Point-based neuron only supports single compartment. '
                                     f'But got {self.n_compartment} compartments.')
    self.C = C
    self.A = A
    self.V_th = V_th
    self._V_initializer = V_initializer

  def init_state(self, batch_size=None):
    self.V = State4Integral(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.spike = bst.ShortTermState(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    super().init_state(batch_size)

  def reset_state(self, batch_size=None):
    self.V.value = bst.init.param(self._V_initializer, self.varshape, batch_size)
    self.spike.value = bst.init.param(bu.math.zeros, self.varshape, batch_size)
    super().init_state(batch_size)

  def before_integral(self, *args):
    self._last_V = self.V.value

    channels = self.nodes(level=1, include_self=False).subset(IonChannel)
    for node in channels.values():
      node.before_integral(self.V.value)

  def compute_derivative(self, x=0.):
    # [ Compute the derivative of membrane potential ]
    # 1. inputs
    x = x * (1e-3 / self.A)
    # 2. synapses
    x = self.sum_current_inputs(self.V.value, init=x)
    # 3. channels
    for ch in self.nodes(level=1, include_self=False).subset(IonChannel).values():
      x = x + ch.current(self.V.value)
    # 4. derivatives
    self.V.derivative = x / self.C

    # [ integrate dynamics of ion and ion channels ]
    # check whether the children channels have the correct parents.
    channels = self.nodes(level=1, include_self=False).subset(IonChannel)
    for node in channels.values():
      node.compute_derivative(self.V.value)

  def after_integral(self, *args):
    self.V.value = self.sum_delta_inputs(init=self.V.value)
    spike = bu.math.logical_and(self._last_V >= self.V_th, self.V.value < self.V_th)
    self.spike.value = bu.math.asarray(spike, dtype=self.spike.value.dtype)

    channels = self.nodes(level=1, include_self=False).subset(IonChannel)
    for node in channels.values():
      node.after_integral(self.V.value)
