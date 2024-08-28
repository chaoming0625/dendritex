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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import dendritex as dx
from dendritex import IonInfo


class INa(dx.channels.SodiumChannel):
  def __init__(self, size, g_max):
    super().__init__(size)

    self.g_max = bst.init.param(g_max, self.varshape)

  def init_state(self, V, Na: IonInfo, batch_size: int = None):
    self.m = dx.State4Integral(self.m_inf(V))
    self.h = dx.State4Integral(self.h_inf(V))

  def compute_derivative(self, V, Na: IonInfo):
    self.m.derivative = (self.m_alpha(V) * (1 - self.m.value) - self.m_beta(V) * self.m.value) / u.ms
    self.h.derivative = (self.h_alpha(V) * (1 - self.h.value) - self.h_beta(V) * self.h.value) / u.ms

  def current(self, V, Na: IonInfo):
    return self.g_max * self.m.value ** 3 * self.h.value * (Na.E - V)

  # m channel
  m_alpha = lambda self, V: 1. / u.math.exprel(-(V / u.mV + 40.) / 10.)  # nan
  m_beta = lambda self, V: 4. * u.math.exp(-(V / u.mV + 65.) / 18.)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

  # h channel
  h_alpha = lambda self, V: 0.07 * u.math.exp(-(V / u.mV + 65.) / 20.)
  h_beta = lambda self, V: 1. / (1. + u.math.exp(-(V / u.mV + 35.) / 10.))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))


class IK(dx.channels.PotassiumChannel):
  def __init__(self, size, g_max):
    super().__init__(size)
    self.g_max = bst.init.param(g_max, self.varshape)

  def init_state(self, V, K: IonInfo, batch_size: int = None):
    self.n = dx.State4Integral(self.n_inf(V))

  def compute_derivative(self, V, K: IonInfo):
    self.n.derivative = (self.n_alpha(V) * (1 - self.n.value) - self.n_beta(V) * self.n.value) / u.ms

  def current(self, V, K: IonInfo):
    return self.g_max * self.n.value ** 4 * (K.E - V)

  n_alpha = lambda self, V: 0.1 / u.math.exprel(-(V / u.mV + 55.) / 10.)
  n_beta = lambda self, V: 0.125 * u.math.exp(-(V / u.mV + 65.) / 80.)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))


class ThreeCompartmentHH(dx.neurons.MultiCompartment):
  def __init__(self, n_neuron: int):
    super().__init__(
      size=(n_neuron, 3),
      connection=((0, 1), (1, 2)),
      Ra=100. * u.ohm * u.cm,
      cm=1.0 * u.uF / u.cm ** 2,
      diam=(12.6157, 1., 1.) * u.um,
      L=(12.6157, 200., 400.) * u.um,
      V_th=20. * u.mV,
      V_initializer=bst.init.Uniform(-70 * u.mV, -60. * u.mV),
      spk_fun=bst.surrogate.ReluGrad(),
    )

    self.IL = dx.channels.IL(self.size,
                             E=(-54.3, -65., -65.) * u.mV,
                             g_max=(0.0003, 0.001, 0.001) * (u.siemens / u.cm ** 2))

    self.na = dx.ions.SodiumFixed(self.size, E=50. * u.mV)
    self.na.add_elem(INa(self.size, g_max=(0.12, 0., 0.) * (u.siemens / u.cm ** 2)))

    self.k = dx.ions.PotassiumFixed(self.size, E=-77. * u.mV)
    self.k.add_elem(IK(self.size, g_max=(0.036, 0., 0.) * (u.siemens / u.cm ** 2)))

  def step_run(self, t, inp):
    dx.rk4_step(self, t, inp)
    return self.V.value

  def step(self, t, *args):
    currents = np.asarray([0.2, 0., 0.]) * u.nA
    self.compute_derivative(currents)

  def save(self, t, *args):
    return self.V.value


def simulate():
  hh = ThreeCompartmentHH(n_neuron=1)
  hh.init_state()

  ts, ys, steps = dx.diffrax_solve(
    hh.step, 'tsit5', 0. * u.ms, 100. * u.ms, 0.01 * u.ms,
    savefn=hh.save, atol=1e-5, max_steps=1000,
  )
  print(steps)
  plt.plot(ts.to_decimal(u.ms), u.math.squeeze(ys).to_decimal(u.mV))
  plt.xlabel('Time [ms]')
  plt.ylabel('Potential [mV]')
  plt.show()

  # ts, ys, steps = dx.diffrax_solve_adjoint(
  #   hh.step, 'kvaerno4', 0. * u.ms, 100. * u.ms, 0.1 * u.ms,
  #   saveat=u.math.arange(0., 100., 0.1) * u.ms, atol=1e-5,
  # )
  # print(steps)
  # plt.plot(ts.to_decimal(u.ms), u.math.squeeze(ys[0]).to_decimal(u.mV))
  # plt.xlabel('Time [ms]')
  # plt.ylabel('Potential [mV]')
  # plt.show()


if __name__ == '__main__':
  simulate()
