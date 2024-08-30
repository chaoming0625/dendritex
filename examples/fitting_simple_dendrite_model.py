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

from typing import Callable, Sequence

import brainstate as bst
import braintools as bts
import brainunit as bu
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import dendritex as dx
from dendritex import IonInfo

bst.environ.set(dt=0.01 * bu.ms)
s = bu.siemens / bu.cm ** 2


class INa(dx.channels.SodiumChannel):
  def __init__(self, size, g_max):
    super().__init__(size)

    self.g_max = bst.init.param(g_max, self.varshape)

  def init_state(self, V, Na: IonInfo, batch_size: int = None):
    self.m = dx.State4Integral(self.m_inf(V))
    self.h = dx.State4Integral(self.h_inf(V))

  def compute_derivative(self, V, Na: IonInfo):
    self.m.derivative = (self.m_alpha(V) * (1 - self.m.value) - self.m_beta(V) * self.m.value) / bu.ms
    self.h.derivative = (self.h_alpha(V) * (1 - self.h.value) - self.h_beta(V) * self.h.value) / bu.ms

  def current(self, V, Na: IonInfo):
    return self.g_max * self.m.value ** 3 * self.h.value * (Na.E - V)

  # m channel
  m_alpha = lambda self, V: 1. / bu.math.exprel(-(V / bu.mV + 40.) / 10.)  # nan
  m_beta = lambda self, V: 4. * bu.math.exp(-(V / bu.mV + 65.) / 18.)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

  # h channel
  h_alpha = lambda self, V: 0.07 * bu.math.exp(-(V / bu.mV + 65.) / 20.)
  h_beta = lambda self, V: 1. / (1. + bu.math.exp(-(V / bu.mV + 35.) / 10.))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))


class IK(dx.channels.PotassiumChannel):
  def __init__(self, size, g_max):
    super().__init__(size)
    self.g_max = bst.init.param(g_max, self.varshape)

  def init_state(self, V, K: IonInfo, batch_size: int = None):
    self.n = dx.State4Integral(self.n_inf(V))

  def compute_derivative(self, V, K: IonInfo):
    self.n.derivative = (self.n_alpha(V) * (1 - self.n.value) - self.n_beta(V) * self.n.value) / bu.ms

  def current(self, V, K: IonInfo):
    return self.g_max * self.n.value ** 4 * (K.E - V)

  n_alpha = lambda self, V: 0.1 / bu.math.exprel(-(V / bu.mV + 55.) / 10.)
  n_beta = lambda self, V: 0.125 * bu.math.exp(-(V / bu.mV + 65.) / 80.)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))


class ThreeCompartmentHH(dx.neurons.MultiCompartment):
  def __init__(self, n_neuron: int, g_na, g_k, g_l):
    super().__init__(
      size=(n_neuron, 3),
      connection=((0, 1), (1, 2)),
      Ra=100. * bu.ohm * bu.cm,
      cm=1.0 * bu.uF / bu.cm ** 2,
      diam=(12.6157, 1., 1.) * bu.um,
      L=(12.6157, 200., 400.) * bu.um,
      V_th=20. * bu.mV,
      V_initializer=bst.init.Constant(-65 * bu.mV),
      spk_fun=bst.surrogate.ReluGrad(),
    )

    self.IL = dx.channels.IL(self.size, E=(-54.3, -65., -65.) * bu.mV, g_max=g_l * s)

    self.na = dx.ions.SodiumFixed(self.size, E=50. * bu.mV)
    self.na.add_elem(INa(self.size, g_max=(g_na, 0., 0.) * s))

    self.k = dx.ions.PotassiumFixed(self.size, E=-77. * bu.mV)
    self.k.add_elem(IK(self.size, g_max=(g_k, 0., 0.) * s))

  def step_run(self, t, inp):
    dx.rk4_step(self, t, inp)
    return self.V.value, self.spike.value


def visualize_a_simulate(currents, params, show=True):
  times = np.arange(0, currents.shape[0]) * bst.environ.get_dt()
  vs, spks = simulate(currents, params)

  fig, gs = bts.visualize.get_figure(1, 1, 3.0, 4.0)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(times / bu.ms, bu.math.squeeze(vs / bu.mV))
  plt.xlabel('Time [ms]')
  plt.ylabel('Potential [mV]')
  if show:
    plt.show()


@jax.jit
def simulate(currents, params):
  hh = ThreeCompartmentHH(n_neuron=1, g_na=params[0], g_k=params[1], g_l=params[2:])
  hh.init_state()

  times = np.arange(0, currents.shape[0]) * bst.environ.get_dt()
  # vs, spks = bst.transform.for_loop(hh.step_run, times, currents)
  vs, spks = bst.transform.checkpointed_for_loop(hh.step_run, times, currents)
  return vs, spks


def compare_potentials(param, currents, target_potentials, n_point=10):
  vs = simulate(currents, param)[0]  # (T, B)
  indices = np.arange(0, vs.shape[0], vs.shape[0] // n_point)
  losses = bts.metric.squared_error(vs[indices] / bu.mV, target_potentials[indices] / bu.mV)
  return losses.mean()


class ScipyOptimizer:
  def __init__(
      self,
      fun: Callable,
      bounds: np.ndarray | Sequence,
      method: str = 'L-BFGS-B',
  ):
    self.loss_fun = jax.jit(fun)
    self.method = method
    self.bounds = bounds
    assert len(bounds) == 2, "Bounds must be a tuple of two elements: (min, max)"

    # Wrap the gradient in a similar manner
    self.jac = jax.jit(jax.grad(fun))

  def minimize(self, num_sample=1):
    bounds = np.asarray(self.bounds).T
    xs = np.random.uniform(self.bounds[0], self.bounds[1], size=(num_sample,) + self.bounds[0].shape)
    best_l = np.inf
    best_r = None

    for x0 in xs:
      results = minimize(
        self.loss_fun,
        x0,
        method=self.method,
        jac=self.jac,
        bounds=bounds,
      )
      if results.fun < best_l:
        best_l = results.fun
        best_r = results
    return best_r


def fitting_example(method='L-BFGS-B', n_sample=1):
  print(f"Method: {method}, n_sample: {n_sample}")

  # 1. generating the target data
  bst.environ.set(dt=0.01 * bu.ms)
  n_seq, n_batch = 10000, 5
  inp_traces = np.random.uniform(0., 1., (n_batch, n_seq, 3)) * bu.nA
  inp_traces[..., 1:] = 0. * bu.nA
  target_params = np.asarray([0.12, 0.036, 0.0003, 0.001, 0.001])
  target_vs, target_spks = jax.vmap(simulate, in_axes=(0, None))(inp_traces, target_params)

  # 2. set the parameter bound
  # inp_traces: [B, T]
  bounds = [
    np.asarray([0.05, 0.01, 0.000, 0.00, 0.00]),
    np.asarray([0.2, 0.1, 0.001, 0.01, 0.01])
  ]
  print('Lower bound:', bounds[0])
  print('Upper bound:', bounds[1])

  @jax.jit
  def jit_potential(params):
    return jax.vmap(compare_potentials, in_axes=(None, 0, 0))(params, inp_traces, target_vs).mean()

  # 3. optimization
  opt = ScipyOptimizer(jit_potential, bounds=bounds, method=method)
  param = opt.minimize(num_sample=n_sample)

  # 4. verification
  loss = jit_potential(param.x)
  print('Param = ', param.x)
  print('Loss = ', loss)
  visualize_a_simulate(inp_traces[0], param.x, show=False)
  visualize_a_simulate(inp_traces[0], target_params)
  return param, loss


if __name__ == '__main__':
  pass
  # visualize_a_simulate(np.random.rand(1000, 3) * u.nA, np.asarray([0.12, 0.036, 0.0003, 0.001, 0.001]))
  fitting_example()

