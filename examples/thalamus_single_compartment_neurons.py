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

"""
Implementation of the following models in the paper:

- Li, Guoshi, Craig S. Henriquez, and Flavio Fröhlich. “Unified thalamic model generates
  multiple distinct oscillations with state-dependent entrainment by stimulation.”
  PLoS computational biology 13.10 (2017): e1005797.
"""

import brainstate as bst
import braintools as bts
import brainunit as bu
import matplotlib.pyplot as plt

import dendritex as dx

S = bu.mS / bu.cm ** 2


class SingleCompartmentThalamusNeuron(dx.neurons.SingleCompartment):
  def step_run(self, t, inp):
    dx.rk4_step(self, t, inp)
    return self.V.value


class HTC(SingleCompartmentThalamusNeuron):
  def __init__(self, size, gKL=0.01 * S, V_initializer=bst.init.Constant(-65. * bu.mV)):
    super().__init__(size, A=2.9e-4 * bu.cm ** 2, V_initializer=V_initializer, V_th=20. * bu.mV)

    self.na = dx.ions.SodiumFixed(size, E=50. * bu.mV)
    self.na.add_elem(INa=dx.channels.INa_Ba2002(size, V_sh=-30 * bu.mV))

    self.k = dx.ions.PotassiumFixed(size, E=-90. * bu.mV)
    self.k.add_elem(IKL=dx.channels.IK_Leak(size, g_max=gKL))
    self.k.add_elem(IDR=dx.channels.IKDR_Ba2002(size, V_sh=-30. * bu.mV, phi=0.25))

    self.ca = dx.ions.CalciumDetailed(size, C_rest=5e-5 * bu.mM, tau=10. * bu.ms, d=0.5 * bu.um)
    self.ca.add_elem(ICaL=dx.channels.ICaL_IS2008(size, g_max=0.5 * S))
    self.ca.add_elem(ICaN=dx.channels.ICaN_IS2008(size, g_max=0.5 * S))
    self.ca.add_elem(ICaT=dx.channels.ICaT_HM1992(size, g_max=2.1 * S))
    self.ca.add_elem(ICaHT=dx.channels.ICaHT_HM1992(size, g_max=3.0 * S))

    self.kca = dx.MixIons(self.k, self.ca)
    self.kca.add_elem(IAHP=dx.channels.IAHP_De1994(size, g_max=0.3 * S))

    self.Ih = dx.channels.Ih_HM1992(size, g_max=0.01 * S, E=-43 * bu.mV)
    self.IL = dx.channels.IL(size, g_max=bst.init.Uniform(0.0075 * S, 0.0125 * S), E=-70 * bu.mV)


class RTC(SingleCompartmentThalamusNeuron):
  def __init__(self, size, gKL=0.01 * S, V_initializer=bst.init.Constant(-65. * bu.mV)):
    super().__init__(size, A=2.9e-4 * bu.cm ** 2, V_initializer=V_initializer, V_th=20 * bu.mV)

    self.na = dx.ions.SodiumFixed(size)
    self.na.add_elem(INa=dx.channels.INa_Ba2002(size, V_sh=-40 * bu.mV))

    self.k = dx.ions.PotassiumFixed(size, E=-90. * bu.mV)
    self.k.add_elem(IDR=dx.channels.IKDR_Ba2002(size, V_sh=-40 * bu.mV, phi=0.25))
    self.k.add_elem(IKL=dx.channels.IK_Leak(size, g_max=gKL))

    self.ca = dx.ions.CalciumDetailed(size, C_rest=5e-5 * bu.mM, tau=10. * bu.ms, d=0.5 * bu.um)
    self.ca.add_elem(ICaL=dx.channels.ICaL_IS2008(size, g_max=0.3 * S))
    self.ca.add_elem(ICaN=dx.channels.ICaN_IS2008(size, g_max=0.6 * S))
    self.ca.add_elem(ICaT=dx.channels.ICaT_HM1992(size, g_max=2.1 * S))
    self.ca.add_elem(ICaHT=dx.channels.ICaHT_HM1992(size, g_max=0.6 * S))

    self.kca = dx.MixIons(self.k, self.ca)
    self.kca.add_elem(IAHP=dx.channels.IAHP_De1994(size, g_max=0.1 * S))

    self.Ih = dx.channels.Ih_HM1992(size, g_max=0.01 * S, E=-43 * bu.mV)
    self.IL = dx.channels.IL(size, g_max=bst.init.Uniform(0.0075 * S, 0.0125 * S), E=-70 * bu.mV)


class IN(SingleCompartmentThalamusNeuron):
  def __init__(self, size, V_initializer=bst.init.Constant(-70. * bu.mV)):
    super(IN, self).__init__(size, A=1.7e-4 * bu.cm ** 2, V_initializer=V_initializer, V_th=20. * bu.mV)

    self.na = dx.ions.SodiumFixed(size)
    self.na.add_elem(INa=dx.channels.INa_Ba2002(size, V_sh=-30 * bu.mV))

    self.k = dx.ions.PotassiumFixed(size, E=-90. * bu.mV)
    self.k.add_elem(IDR=dx.channels.IKDR_Ba2002(size, V_sh=-30 * bu.mV, phi=0.25))
    self.k.add_elem(IKL=dx.channels.IK_Leak(size, g_max=0.01 * S))

    self.ca = dx.ions.CalciumDetailed(size, C_rest=5e-5 * bu.mM, tau=10. * bu.ms, d=0.5 * bu.um)
    self.ca.add_elem(ICaN=dx.channels.ICaN_IS2008(size, g_max=0.1 * S))
    self.ca.add_elem(ICaHT=dx.channels.ICaHT_HM1992(size, g_max=2.5 * S))

    self.kca = dx.MixIons(self.k, self.ca)
    self.kca.add_elem(IAHP=dx.channels.IAHP_De1994(size, g_max=0.2 * S))

    self.IL = dx.channels.IL(size, g_max=bst.init.Uniform(0.0075 * S, 0.0125 * S), E=-60 * bu.mV)
    self.Ih = dx.channels.Ih_HM1992(size, g_max=0.05 * S, E=-43 * bu.mV)


class TRN(SingleCompartmentThalamusNeuron):
  def __init__(self, size, V_initializer=bst.init.Constant(-70. * bu.mV)):
    super(TRN, self).__init__(size, A=1.43e-4 * bu.cm ** 2, V_initializer=V_initializer, V_th=20. * bu.mV)

    self.na = dx.ions.SodiumFixed(size)
    self.na.add_elem(INa=dx.channels.INa_Ba2002(size, V_sh=-40 * bu.mV))

    self.k = dx.ions.PotassiumFixed(size, E=-90. * bu.mV)
    self.k.add_elem(IDR=dx.channels.IKDR_Ba2002(size, V_sh=-40 * bu.mV))
    self.k.add_elem(IKL=dx.channels.IK_Leak(size, g_max=0.01 * S))

    self.ca = dx.ions.CalciumDetailed(size, C_rest=5e-5 * bu.mM, tau=100. * bu.ms, d=0.5 * bu.um)
    self.ca.add_elem(ICaN=dx.channels.ICaN_IS2008(size, g_max=0.2 * S))
    self.ca.add_elem(ICaT=dx.channels.ICaT_HP1992(size, g_max=1.3 * S))

    self.kca = dx.MixIons(self.k, self.ca)
    self.kca.add_elem(IAHP=dx.channels.IAHP_De1994(size, g_max=0.2 * S))

    self.IL = dx.channels.IL(size, g_max=bst.init.Uniform(0.0075 * S, 0.0125 * S), E=-60 * bu.mV)


def try_trn_neuron():
  bst.environ.set(dt=0.01 * bu.ms)

  # trn = RTC([1, 1])  # [n_neuron, n_compartment]
  trn = RTC(1)  # [n_neuron, n_compartment]
  # trn = TRN([1, 1])  # [n_neuron, n_compartment]
  trn.init_state()

  I = bts.input.section_input(values=[0, -0.05, 0], durations=[100, 100, 500], dt=0.01) * bu.nA
  times = bu.math.arange(I.shape[0]) * bst.environ.get_dt()

  vs = bst.transform.for_loop(trn.step_run, times, I)

  plt.plot(times / bu.ms, bu.math.squeeze(vs / bu.mV))
  plt.show()


if __name__ == '__main__':
  try_trn_neuron()
