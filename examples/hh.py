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

import brainstate as bst
import brainunit as bu

import dendritex as den


class HH(den.SingleCompartmentNeuron):
  def __init__(self, size):
    super().__init__(size)

    self.na = den.SodiumFixed(size, E=-50. * bu.mV)
    self.na.add_elem(den.INa_HH1952(size))

    self.k = den.PotassiumFixed(size, E=-77. * bu.mV)
    self.k.add_elem(den.IK_HH1952(size))

    self.IL = den.IL(size, E=-54.387 * bu.mV, g_max=0.03 * bu.mS / bu.cm ** 2)


hh = HH([1, 1])
hh.init_state()


def step_fun(t):
  with bst.environ.context(dt=0.1 * bu.ms):
    den.rk4_step(hh, t, 0.001 * bu.nA)
  return hh.V.value


vs = bst.transform.for_loop(step_fun, bu.math.arange(1000) * bu.ms)
print(vs / bu.mV)
