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


from typing import Union, Callable, Optional

import brainunit as bu
import brainstate as bst

from .._base import Ion, Channel, check_hierarchies

__all__ = [
  'Potassium',
  'PotassiumFixed',
]


class Potassium(Ion):
  """Base class for modeling Potassium ion."""
  pass


class PotassiumFixed(Potassium):
  """Fixed Sodium dynamics.

  This calcium model has no dynamics. It holds fixed reversal
  potential :math:`E` and concentration :math:`C`.
  """

  def __init__(
      self,
      size: bst.typing.Size,
      E: Union[bst.typing.ArrayLike, Callable] = -95. * bu.mV,
      C: Union[bst.typing.ArrayLike, Callable] = 0.0400811 * bu.mM,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      **channels
  ):
    super().__init__(
      size,
      name=name,
      mode=mode,
      **channels
    )
    self.E = bst.init.param(E, self.varshape)
    self.C = bst.init.param(C, self.varshape)

  def reset_state(self, V, batch_size=None):
    nodes = self.nodes(level=1, include_self=False).subset(Channel).values()
    check_hierarchies(type(self), *tuple(nodes))
    ion_info = self.pack_info()
    for node in nodes:
      node.reset_state(V, ion_info, batch_size)
