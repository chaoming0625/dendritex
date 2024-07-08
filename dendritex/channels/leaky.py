# -*- coding: utf-8 -*-

"""
This module implements leakage channels.

"""

from __future__ import annotations

from typing import Union, Callable, Sequence, Optional

import brainstate as bst
import brainunit as bu

from .._base import HHTypedNeuron, Channel

__all__ = [
  'LeakageChannel',
  'IL',
]


class LeakageChannel(Channel):
  """
  Base class for leakage channel dynamics.
  """

  root_type = HHTypedNeuron

  def reset_state(self, V, batch_size=None):
    pass


class IL(LeakageChannel):
  """The leakage channel current.

  Parameters
  ----------
  g_max : float
    The leakage conductance.
  E : float
    The reversal potential.
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 0.1 * (bu.mS / bu.cm ** 2),
      E: Union[bst.typing.ArrayLike, Callable] = -70. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    self.E = bst.init.param(E, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def reset_state(self, V, batch_size=None):
    pass

  def update(self, V):
    pass

  def current(self, V):
    return self.g_max * (self.E - V)
