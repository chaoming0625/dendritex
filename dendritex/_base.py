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


# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Dict, Sequence, Callable, NamedTuple, Tuple

import brainstate as bst
import numpy as np
from brainstate.mixin import _JointGenericAlias

__all__ = [
  'DendriticDynamics',
  'State4Integral',
  'HHTypedNeuron',
  'IonChannel',
  'Ion',
  'MixIons',
  'Channel',

  'IonInfo',
]


#
# - DendriticDynamics
#   - HHTypedNeuron
#     - SingleCompartmentNeuron
#   - IonChannel
#     - Ion
#       - Calcium
#       - Potassium
#       - Sodium
#     - MixIons
#     - Channel
#

class State4Integral(bst.ShortTermState):
  """
  A state that integrates the state of the system to the integral of the state.

  Attributes
  ----------
  derivative: The derivative of the state.

  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.derivative = None


class DendriticDynamics(bst.Dynamics):
  """
  Base class for dendritic dynamics.

  Attributes:
    size: The size of the simulation target.
    pop_size: The size of the population, storing the number of neurons in each population.
    n_compartment: The number of compartments in each neuron.
    varshape: The shape of the state variables.
  """

  def __init__(
      self,
      size: bst.typing.Size,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    # size
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise ValueError(f'size must be int, or a tuple/list of int. '
                         f'But we got {type(size)}')
      if not isinstance(size[0], (int, np.integer)):
        raise ValueError('size must be int, or a tuple/list of int.'
                         f'But we got {type(size)}')
      size = tuple(size)
    elif isinstance(size, (int, np.integer)):
      size = (size,)
    else:
      raise ValueError('size must be int, or a tuple/list of int.'
                       f'But we got {type(size)}')
    self.size = size
    assert len(size) >= 2, 'The size of the dendritic dynamics should be at least 2D: (n_neuron, n_compartment).'
    self.pop_size: Tuple[int, ...] = size[:-1]
    self.n_compartment: int = size[-1]

    # -- Attribute for "InputProjMixIn" -- #
    # each instance of "SupportInputProj" should have
    # "_current_inputs" and "_delta_inputs" attributes
    self._current_inputs: Optional[Dict[str, Callable]] = None
    self._delta_inputs: Optional[Dict[str, Callable]] = None

    # initialize
    super().__init__(size, name=name, mode=mode, keep_size=True)

  def current(self, *args, **kwargs):
    raise NotImplementedError('Must be implemented by the subclass.')

  def before_integral(self, *args, **kwargs):
    raise NotImplementedError

  def compute_derivative(self, *args, **kwargs):
    raise NotImplementedError('Must be implemented by the subclass.')

  def after_integral(self, *args, **kwargs):
    raise NotImplementedError

  def init_state(self, *args, **kwargs):
    raise NotImplementedError

  def reset_state(self, *args, **kwargs):
    raise NotImplementedError


class Container(bst.mixin.Mixin):

  @staticmethod
  def _get_elem_name(elem):
    if isinstance(elem, bst.Module):
      return elem.name
    else:
      return bst.util.get_unique_name('ContainerElem')

  @staticmethod
  def _format_elements(child_type: type, *children_as_tuple, **children_as_dict):
    res = dict()

    # add tuple-typed components
    for module in children_as_tuple:
      if isinstance(module, child_type):
        res[Container._get_elem_name(module)] = module
      elif isinstance(module, (list, tuple)):
        for m in module:
          if not isinstance(m, child_type):
            raise TypeError(f'Should be instance of {child_type.__name__}. '
                            f'But we got {type(m)}')
          res[Container._get_elem_name(m)] = m
      elif isinstance(module, dict):
        for k, v in module.items():
          if not isinstance(v, child_type):
            raise TypeError(f'Should be instance of {child_type.__name__}. '
                            f'But we got {type(v)}')
          res[k] = v
      else:
        raise TypeError(f'Cannot parse sub-systems. They should be {child_type.__name__} '
                        f'or a list/tuple/dict of {child_type.__name__}.')
    # add dict-typed components
    for k, v in children_as_dict.items():
      if not isinstance(v, child_type):
        raise TypeError(f'Should be instance of {child_type.__name__}. '
                        f'But we got {type(v)}')
      res[k] = v
    return res

  def add_elem(self, *elems, **elements):
    """
    Add new elements.

    Args:
      elements: children objects.
    """
    raise NotImplementedError('Must be implemented by the subclass.')


class TreeNode(bst.mixin.Mixin):
  root_type: type

  @staticmethod
  def _root_leaf_pair_check(root: type, leaf: 'TreeNode'):
    if hasattr(leaf, 'root_type'):
      root_type = leaf.root_type
    else:
      raise ValueError('Child class should define "root_type" to '
                       'specify the type of the root node. '
                       f'But we did not found it in {leaf}')
    if not issubclass(root, root_type):
      raise TypeError(f'Type does not match. {leaf} requires a root with type '
                      f'of {leaf.root_type}, but the root now is {root}.')

  @staticmethod
  def check_hierarchies(root: type, *leaves, check_fun: Callable = None, **named_leaves):
    if check_fun is None:
      check_fun = TreeNode._root_leaf_pair_check

    for leaf in leaves:
      if isinstance(leaf, bst.Module):
        check_fun(root, leaf)
      elif isinstance(leaf, (list, tuple)):
        TreeNode.check_hierarchies(root, *leaf, check_fun=check_fun)
      elif isinstance(leaf, dict):
        TreeNode.check_hierarchies(root, **leaf, check_fun=check_fun)
      else:
        raise ValueError(f'Do not support {type(leaf)}.')
    for leaf in named_leaves.values():
      if not isinstance(leaf, bst.Module):
        raise ValueError(f'Do not support {type(leaf)}. Must be instance of {bst.Module}')
      check_fun(root, leaf)


class HHTypedNeuron(DendriticDynamics, Container):
  """
  The base class for the Hodgkin-Huxley typed neuronal dynamics.
  """

  def __init__(
      self,
      size: bst.typing.Size,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      **ion_channels
  ):
    super().__init__(size, mode=mode, name=name)

    # attribute for ``Container``
    self.ion_channels = bst.visible_module_dict(self._format_elements(IonChannel, **ion_channels))

  def add_elem(self, *elems, **elements):
    """
    Add new elements.

    Args:
      elements: children objects.
    """
    TreeNode.check_hierarchies(type(self), *elems, **elements)
    self.ion_channels.update(self._format_elements(object, *elems, **elements))


class IonChannel(DendriticDynamics, TreeNode):
  def current(self, *args, **kwargs):
    raise NotImplementedError

  def before_integral(self, *args, **kwargs):
    raise NotImplementedError

  def compute_derivative(self, *args, **kwargs):
    raise NotImplementedError

  def after_integral(self, *args, **kwargs):
    raise NotImplementedError

  def reset_state(self, *args, **kwargs):
    raise NotImplementedError

  def init_state(self, *args, **kwargs):
    raise NotImplementedError


class IonInfo(NamedTuple):
  C: bst.typing.ArrayLike
  E: bst.typing.ArrayLike


class Ion(IonChannel, Container):
  """The brainpy_object calcium dynamics.

  Args:
    size: The size of the simulation target.
    name: The name of the object.
  """

  # The type of the master object.
  root_type = HHTypedNeuron

  # Reversal potential.
  E: bst.typing.ArrayLike | bst.State

  # Ion concentration.
  C: bst.typing.ArrayLike | bst.State

  def __init__(
      self,
      size: bst.typing.Size,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      **channels
  ):
    super().__init__(size, mode=mode, name=name, **channels)
    self.channels: Dict[str, Channel] = bst.visible_module_dict()
    self.channels.update(self._format_elements(Channel, **channels))

    self._external_currents: Dict[str, Callable] = dict()

  def before_integral(self, V):
    nodes = self.nodes(level=1, include_self=False).subset(Channel)
    for node in nodes.values():
      node.before_integral(V, self.pack_info())

  def compute_derivative(self, V):
    nodes = self.nodes(level=1, include_self=False).subset(Channel)
    for node in nodes.values():
      node.compute_derivative(V, self.pack_info())

  def after_integral(self, V):
    nodes = self.nodes(level=1, include_self=False).subset(Channel)
    for node in nodes.values():
      node.after_integral(V, self.pack_info())

  def current(self, V, include_external: bool = False):
    """
    Generate ion channel current.

    Args:
      V: The membrane potential.
      include_external: Include the external current.

    Returns:
      Current.
    """
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())

    ion_info = self.pack_info()
    current = None
    if len(nodes) > 0:
      for node in nodes:
        node: Channel
        new_current = node.current(V, ion_info)
        current = new_current if current is None else (current + new_current)
    if include_external:
      for key, node in self._external_currents.items():
        node: Callable
        current = current + node(V, ion_info)
    return current

  def init_state(self, V, batch_size: int = None):
    nodes = self.nodes(level=1, include_self=False).subset(Channel).values()
    self.check_hierarchies(type(self), *tuple(nodes))
    ion_info = self.pack_info()
    for node in nodes:
      node: Channel
      node.init_state(V, ion_info, batch_size)

  def reset_state(self, V, batch_size: int = None):
    nodes = self.nodes(level=1, include_self=False).subset(Channel).values()
    ion_info = self.pack_info()
    for node in nodes:
      node: Channel
      node.reset_state(V, ion_info, batch_size)

  def register_external_current(self, key: str, fun: Callable):
    if key in self._external_currents:
      raise ValueError
    self._external_currents[key] = fun

  def pack_info(self):
    E = self.E.value if isinstance(self.E, bst.State) else self.E
    C = self.C.value if isinstance(self.C, bst.State) else self.C
    return IonInfo(E=E, C=C)

  def add_elem(self, *elems, **elements):
    """
    Add new elements.

    Args:
      elements: children objects.
    """
    self.check_hierarchies(type(self), *elems, **elements)
    self.channels.update(self._format_elements(object, *elems, **elements))


class MixIons(IonChannel, Container):
  """
  Mixing Ions.

  Args:
    ions: Instances of ions. This option defines the master types of all children objects.
  """

  root_type = HHTypedNeuron

  def __init__(
      self,
      *ions,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      **channels
  ):
    # TODO: check "ions" should be independent from each other
    assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ions. '
    assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
    size = ions[0].size
    for ion in ions:
      assert ion.size == size, f'The size of all ions should be the same. But we got {ions}.'
    super().__init__(size=size, name=name, mode=mode)

    # Store the ion instances
    self.ions: Sequence['Ion'] = tuple(ions)
    self._ion_types = tuple([type(ion) for ion in self.ions])

    # Store the ion channel channels
    self.channels: Dict[str, Channel] = bst.visible_module_dict()
    self.channels.update(self._format_elements(Channel, **channels))

  def before_integral(self, V):
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())
    for node in nodes:
      ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
      node.before_integral(V, *ion_infos)

  def compute_derivative(self, V):
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())
    for node in nodes:
      ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
      node.compute_derivative(V, *ion_infos)

  def after_integral(self, V):
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())
    for node in nodes:
      ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
      node.after_integral(V, *ion_infos)

  def current(self, V):
    """Generate ion channel current.

    Args:
      V: The membrane potential.

    Returns:
      Current.
    """
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())

    if len(nodes) == 0:
      return 0.
    else:
      current = None
      for node in nodes:
        infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
        current = node.current(V, *infos) if current is None else (current + node.current(V, *infos))
      return current

  def init_state(self, V, batch_size: int = None):
    nodes = self.nodes(level=1, include_self=False).subset(Channel).values()
    self.check_hierarchies(self._ion_types, *tuple(nodes), check_fun=self._check_hierarchy)
    for node in nodes:
      node: Channel
      infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
      node.init_state(V, *infos, batch_size)

  def reset_state(self, V, batch_size=None):
    nodes = tuple(self.nodes(level=1, include_self=False).subset(Channel).values())
    for node in nodes:
      infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
      node.reset_state(V, *infos, batch_size)

  def _check_hierarchy(self, ions, leaf):
    # 'root_type' should be a brainpy.mixin.JointType
    self._check_root(leaf)
    for cls in leaf.root_type.__args__:
      if not any([issubclass(root, cls) for root in ions]):
        raise TypeError(
          f'Type does not match. {leaf} requires a master with type '
          f'of {leaf.root_type}, but the master type now is {ions}.'
        )

  def add_elem(self, *elems, **elements):
    """
    Add new elements.

    Args:
      elements: children objects.
    """
    self.check_hierarchies(self._ion_types, *elems, check_fun=self._check_hierarchy, **elements)
    self.channels.update(self._format_elements(Channel, *elems, **elements))
    for elem in tuple(elems) + tuple(elements.values()):
      elem: Channel
      for ion_root in elem.root_type.__args__:
        ion = self._get_ion(ion_root)
        ion.register_external_current(elem.name, self._get_ion_fun(ion, elem))

  def _get_ion_fun(self, ion: 'Ion', node: 'Channel'):
    def fun(V, ion_info):
      infos = tuple(
        [(ion_info if isinstance(ion, root) else self._get_ion(root).pack_info())
         for root in node.root_type.__args__]
      )
      return node.current(V, *infos)

    return fun

  def _get_ion(self, cls):
    for ion in self.ions:
      if isinstance(ion, cls):
        return ion
    else:
      raise ValueError(f'No instance of {cls} is found.')

  def _check_root(self, leaf):
    if not isinstance(leaf.root_type, _JointGenericAlias):
      raise TypeError(
        f'{self.__class__.__name__} requires leaf nodes that have the root_type of '
        f'"brainpy.mixin.JointType". However, we got {leaf.root_type}'
      )


def mix_ions(*ions) -> MixIons:
  """Create mixed ions.

  Args:
    ions: Ion instances.

  Returns:
    Instance of MixIons.
  """
  for ion in ions:
    assert isinstance(ion, Ion), f'Must be instance of {Ion.__name__}. But got {type(ion)}'
  assert len(ions) > 0, ''
  return MixIons(*ions)


class Channel(IonChannel):
  """Base class for ion channels."""
