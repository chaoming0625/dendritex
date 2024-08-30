# -*- coding: utf-8 -*-


"""
This module implements calcium-dependent potassium channels.
"""

from __future__ import annotations

from typing import Union, Callable, Optional

import brainstate as bst
import brainunit as bu

from .._base import IonInfo, Channel, State4Integral
from ..ions import Calcium, Potassium

__all__ = [
  'IAHP_De1994',
  'IKca3_1_Ma2020',
]


class KCaChannel(Channel):
  __module__ = 'dendritex.channels'

  root_type = bst.mixin.JointTypes[Potassium, Calcium]

  def before_integral(self, V, K: IonInfo, Ca: IonInfo):
    pass

  def after_integral(self, V, K: IonInfo, Ca: IonInfo):
    pass

  def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
    pass

  def current(self, V, K: IonInfo, Ca: IonInfo):
    raise NotImplementedError

  def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
    pass

  def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
    pass


class IAHP_De1994(KCaChannel):
  r"""The calcium-dependent potassium current model proposed by (Destexhe, et al., 1994) [1]_.

  Both in vivo (Contreras et al. 1993; Mulle et al. 1986) and in
  vitro recordings (Avanzini et al. 1989) show the presence of a
  marked after-hyper-polarization (AHP) after each burst of the RE
  cell. This slow AHP is mediated by a slow :math:`Ca^{2+}`-dependent K+
  current (Bal and McCormick 1993). (Destexhe, et al., 1994) adopted a
  modified version of a model of :math:`I_{KCa}` introduced previously (Yamada et al.
  1989) that requires the binding of :math:`nCa^{2+}` to open the channel

  .. math::

      (\text { closed })+n \mathrm{Ca}_{i}^{2+} \underset{\beta}{\stackrel{\alpha}{\rightleftharpoons}(\text { open })

  where :math:`Ca_i^{2+}` is the intracellular calcium and :math:`\alpha` and
  :math:`\beta` are rate constants. The ionic current is then given by

  .. math::

      \begin{aligned}
      I_{AHP} &= g_{\mathrm{max}} p^2 (V - E_K) \\
      {dp \over dt} &= \phi {p_{\infty}(V, [Ca^{2+}]_i) - p \over \tau_p(V, [Ca^{2+}]_i)} \\
      p_{\infty} &=\frac{\alpha[Ca^{2+}]_i^n}{\left(\alpha[Ca^{2+}]_i^n + \beta\right)} \\
      \tau_p &=\frac{1}{\left(\alpha[Ca^{2+}]_i +\beta\right)}
      \end{aligned}

  where :math:`E` is the reversal potential, :math:`g_{max}` is the maximum conductance,
  :math:`[Ca^{2+}]_i` is the intracellular Calcium concentration.
  The values :math:`n=2, \alpha=48 \mathrm{~ms}^{-1} \mathrm{mM}^{-2}` and
  :math:`\beta=0.03 \mathrm{~ms}^{-1}` yielded AHPs very similar to those RE cells
  recorded in vivo and in vitro.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).

  References
  ----------

  .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.

  """
  __module__ = 'dendritex.channels'

  root_type = bst.mixin.JointTypes[Potassium, Calcium]

  def __init__(
      self,
      size: bst.typing.Size,
      n: Union[bst.typing.ArrayLike, Callable] = 2,
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      alpha: Union[bst.typing.ArrayLike, Callable] = 48.,
      beta: Union[bst.typing.ArrayLike, Callable] = 0.09,
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.n = bst.init.param(n, self.varshape, allow_none=False)
    self.alpha = bst.init.param(alpha, self.varshape, allow_none=False)
    self.beta = bst.init.param(beta, self.varshape, allow_none=False)
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)

  def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
    C2 = self.alpha * bu.math.power(Ca.C / bu.mM, self.n)
    C3 = C2 + self.beta
    self.p.derivative = self.phi * (C2 / C3 - self.p.value) * C3 / bu.ms

  def current(self, V, K: IonInfo, Ca: IonInfo):
    return self.g_max * self.p.value * self.p.value * (Ca.E - V)

  def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
    C2 = self.alpha * bu.math.power(Ca.C / bu.mM, self.n)
    C3 = C2 + self.beta
    if batch_size is None:
      self.p.value = bu.math.broadcast_to(C2 / C3, self.varshape)
    else:
      self.p.value = bu.math.broadcast_to(C2 / C3, (batch_size,) + self.varshape)
      assert self.p.value.shape[0] == batch_size


class IKca3_1_Ma2020(KCaChannel):
  r'''
    TITLE Calcium dependent potassium channel
  : Implemented in Rubin and Cleland (2006) J Neurophysiology
  : Parameters from Bhalla and Bower (1993) J Neurophysiology
  : Adapted from /usr/local/neuron/demo/release/nachan.mod - squid
  :   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]
  '''
  __module__ = 'dendritex.channels'

  root_type = bst.mixin.JointTypes[Potassium ,Calcium]

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 120. * (bu.mS / bu.cm ** 2),
      T_base: bst.typing.ArrayLike = 3.,
      T: bst.typing.ArrayLike = 22,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.phi = bst.init.param(T_base ** ((T - 37) / 10), self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

    self.p_beta =0.05

  def current(self, V, K: IonInfo, Ca: IonInfo):
    return self.g_max * self.p.value * (K.E - V)
  
  def p_tau(self,V, Ca):
    return 1/(self.p_alpha(V,Ca) + self.p_beta)
  
  def p_inf(self, V , Ca):
    return self.p_alpha(V,Ca)/(self.p_alpha(V,Ca) + self.p_beta)
  
  def p_alpha(self, V ,Ca):
    V = V / bu.mV
    return self.p_vdep(V) * self.p_concdep(Ca)
  
  def p_vdep(self, V ):
    return bu.math.exp((V+70.)/27.)
  
  def p_concdep(self, Ca):
    concdep_1 = 500 * (0.015-Ca.C/ bu.mM)/( bu.math.exp((0.015-Ca.C/bu.mM)/0.0013) -1 )
    concdep_2 = 500 * 0.005/( bu.math.exp(0.005/0.0013) -1 )
    return bu.math.where(Ca.C/bu.mM<0.01,concdep_1,concdep_2)
  
  def init_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.reset_state(V,K,Ca)

  def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size=None):
    self.p.value = self.p_inf(V,Ca)

  def compute_derivative(self, V, K: IonInfo, Ca: IonInfo):
    self.p.derivative = self.phi * (self.p_inf(V,Ca) - self.p.value) / self.p_tau(V,Ca) / bu.ms
