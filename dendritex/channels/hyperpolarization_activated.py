# -*- coding: utf-8 -*-

"""
This module implements hyperpolarization-activated cation channels.
"""

from __future__ import annotations

from typing import Union, Callable, Optional

import brainstate as bst
import brainunit as bu

from .._base import Channel, HHTypedNeuron, State4Integral

__all__ = [
  'Ih_HM1992',
]


class Ih_HM1992(Channel):
  r"""
  The hyperpolarization-activated cation current model propsoed by (Huguenard & McCormick, 1992) [1]_.

  The hyperpolarization-activated cation current model is adopted from
  (Huguenard, et, al., 1992) [1]_. Its dynamics is given by:

  .. math::

      \begin{aligned}
      I_h &= g_{\mathrm{max}} p \\
      \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p} \\
      p_{\infty} &=\frac{1}{1+\exp ((V+75) / 5.5)} \\
      \tau_{p} &=\frac{1}{\exp (-0.086 V-14.59)+\exp (0.0701 V-1.87)}
      \end{aligned}

  where :math:`\phi=1` is a temperature-dependent factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature-dependent factor.

  References
  ----------
  .. [1] Huguenard, John R., and David A. McCormick. "Simulation of the currents
         involved in rhythmic oscillations in thalamic relay neurons." Journal
         of neurophysiology 68, no. 4 (1992): 1373-1383.

  """
  __module__ = 'dendritex.channels'

  root_type = HHTypedNeuron

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      E: Union[bst.typing.ArrayLike, Callable] = 43. * bu.mV,
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.E = bst.init.param(E, self.varshape, allow_none=False)

  def init_state(self, V, batch_size=None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, batch_size=None):
    self.p.value = self.f_p_inf(V)

  def before_integral(self, V):
    pass

  def compute_derivative(self, V):
    self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / bu.ms

  def after_integral(self, V):
    pass

  def current(self, V):
    return self.g_max * self.p.value * (self.E - V)

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp((V + 75.) / 5.5))

  def f_p_tau(self, V):
    V = V / bu.mV
    return 1. / (bu.math.exp(-0.086 * V - 14.59) + bu.math.exp(0.0701 * V - 1.87))

# class Ih_De1996(Channel):
#   r"""
#   The hyperpolarization-activated cation current model propsoed by (Destexhe, et al., 1996) [1]_.
#
#   The full kinetic schema was
#
#   .. math::
#
#      \begin{gathered}
#      C \underset{\beta(V)}{\stackrel{\alpha(V)}{\rightleftarrows}} O \\
#      P_{0}+2 \mathrm{Ca}^{2+} \underset{k_{2}}{\stackrel{k_{1}}{\rightleftarrows}} P_{1} \\
#      O+P_{1} \underset{k_{4}}{\rightleftarrows} O_{\mathrm{L}}
#      \end{gathered}
#
#   where the first reaction represents the voltage-dependent transitions of :math:`I_h` channels
#   between closed (C) and open (O) forms, with :math:`\alpha` and :math:`\beta` as transition rates.
#   The second reaction represents the biding of intracellular :math:`\mathrm{Ca^{2+}}` ions to a
#   regulating factor (:math:`P_0` for unbound and :math:`P_1` for bound) with four binding sites for
#   calcium and rates of :math:`k_1 = 2.5e^7\, mM^{-4} \, ms^{-1}` and :math:`k_2=4e-4 \, ms^{-1}`
#   (half-activation of 0.002 mM :math:`Ca^{2+}`). The calcium-bound form :math:`P_1` associates
#   with the open form of the channel, leading to a locked open form :math:`O_L`, with rates of
#   :math:`k_3=0.1 \, ms^{-1}` and :math:`k_4 = 0.001 \, ms^{-1}`.
#
#   The current is the proportional to the relative concentration of open channels
#
#   .. math::
#
#      I_h = g_h (O+g_{inc}O_L) (V - E_h)
#
#   with a maximal conductance of :math:`\bar{g}_{\mathrm{h}}=0.02 \mathrm{mS} / \mathrm{cm}^{2}`
#   and a reversal potential of :math:`E_{\mathrm{h}}=-40 \mathrm{mV}`. Because of the factor
#   :math:`g_{\text {inc }}=2`, the conductance of the calcium-bound open state of
#   :math:`I_{\mathrm{h}}` channels is twice that of the unbound open state. This produces an
#   augmentation of conductance after the binding of :math:`\mathrm{Ca}^{2+}`, as observed in
#   sino-atrial cells (Hagiwara and Irisawa 1989).
#
#   The rates of :math:`\alpha` and :math:`\beta` are:
#
#   .. math::
#
#      & \alpha = m_{\infty} / \tau_m \\
#      & \beta = (1-m_{\infty}) / \tau_m \\
#      & m_{\infty} = 1/(1+\exp((V+75-V_{sh})/5.5)) \\
#      & \tau_m = (5.3 + 267/(\exp((V+71.5-V_{sh})/14.2) + \exp(-(V+89-V_{sh})/11.6)))
#
#   and the temperature regulating factor :math:`\phi=2^{(T-24)/10}`.
#
#   References
#   ----------
#   .. [1] Destexhe, Alain, et al. "Ionic mechanisms underlying synchronized
#          oscillations and propagating waves in a model of ferret thalamic
#          slices." Journal of neurophysiology 76.3 (1996): 2049-2070.
#   """
#
#   root_type = Calcium
#
#   def __init__(
#       self,
#       size: bst.typing.Size,
#       E: Union[bst.typing.ArrayLike, Callable] = -40. * bu.mV,
#       k2: Union[bst.typing.ArrayLike, Callable] = 4e-4,
#       k4: Union[bst.typing.ArrayLike, Callable] = 1e-3,
#       V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
#       g_max: Union[bst.typing.ArrayLike, Callable] = 0.02 * (bu.mS / bu.cm ** 2),
#       g_inc: Union[bst.typing.ArrayLike, Callable] = 2.,
#       Ca_half: Union[bst.typing.ArrayLike, Callable] = 2e-3,
#       T: bst.typing.ArrayLike = 36.,
#       T_base: bst.typing.ArrayLike = 3.,
#       phi: Union[bst.typing.ArrayLike, Callable] = None,
#       name: Optional[str] = None,
#       mode: Optional[bst.mixin.Mode] = None,
#   ):
#     super().__init__(
#       size,
#       name=name,
#       mode=mode
#     )
#
#     # parameters
#     self.T = bst.init.param(T, self.varshape, allow_none=False)
#     self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
#     if phi is None:
#       self.phi = self.T_base ** ((self.T - 24.) / 10)
#     else:
#       self.phi = bst.init.param(phi, self.varshape, allow_none=False)
#     self.E = bst.init.param(E, self.varshape, allow_none=False)
#     self.k2 = bst.init.param(k2, self.varshape, allow_none=False)
#     self.Ca_half = bst.init.param(Ca_half, self.varshape, allow_none=False)
#     self.k1 = self.k2 / self.Ca_half ** 4
#     self.k4 = bst.init.param(k4, self.varshape, allow_none=False)
#     self.k3 = self.k4 / 0.01
#     self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
#     self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
#     self.g_inc = bst.init.param(g_inc, self.varshape, allow_none=False)
#
#   def dO(self, O, t, OL, V):
#     inf = self.f_inf(V)
#     tau = self.f_tau(V)
#     alpha = inf / tau
#     beta = (1 - inf) / tau
#     return alpha * (1 - O - OL) - beta * O
#
#   def dOL(self, OL, t, O, P1):
#     return self.k3 * P1 * O - self.k4 * OL
#
#   def dP1(self, P1, t, C_Ca):
#     return self.k1 * C_Ca ** 4 * (1 - P1) - self.k2 * P1
#
#   def update_state(self, V, Ca: IonInfo):
#     self.O.value, self.OL.value, self.P1.value = self.integral(
#       self.O.value, self.OL.value, self.P1.value, bst.environ.get('t'), V=V,
#     )
#
#   def current(self, V, Ca: IonInfo):
#     return self.g_max * (self.O.value + self.g_inc * self.OL.value) * (self.E - V)
#
#   def init_state(self, V, Ca, batch_size=None):
#     self.O = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
#     self.OL = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
#     self.P1 = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
#
#   def reset_state(self, V, Ca: IonInfo, batch_size=None):
#     varshape = self.varshape if (batch_size is None) else ((batch_size,) + self.varshape)
#     k1 = self.k1 * Ca.C ** 4
#     self.P1.value = bu.math.broadcast_arrays(k1 / (k1 + self.k2), varshape)
#     inf = self.f_inf(V)
#     tau = self.f_tau(V)
#     alpha = inf / tau
#     beta = (1 - inf) / tau
#     self.O.value = alpha / (alpha + alpha * self.k3 * self.P1 / self.k4 + beta)
#     self.OL.value = self.k3 * self.P1.value * self.O.value / self.k4
#
#   def f_inf(self, V):
#     V = V / bu.mV
#     return 1 / (1 + bu.math.exp((V + 75 - self.V_sh) / 5.5))
#
#   def f_tau(self, V):
#     V = V / bu.mV
#     return (20. + 1000 / (bu.math.exp((V + 71.5 - self.V_sh) / 14.2) +
#                           bu.math.exp(-(V + 89 - self.V_sh) / 11.6))) / self.phi
