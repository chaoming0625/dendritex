# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent calcium channels.

"""

from __future__ import annotations

from typing import Union, Callable, Optional

import brainstate as bst
import brainunit as bu

from .._base import Channel, IonInfo
from .._integrators import State4Integral
from ..ions import Calcium

__all__ = [
  'CalciumChannel',

  'ICaN_IS2008',
  'ICaT_HM1992',
  'ICaT_HP1992',
  'ICaHT_HM1992',
  'ICaHT_Re1993',
  'ICaL_IS2008',
]


class CalciumChannel(Channel):
  """Base class for Calcium ion channels."""

  root_type = Calcium

  def update(self, V, Ca):
    raise NotImplementedError

  def current(self, V, Ca):
    raise NotImplementedError

  def reset_state(self, V, Ca, batch_size: int = None):
    raise NotImplementedError('Must be implemented by the subclass.')


class ICaN_IS2008(CalciumChannel):
  r"""The calcium-activated non-selective cation channel model
  proposed by (Inoue & Strowbridge, 2008) [2]_.

  The dynamics of the calcium-activated non-selective cation channel model [1]_ [2]_ is given by:

  .. math::

      \begin{aligned}
      I_{CAN} &=g_{\mathrm{max}} M\left([Ca^{2+}]_{i}\right) p \left(V-E\right)\\
      &M\left([Ca^{2+}]_{i}\right) ={[Ca^{2+}]_{i} \over 0.2+[Ca^{2+}]_{i}} \\
      &{dp \over dt} = {\phi \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1.0 \over 1 + \exp(-(V + 43) / 5.2)} \\
      &\tau_{p} = {2.7 \over \exp(-(V + 55) / 15) + \exp((V + 55) / 15)} + 1.6
      \end{aligned}

  where :math:`\phi` is the temperature factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature factor.

  References
  ----------

  .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.
  .. [2] Inoue T, Strowbridge BW (2008) Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons.
         J Neurophysiol 99: 187–199.
  """

  '''The type of the master object.'''
  root_type = Calcium

  def __init__(
      self,
      size: bst.typing.Size,
      E: Union[bst.typing.ArrayLike, Callable] = 10. * bu.mV,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
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
    self.E = bst.init.param(E, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    V = V / bu.mV
    self.p.value = 1.0 / (1 + bu.math.exp(-(V + 43.) / 5.2))

  def dp(self, p, t, V):
    V = V / bu.mV
    phi_p = 1.0 / (1 + bu.math.exp(-(V + 43.) / 5.2))
    p_inf = 2.7 / (bu.math.exp(-(V + 55.) / 15.) + bu.math.exp((V + 55.) / 15.)) + 1.6
    return self.phi * (phi_p - p) / p_inf / bu.ms

  def update(self, V, Ca):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get('dt')

  def current(self, V, Ca):
    M = Ca.C / (Ca.C + 0.2 * bu.mM)
    g = self.g_max * M * self.p.value
    return g * (self.E - V)


class _ICa_p2q_ss(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with steady-state format.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 3.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      mode: Optional[bst.mixin.Mode] = None,
      name: Optional[str] = None
  ):
    super().__init__(
      size,
      name=name,
      mode=mode,
    )

    # parameters
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if batch_size is not None:
      assert self.p.value.shape[0] == batch_size
      assert self.q.value.shape[0] == batch_size

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V) / bu.ms

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V) / bu.ms

  def update(self, V, Ca):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get('dt')
    self.q.value += self.dq(self.q.value, bst.environ.get('t'), V) * bst.environ.get('dt')

  def current(self, V, Ca):
    return self.g_max * self.p.value * self.p.value * self.q.value * (Ca.E - V)

  def f_p_inf(self, V):
    raise NotImplementedError

  def f_p_tau(self, V):
    raise NotImplementedError

  def f_q_inf(self, V):
    raise NotImplementedError

  def f_q_tau(self, V):
    raise NotImplementedError


class _ICa_p2q_markov(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with first-order Markov chain.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= \phi_p (\alpha_p(V)(1-p) - \beta_p(V)p) \\
      {dq \over dt} &= \phi_q (\alpha_q(V)(1-q) - \beta_q(V)q) \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 3.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Ca, batch_size=None):
    alpha, beta = self.f_p_alpha(V), self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha, beta = self.f_q_alpha(V), self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_alpha(V) * (1 - p) - self.f_p_beta(V) * p) / bu.ms

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_alpha(V) * (1 - q) - self.f_q_beta(V) * q) / bu.ms

  def update(self, V, Ca):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get('dt')
    self.q.value += self.dq(self.q.value, bst.environ.get('t'), V) * bst.environ.get('dt')

  def current(self, V, Ca):
    return self.g_max * self.p.value * self.p.value * self.q.value * (Ca.E - V)

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError

  def f_q_alpha(self, V):
    raise NotImplementedError

  def f_q_beta(self, V):
    raise NotImplementedError


class ICaT_HM1992(_ICa_p2q_ss):
  r"""
  The low-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+59-V_{sh}) / 6.2]} \\
      &\tau_{p} = 0.612 + {1 \over \exp [-(V+132.-V_{sh}) / 16.7]+\exp [(V+16.8-V_{sh}) / 18.2]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+83-V_{sh}) / 4]} \\
      & \begin{array}{l} \tau_{q} = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
          \tau_{q} = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array}

  where :math:`\phi_p = 3.55^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 3.55,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -3. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1. / (1 + bu.math.exp(-(V + 59. - self.V_sh) / 6.2))

  def f_p_tau(self, V):
    V = V / bu.mV
    return 1. / (bu.math.exp(-(V + 132. - self.V_sh) / 16.7) +
                 bu.math.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612

  def f_q_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp((V + 83. - self.V_sh) / 4.0))

  def f_q_tau(self, V):
    V = V / bu.mV
    return bu.math.where(V >= (-80. + self.V_sh),
                         bu.math.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                         bu.math.exp((V + 467. - self.V_sh) / 66.6))


class ICaT_HP1992(_ICa_p2q_ss):
  r"""The low-threshold T-type calcium current model for thalamic
  reticular nucleus proposed by (Huguenard & Prince, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model in thalamic
  reticular nucleus neurons [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+52-V_{sh}) / 7.4]}  \\
      &\tau_{p} = 3+{1 \over \exp [(V+27-V_{sh}) / 10]+\exp [-(V+102-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+80-V_{sh}) / 5]} \\
      & \tau_q = 85+ {1 \over \exp [(V+48-V_{sh}) / 4]+\exp [-(V+407-V_{sh}) / 50]}

  where :math:`\phi_p = 5^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, Prince DA (1992) A novel T-type current underlies
         prolonged Ca2+- dependent burst firing in GABAergic neurons of rat
         thalamic reticular nucleus. J Neurosci 12: 3804–3817.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 5.,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1.75 * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -3. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 52. - self.V_sh) / 7.4))

  def f_p_tau(self, V):
    V = V / bu.mV
    return 3. + 1. / (bu.math.exp((V + 27. - self.V_sh) / 10.) +
                      bu.math.exp(-(V + 102. - self.V_sh) / 15.))

  def f_q_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp((V + 80. - self.V_sh) / 5.))

  def f_q_tau(self, V):
    V = V / bu.mV
    return 85. + 1. / (bu.math.exp((V + 48. - self.V_sh) / 4.) +
                       bu.math.exp(-(V + 407. - self.V_sh) / 50.))


class ICaHT_HM1992(_ICa_p2q_ss):
  r"""The high-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The high-threshold T-type calcium current model is adopted from [1]_.
  Its dynamics is given by

  .. math::

      \begin{aligned}
      I_{\mathrm{Ca/HT}} &= g_{\mathrm{max}} p^2 q (V-E_{Ca})
      \\
      {dp \over dt} &= {\phi_{p} \cdot (p_{\infty} - p) \over \tau_{p}} \\
      &\tau_{p} =\frac{1}{\exp \left(\frac{V+132-V_{sh}}{-16.7}\right)+\exp \left(\frac{V+16.8-V_{sh}}{18.2}\right)}+0.612 \\
      & p_{\infty} = {1 \over 1+exp[-(V+59-V_{sh}) / 6.2]}
      \\
      {dq \over dt} &= {\phi_{q} \cdot (q_{\infty} - h) \over \tau_{q}} \\
      & \begin{array}{l} \tau_q = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
      \tau_q = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array} \\
      &q_{\infty}  = {1 \over 1+exp[(V+83 -V_{shift})/4]}
      \end{aligned}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : bst.typing.ArrayLike, Callable
    The maximum conductance.
  V_sh : bst.typing.ArrayLike, Callable
    The membrane potential shift.

  References
  ----------
  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 3.55,
      T_base_q: bst.typing.ArrayLike = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 2. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 25. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=T_base_p ** ((T / bu.celsius - 24) / 10),
      phi_q=T_base_q ** ((T / bu.celsius - 24) / 10),
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 59. - self.V_sh) / 6.2))

  def f_p_tau(self, V):
    V = V / bu.mV
    return 1. / (bu.math.exp(-(V + 132. - self.V_sh) / 16.7) +
                 bu.math.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612

  def f_q_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp((V + 83. - self.V_sh) / 4.))

  def f_q_tau(self, V):
    V = V / bu.mV
    return bu.math.where(V >= (-80. + self.V_sh),
                         bu.math.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                         bu.math.exp((V + 467. - self.V_sh) / 66.6))


class ICaHT_Re1993(_ICa_p2q_markov):
  r"""The high-threshold T-type calcium current model proposed by (Reuveni, et al., 1993) [1]_.

  HVA Calcium current was described for neocortical neurons by Sayer et al. (1990).
  Its dynamics is given by (the rate functions are measured under 36 Celsius):

  .. math::

     \begin{aligned}
      I_{L} &=\bar{g}_{L} q^{2} r\left(V-E_{\mathrm{Ca}}\right) \\
      \frac{\mathrm{d} q}{\mathrm{~d} t} &= \phi_p (\alpha_{q}(V)(1-q)-\beta_{q}(V) q) \\
      \frac{\mathrm{d} r}{\mathrm{~d} t} &= \phi_q (\alpha_{r}(V)(1-r)-\beta_{r}(V) r) \\
      \alpha_{q} &=\frac{0.055(-27-V+V_{sh})}{\exp [(-27-V+V_{sh}) / 3.8]-1} \\
      \beta_{q} &=0.94 \exp [(-75-V+V_{sh}) / 17] \\
      \alpha_{r} &=0.000457 \exp [(-13-V+V_{sh}) / 50] \\
      \beta_{r} &=\frac{0.0065}{\exp [(-15-V+V_{sh}) / 28]+1},
      \end{aligned}

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
    If `None`, :math:`\phi_p = \mathrm{T_base_p}^{\frac{T-23}{10}}`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.
    If `None`, :math:`\phi_q = \mathrm{T_base_q}^{\frac{T-23}{10}}`.

  References
  ----------
  .. [1] Reuveni, I., et al. "Stepwise repolarization from Ca2+ plateaus
         in neocortical pyramidal cells: evidence for nonhomogeneous
         distribution of HVA Ca2+ channels in dendrites." Journal of
         Neuroscience 13.11 (1993): 4609-4621.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      T_base_p: bst.typing.ArrayLike = 2.3,
      T_base_q: bst.typing.ArrayLike = 2.3,
      phi_p: Union[bst.typing.ArrayLike, Callable] = None,
      phi_q: Union[bst.typing.ArrayLike, Callable] = None,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi_p = T_base_p ** ((T / bu.celsius - 23.) / 10.) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T / bu.celsius - 23.) / 10.) if phi_q is None else phi_q
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = V / bu.mV
    temp = -27 - V + self.V_sh
    return 0.055 * temp / (bu.math.exp(temp / 3.8) - 1)

  def f_p_beta(self, V):
    V = V / bu.mV
    return 0.94 * bu.math.exp((-75. - V + self.V_sh) / 17.)

  def f_q_alpha(self, V):
    V = V / bu.mV
    return 0.000457 * bu.math.exp((-13. - V + self.V_sh) / 50.)

  def f_q_beta(self, V):
    V = V / bu.mV
    return 0.0065 / (bu.math.exp((-15. - V + self.V_sh) / 28.) + 1.)


class ICaL_IS2008(_ICa_p2q_ss):
  r"""The L-type calcium channel model proposed by (Inoue & Strowbridge, 2008) [1]_.

  The L-type calcium channel model is adopted from (Inoue, et, al., 2008) [1]_.
  Its dynamics is given by:

  .. math::

      I_{CaL} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      & p_{\infty} = {1 \over 1+\exp [-(V+10-V_{sh}) / 4.]} \\
      & \tau_{p} = 0.4+{0.7 \over \exp [(V+5-V_{sh}) / 15]+\exp [-(V+5-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      & q_{\infty} = {1 \over 1+\exp [(V+25-V_{sh}) / 2]} \\
      & \tau_q = 300 + {100 \over \exp [(V+40-V_{sh}) / 9.5]+\exp [-(V+40-V_{sh}) / 9.5]}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float
    The temperature.
  T_base_p : float
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float
    The maximum conductance.
  V_sh : float
    The membrane potential shift.

  References
  ----------

  .. [1] Inoue, Tsuyoshi, and Ben W. Strowbridge. "Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons." Journal of
         neurophysiology 99, no. 1 (2008): 187-199.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: Union[bst.typing.ArrayLike, Callable] = 36.,
      T_base_p: Union[bst.typing.ArrayLike, Callable] = 3.55,
      T_base_q: Union[bst.typing.ArrayLike, Callable] = 3.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 1. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=T_base_p ** ((T / bu.celsius - 24) / 10),
      phi_q=T_base_q ** ((T / bu.celsius - 24) / 10),
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base_p = bst.init.param(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = bst.init.param(T_base_q, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = V / bu.mV
    return 1. / (1 + bu.math.exp(-(V + 10. - self.V_sh) / 4.))

  def f_p_tau(self, V):
    V = V / bu.mV
    return 0.4 + .7 / (bu.math.exp(-(V + 5. - self.V_sh) / 15.) +
                       bu.math.exp((V + 5. - self.V_sh) / 15.))

  def f_q_inf(self, V):
    V = V / bu.mV
    return 1. / (1. + bu.math.exp((V + 25. - self.V_sh) / 2.))

  def f_q_tau(self, V):
    V = V / bu.mV
    return 300. + 100. / (bu.math.exp((V + 40 - self.V_sh) / 9.5) +
                          bu.math.exp(-(V + 40 - self.V_sh) / 9.5))
