# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent sodium channels.

"""

from __future__ import annotations

from typing import Union, Callable, Optional

import brainstate as bst
import brainunit as bu

from .._base import Channel, IonInfo, State4Integral
from ..ions import Sodium

__all__ = [
  'SodiumChannel',
  'INa_Ba2002',
  'INa_TM1991',
  'INa_HH1952',
]


class SodiumChannel(Channel):
  """Base class for sodium channel dynamics."""

  root_type = Sodium

  def before_integral(self, V, Na: IonInfo):
    pass

  def after_integral(self, V, Na: IonInfo):
    pass

  def compute_derivative(self, V, Na: IonInfo):
    pass

  def current(self, V, Na: IonInfo):
    raise NotImplementedError

  def init_state(self, V, Na: IonInfo, batch_size: int = None):
    pass

  def reset_state(self, V, Na: IonInfo, batch_size: int = None):
    pass


class INa_p3q_markov(SodiumChannel):
  r"""
  The sodium current model of :math:`p^3q` current which described with first-order Markov chain.

  The general model can be used to model the dynamics with:

  .. math::

    \begin{aligned}
    I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
    \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
    \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
    \end{aligned}

  where :math:`\phi` is a temperature-dependent factor.

  Parameters
  ----------
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  phi : float, ArrayType, Callable, Initializer
    The temperature-dependent factor.
  name: str
    The name of the object.

  """

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 90. * (bu.mS / bu.cm ** 2),
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
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)

  def init_state(self, V, Na: IonInfo, batch_size=None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, Na: IonInfo, batch_size=None):
    alpha = self.f_p_alpha(V)
    beta = self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha = self.f_q_alpha(V)
    beta = self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)

  def compute_derivative(self, V, Na: IonInfo):
    p = self.p.value
    q = self.q.value
    self.p.derivative = self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p) / bu.ms
    self.q.derivative = self.phi * (self.f_q_alpha(V) * (1. - q) - self.f_q_beta(V) * q) / bu.ms

  def current(self, V, Na: IonInfo):
    return self.g_max * self.p.value ** 3 * self.q.value * (Na.E - V)

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError

  def f_q_alpha(self, V):
    raise NotImplementedError

  def f_q_beta(self, V):
    raise NotImplementedError


class INa_Ba2002(INa_p3q_markov):
  r"""
  The sodium current model.

  The sodium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

    \begin{aligned}
    I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
    \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
    \alpha_{p} &=\frac{0.32\left(V-V_{sh}-13\right)}{1-\exp \left(-\left(V-V_{sh}-13\right) / 4\right)} \\
    \beta_{p} &=\frac{-0.28\left(V-V_{sh}-40\right)}{1-\exp \left(\left(V-V_{sh}-40\right) / 5\right)} \\
    \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
    \alpha_q &=0.128 \exp \left(-\left(V-V_{sh}-17\right) / 18\right) \\
    \beta_q &= \frac{4}{1+\exp \left(-\left(V-V_{sh}-40\right) / 5\right)}
    \end{aligned}

  where :math:`\phi` is a temperature-dependent factor, which is given by
  :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

  Parameters
  ----------
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  T : float, ArrayType
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : float, ArrayType, Callable, Initializer
    The shift of the membrane potential to spike.

  References
  ----------

  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  See Also
  --------
  INa_TM1991
  """

  def __init__(
      self,
      size: bst.typing.Size,
      T: bst.typing.ArrayLike = 36.,
      g_max: Union[bst.typing.ArrayLike, Callable] = 90. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -50. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi=3 ** ((T - 36) / 10),
      g_max=g_max,
      mode=mode
    )
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = (V - self.V_sh) / bu.mV
    temp = V - 13.
    return 0.32 * temp / (1. - bu.math.exp(-temp / 4.))

  def f_p_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    temp = V - 40.
    return -0.28 * temp / (1. - bu.math.exp(temp / 5.))

  def f_q_alpha(self, V):
    V = (V - self.V_sh) / bu.mV
    return 0.128 * bu.math.exp(-(V - 17.) / 18.)

  def f_q_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 4. / (1. + bu.math.exp(-(V - 40.) / 5.))


class INa_TM1991(INa_p3q_markov):
  r"""
  The sodium current model described by (Traub and Miles, 1991) [1]_.

  The dynamics of this sodium current model is given by:

  .. math::

     \begin{split}
     \begin{aligned}
        I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
        \frac {dm} {dt} &= \phi(\alpha_m (1-x)  - \beta_m) \\
        &\alpha_m(V) = 0.32 \frac{(13 - V + V_{sh})}{\exp((13 - V +V_{sh}) / 4) - 1.}  \\
        &\beta_m(V) = 0.28 \frac{(V - V_{sh} - 40)}{(\exp((V - V_{sh} - 40) / 5) - 1)}  \\
        \frac {dh} {dt} &= \phi(\alpha_h (1-x)  - \beta_h) \\
        &\alpha_h(V) = 0.128 * \exp((17 - V + V_{sh}) / 18)  \\
        &\beta_h(V) = 4. / (1 + \exp(-(V - V_{sh} - 40) / 5)) \\
     \end{aligned}
     \end{split}

  where :math:`V_{sh}` is the membrane shift (default -63 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh: float, ArrayType, Callable, Initializer
    The membrane shift.

  References
  ----------
  .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
         Vol. 777. Cambridge University Press, 1991.

  See Also
  --------
  INa_Ba2002
  """

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 120. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      V_sh: Union[bst.typing.ArrayLike, Callable] = -63. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi=phi,
      g_max=g_max,
      mode=mode
    )
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = (self.V_sh - V) / bu.mV
    temp = 13 + V
    return 0.32 * temp / (bu.math.exp(temp / 4) - 1.)

  def f_p_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    temp = V - 40
    return 0.28 * temp / (bu.math.exp(temp / 5) - 1)

  def f_q_alpha(self, V):
    V = (- V + self.V_sh) / bu.mV
    return 0.128 * bu.math.exp((17 + V) / 18)

  def f_q_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 4. / (1 + bu.math.exp(-(V - 40) / 5))


class INa_HH1952(INa_p3q_markov):
  r"""
  The sodium current model described by Hodgkin–Huxley model [1]_.

  The dynamics of this sodium current model is given by:

  .. math::

     \begin{split}
     \begin{aligned}
        I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
        \frac {dm} {dt} &= \phi (\alpha_m (1-x)  - \beta_m) \\
        &\alpha_m(V) = \frac {0.1(V-V_{sh}-5)}{1-\exp(\frac{-(V -V_{sh} -5)} {10})}  \\
        &\beta_m(V) = 4.0 \exp(\frac{-(V -V_{sh}+ 20)} {18})  \\
        \frac {dh} {dt} &= \phi (\alpha_h (1-x)  - \beta_h) \\
        &\alpha_h(V) = 0.07 \exp(\frac{-(V-V_{sh}+20)}{20})  \\
        &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V -V_{sh}-10)} {10})} \\
     \end{aligned}
     \end{split}

  where :math:`V_{sh}` is the membrane shift (default -45 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  name: str
    The name of the object.
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh: float, ArrayType, Callable, Initializer
    The membrane shift.

  References
  ----------
  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
         membrane current and its application to conduction and excitation in
         nerve." The Journal of physiology 117.4 (1952): 500.

  See Also
  --------
  IK_HH1952
  """

  def __init__(
      self,
      size: bst.typing.Size,
      g_max: Union[bst.typing.ArrayLike, Callable] = 120. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      V_sh: Union[bst.typing.ArrayLike, Callable] = -45. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi=phi,
      g_max=g_max,
      mode=mode
    )
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = (V - self.V_sh) / bu.mV - 5
    return 0.1 * temp / (1 - bu.math.exp(-temp / 10))

  def f_p_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 4.0 * bu.math.exp(-(V + 20) / 18)

  def f_q_alpha(self, V):
    V = (V - self.V_sh) / bu.mV
    return 0.07 * bu.math.exp(-(V + 20) / 20.)

  def f_q_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1 / (1 + bu.math.exp(-(V - 10) / 10))
