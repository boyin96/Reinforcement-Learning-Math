Notations
==========

This section provides the notations and definitions commonly used in reinforcement learning. The following table outlines the symbols and their meanings.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`s \in \mathcal{S}`
     - State space.
   * - :math:`a \in \mathcal{A}`
     - Action space.
   * - :math:`r \in \mathcal{R}`
     - Reward space, being equal to the space of values of the reward function.
   * - :math:`\mathcal{R}^a_s`
     - Reward funciton, :math:`\mathcal{R}_s^a=\mathbb{E}\left[R_{t+1} \mid S_t=s, A_t=a\right]`.
   * - :math:`\mathcal{H}(\cdot)`
     - Entropy of the source, :math:`\mathcal{H}(X):=-\sum_{x \in \mathcal{X}} p(x) \log p(x)`.
   * - :math:`S_t, A_t, R_t`
     - State, action, and reward at time step :math:`t` of one trajectory.
   * - :math:`\gamma`
     - Discount factor (:math:`0 < \gamma \leq 1`).
   * - :math:`G_t`
     - Return (:math:`G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}`).
   * - :math:`P(s' | s, a)`
     - Transition probability of getting to the next state :math:`s'` from the current state :math:`s` with action :math:`a`.
   * - :math:`\pi(a|s)`
     - Stochastic policy (agent behavior strategy), :math:`\pi_\theta(.)` is a policy parameterized by :math:`\theta`.
   * - :math:`\mu(s)`
     - Deterministic policy.
   * - :math:`V(s)`
     - State-value function of a given state :math:`s`, :math:`V_w(.)` is parameterized by :math:`w`.
   * - :math:`V^\pi(s)`
     - The value of state :math:`s` when we follow a policy :math:`\pi`, :math:`V^\pi(s) = \mathbb{E}_{\pi}[G_t | S_t = s]`.
   * - :math:`Q(s, a)`
     - Action-value function of a given a pair of state and action :math:`(s, a)`, :math:`Q_w(.)` is parameterized by :math:`w`.
   * - :math:`Q^\pi(s, a)`
     - The value of (state, action) pair when we follow a policy :math:`\pi`, :math:`Q^\pi(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]`.
   * - :math:`A(s, a)`
     - Advantage function, :math:`A(s, a) = Q(s, a) - V(s)`.

.. note::

   In this document, we adopt the following conventions:
   
   - **Uppercase letters** represent **random variables** or **functions**, such as :math:`S, A, R`, etc.
   - **Calligraphic uppercase letters** represent **sets**, such as :math:`\mathcal{S}, \mathcal{A}, \mathcal{R}`, etc.
   - **Lowercase letters** represent **deterministic values**, such as :math:`s, a , r`, etc.

Bellman Expectation Equation
------------------------------
.. important::

   .. math::
      \begin{aligned}
      	&V^{\pi}(s)=\sum_{a\in \mathcal{A}}{\pi}(a\mid s)Q^{\pi}(s,a)\\[5pt]
      	&Q^{\pi}(s,a)=\mathcal{R} _{s}^{a}+\gamma \sum_{s^{\prime}\in \mathcal{S}}{P}\left( s^{\prime}\mid s,a \right) V^{\pi}\left( s^{\prime} \right)\\[5pt]
      	&V^{\pi}(s)=\sum_{a\in \mathcal{A}}{\pi}(a\mid s)\left( \mathcal{R} _{s}^{a}+\gamma \sum_{s^{\prime}\in \mathcal{S}}{P}\left( s^{\prime}\mid s,a \right) V^{\pi}\left( s^{\prime} \right) \right)\\[5pt]
      	&Q^{\pi}(s,a)=\mathcal{R} _{s}^{a}+\gamma \sum_{s^{\prime}\in \mathcal{S}}{P}\left( s^{\prime}\mid s,a \right) \sum_{a^{\prime}\in \mathcal{A}}{\pi}\left( a^{\prime}\mid s^{\prime} \right) Q^{\pi}\left( s^{\prime},a^{\prime} \right)\\
      \end{aligned}

References
----------------

- https://www.davidsilver.uk/teaching/

