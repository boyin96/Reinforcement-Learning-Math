Notations
==========

This section provides the notations and definitions commonly used in reinforcement learning. The following table outlines the symbols and their meanings.

.. list-table::
   :widths: 15 60
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`s \in \mathcal{S}`
     - State space.
   * - :math:`a \in \mathcal{A}`
     - Action space.
   * - :math:`r \in \mathcal{R}`
     - Reward space, being equal to the space of values of the reward function.
   * - :math:`S_t, A_t, R_t`
     - State, action, and reward at time step :math:`t` of one trajectory.
   * - :math:`\gamma`
     - Discount factor (:math:`0 < \gamma \leq 1`).
   * - :math:`G_t`
     - Return (:math:`G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}`).
   * - :math:`P(s', r|s, a)`
     - Transition probability of getting to the next state :math:`s'` from the current state :math:`s` with action :math:`a` and reward :math:`r`.
   * - :math:`\pi(a|s)`
     - Stochastic policy (agent behavior strategy), :math:`\pi_\theta(.)` is a policy parameterized by :math:`\theta`.
   * - :math:`\mu(s)`
     - Deterministic policy.
   * - :math:`V(s)`
     - State-value function measures the expected return of state :math:`s`, :math:`V_w(.)` is a value function parameterized by :math:`w`.
   * - :math:`V^\pi(s)`
     - The value of state :math:`s` when we follow a policy :math:`\pi`, :math:`V^\pi(s) = \mathbb{E}_{\pi}[G_t | S_t = s]`.
   * - :math:`Q(s, a)`
     - Action-value function assesses the expected return of a pair of state and action :math:`(s, a)`, :math:`Q_w(.)` is a value function parameterized by :math:`w`.
   * - :math:`Q^\pi(s, a)`
     - The value of (state, action) pair when we follow a policy :math:`\pi`, :math:`Q^\pi(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]`.
   * - :math:`A(s, a)`
     - Advantage function, :math:`A(s, a) = Q(s, a) - V(s)`.

.. note::

   In this document, we adopt the following conventions:
   
   - **Uppercase letters** represent **random variables** or **functions**, such as :math:` S, A, R`, etc.
   - **Calligraphic uppercase letters** represent **sets**, such as :math:`\mathcal{S}, \mathcal{A}, \mathcal{R}`, etc.
   - **Lowercase letters** represent **deterministic values**, such as :math:`s, a , r`, etc.
