Notations
==========

This section provides the notations and definitions commonly used in reinforcement learning. The following table outlines the symbols and their meanings.

.. list-table::
   :widths: 15 60
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`s \in \mathcal{S}`
     - States.
   * - :math:`a \in \mathcal{A}`
     - Actions.
   * - :math:`r \in \mathcal{R}`
     - Rewards.
   * - :math:`S_t, A_t, R_t`
     - State, action, and reward at time step :math:`t` of one trajectory. I may occasionally use :math:`s_t, a_t, r_t` as well.
   * - :math:`\gamma`
     - Discount factor; penalty to uncertainty of future rewards; :math:`0 < \gamma \leq 1`.
   * - :math:`G_t`
     - Return; or discounted future reward: :math:`G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}`.
   * - :math:`P(s', r|s, a)`
     - Transition probability of getting to the next state :math:`s'` from the current state :math:`s` with action :math:`a` and reward :math:`r`.
   * - :math:`\pi(a|s)`
     - Stochastic policy (agent behavior strategy); :math:`\pi_\theta(.)` is a policy parameterized by :math:`\theta`.
   * - :math:`\mu(s)`
     - Deterministic policy; we can also label this as :math:`\pi(s)`, but using a different letter gives better distinction so that we can easily tell when the policy is stochastic or deterministic without further explanation. Either :math:`\pi` or :math:`\mu` is what a reinforcement learning algorithm aims to learn.
   * - :math:`V(s)`
     - State-value function measures the expected return of state :math:`s`; :math:`V_w(.)` is a value function parameterized by :math:`w`.
   * - :math:`V^\pi(s)`
     - The value of state :math:`s` when we follow a policy :math:`\pi`; :math:`V^\pi(s) = \mathbb{E}_{\pi}[G_t | S_t = s]`.
   * - :math:`Q(s, a)`
     - Action-value function is similar to :math:`V(s)`, but it assesses the expected return of a pair of state and action :math:`(s, a)`; :math:`Q_w(.)` is a value function parameterized by :math:`w`.
   * - :math:`Q^\pi(s, a)`
     - Similar to :math:`V^\pi(.)`, the value of (state, action) pair when we follow a policy :math:`\pi`; :math:`Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}[G_t | S_t = s, A_t = a]`.
   * - :math:`A(s, a)`
     - Advantage function, :math:`A(s, a) = Q(s, a) - V(s)`; it can be considered as another version of Q-value with lower variance by taking the state-value off as the baseline.
