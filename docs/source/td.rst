Temporal Difference Learning
=============================

Introduction
------------
Temporal Difference (TD) learning is a class of model-free reinforcement learning methods that learn by bootstrapping from the current estimate of the value function. TD methods update the value function based on other learned estimates, without waiting for a final outcome (as in Monte Carlo methods). TD learning combines the sampling of Monte Carlo with the bootstrapping of dynamic programming.

Key characteristics of TD learning:

-   **Model-free**: They learn directly from experience without requiring a model of the environment.
-   **Bootstrapping**: They update value function estimates based on other learned estimates.
-   **Online learning**: They can learn from incomplete episodes.
-   **Lower variance, potentially higher bias**: Compared to Monte Carlo methods, they typically have lower variance but can introduce bias due to bootstrapping.

TD Prediction
-------------
TD prediction involves estimating the value function for a given policy. The simplest TD method is TD(0), which updates the value function as follows:

.. math::
    V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]

Where:

-   :math:`V(S_t)` is the estimated value of state :math:`S_t`.
-   :math:`\alpha` is the learning rate (0 < :math:`\alpha` ≤ 1).
-   :math:`R_{t+1}` is the reward received after transitioning from :math:`S_t` to :math:`S_{t+1}`.
-   :math:`\gamma` is the discount factor (0 ≤ :math:`\gamma` ≤ 1).
-   :math:`V(S_{t+1})` is the estimated value of the next state :math:`S_{t+1}`.

The term :math:`R_{t+1} + \gamma V(S_{t+1}) - V(S_t)` is known as the TD error, which represents the difference between the actual reward received and the expected reward based on the current value function.

TD Control
----------
TD control aims to find the optimal policy by iteratively improving the policy based on the estimated action values. Two main TD control algorithms are:

1.  **SARSA (State-Action-Reward-State-Action)**: An on-policy TD control algorithm.

    -   **On-policy**: SARSA updates the action-value function :math:`Q(S_t, A_t)` using the action :math:`A_{t+1}` that is actually taken in the next state :math:`S_{t+1}` according to the current policy.
    -   **Update rule**:

        .. math::
            Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    Where :math:`A_{t+1} \sim \pi(·|S_{t+1})`.

2.  **Q-learning**: An off-policy TD control algorithm.

    -   **Off-policy**: Q-learning updates the action-value function :math:`Q(S_t, A_t)` using the maximum possible action-value in the next state :math:`S_{t+1}`, regardless of the action that is actually taken.
    -   **Update rule**:

        .. math::
            Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

Eligibility Traces
------------------
Eligibility traces provide a mechanism for TD learning to assign credit to past states and actions. They are used in algorithms like TD(:math:`\lambda`) and SARSA(:math:`\lambda`). The eligibility trace for a state :math:`s` at time :math:`t` is denoted by :math:`e_t(s)`.

Conclusion
----------
Temporal Difference learning offers an efficient way to learn value functions and optimal policies in reinforcement learning. By bootstrapping and updating estimates based on experience, TD methods can effectively solve a wide range of control problems.

References
----------
-   `Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
