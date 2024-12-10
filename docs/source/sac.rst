Soft Actor-Critic (SAC) Algorithm
===================================

Introduction
------------

Soft Actor-Critic (SAC) is an advanced reinforcement learning algorithm that integrates the principle of maximum entropy into continuous control tasks. It is designed to balance the **exploitation of high-reward actions** with the **exploration of diverse strategies** by maximizing both the cumulative reward and the entropy of the policy. SAC has demonstrated state-of-the-art performance on various challenging continuous control problems.

The core idea of SAC is to optimize a stochastic policy by introducing an entropy term into the objective function, encouraging the policy to maintain randomness while improving performance. This enables SAC to efficiently explore the environment and avoid suboptimal deterministic solutions.

Key features of SAC include:
- **Maximum entropy framework**: Encourages diverse action selection.
- **Stochastic policies**: Enables better exploration compared to deterministic approaches.
- **Off-policy learning**: Utilizes a replay buffer for data efficiency.

Theoretical Derivation
-----------------------

### Objective Function

In SAC, the objective is to maximize the expected cumulative reward while incorporating an entropy term to encourage exploration:

.. math::

   J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t \big( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \big) \right],

where:
- :math:`\mathcal{H}(\pi(\cdot|s_t)) = -\sum_{a} \pi(a|s_t) \log \pi(a|s_t)` is the entropy of the policy.
- :math:`\alpha` is a temperature parameter controlling the trade-off between reward and entropy.

The inclusion of the entropy term allows SAC to balance exploitation and exploration effectively.

### Soft Q-Function and Soft Value Function

The **soft Q-function** and **soft value function** are defined as follows:

1. **Soft Q-Function**:
   .. math::

      Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P} \big[ V^{\text{soft}}(s') \big],

   where the soft value function :math:`V^{\text{soft}}(s)` is given by:
   .. math::

      V^{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} \big[ Q^{\text{soft}}(s, a) - \alpha \log \pi(a|s) \big].

2. **Policy Objective**:
   The policy is updated to minimize the KL-divergence between the policy distribution and the exponentiated soft Q-function:
   .. math::

      \pi^*(a|s) \propto \exp \left( \frac{1}{\alpha} Q^{\text{soft}}(s, a) \right).

### Bellman Backup for SAC

The SAC algorithm uses the following soft Bellman equation to iteratively update the Q-function:

.. math::

   Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P} \big[ \mathbb{E}_{a' \sim \pi} \big[ Q^{\text{soft}}(s', a') - \alpha \log \pi(a'|s') \big] \big].

The policy is updated to maximize the soft Q-function while minimizing the entropy regularization term.

Algorithmic flow
-----------------

1. **Initialize** the replay buffer, policy network, and Q-function network.
2. **Sample a batch** of transitions :math:`(s, a, r, s')` from the replay buffer.
3. **Update Q-Function**:
   .. math::

      \mathcal{L}_Q = \mathbb{E}_{(s, a, r, s')} \left[ \big( Q(s, a) - (r + \gamma V(s')) \big)^2 \right],

   where :math:`V(s')` is computed using the soft value function.
4. **Update Policy**:
   .. math::

      \mathcal{L}_\pi = \mathbb{E}_{s \sim D, a \sim \pi} \big[ \alpha \log \pi(a|s) - Q(s, a) \big].
5. **Adjust Temperature** (optional): Update :math:`\alpha` to ensure entropy matches a target value.
6. **Repeat** steps 2-5 until convergence.


References
-----------

- `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_
- `Soft Actor-Critic Algorithms and Applications <https://arxiv.org/abs/1812.05905>`_
- https://docs.cleanrl.dev/rl-algorithms/sac/
- https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95
- https://spinningup.openai.com/en/latest/algorithms/sac.html
