Soft Actor-Critic (SAC) Algorithm
===================================

Introduction
------------

Soft Actor-Critic (SAC) is an advanced reinforcement learning algorithm that integrates the principle of maximum entropy into continuous control tasks. It is designed to balance the **exploitation of high-reward actions** with the **exploration of diverse strategies** by maximizing both the cumulative reward and the entropy of the policy.

Key features of SAC include:

- **Maximum entropy framework**: Encourages diverse action selection.
- **Stochastic policies**: Enables better exploration compared to deterministic approaches.
- **Off-policy learning**: Utilizes a replay buffer for data efficiency.
- **Twin soft Q-network**: Uses two separate Q-networks to mitigate overestimation bias.
- **Automatic temperature adjustment**: Adaptively tunes the temperature parameter
- **Continuous action space via reparameterization**: Employs the reparameterization trick to optimize stochastic policies in continuous action spaces.


Theoretical Derivation
-----------------------
For ease of proof, consider a finite-horizon undiscounted return MDP, ignoring :math:`\gamma`, the objective function of SAC is as follows,

.. math::
   J(\theta)=\sum_{t=0}^T \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[\mathcal{R}\left(\mathbf{s}_t, \mathbf{a}_t\right)+\alpha \mathcal{H}\left(\pi_\theta\left(\cdot \mid \mathbf{s}_t\right)\right)\right],

where :math:`\alpha` controls how important the entropy term is, known as temperature parameter.

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

ellman Backup for SAC

The SAC algorithm uses the following soft Bellman equation to iteratively update the Q-function:

.. math::

   Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P} \big[ \mathbb{E}_{a' \sim \pi} \big[ Q^{\text{soft}}(s', a') - \alpha \log \pi(a'|s') \big] \big].

The policy is updated to maximize the soft Q-function while minimizing the entropy regularization term.

Algorithmic flow
-----------------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Soft Actor-Critic}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \STATE Observe state $s$ and select action $a \sim \pi_{\theta}(\cdot|s)$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets for the Q functions:
                    \begin{align*}
                        y (r,s',d) &= r + \gamma (1-d) \left(\min_{i=1,2} Q_{\phi_{\text{targ}, i}} (s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s')\right), && \tilde{a}' \sim \pi_{\theta}(\cdot|s')
                    \end{align*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big(\min_{i=1,2} Q_{\phi_i}(s, \tilde{a}_{\theta}(s)) - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) \right| s\right) \Big),
                    \end{equation*}
                    where $\tilde{a}_{\theta}(s)$ is a sample from $\pi_{\theta}(\cdot|s)$ which is differentiable wrt $\theta$ via the reparametrization trick.
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}

References
-----------

- `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_
- `Soft Actor-Critic Algorithms and Applications <https://arxiv.org/abs/1812.05905>`_
- https://docs.cleanrl.dev/rl-algorithms/sac/
- https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95
- https://spinningup.openai.com/en/latest/algorithms/sac.html
