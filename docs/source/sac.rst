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

.. important::

   .. math::
      J(\theta)=\sum_{t=0}^T \mathbb{E}_{\pi_\theta}\left[\mathcal{R}\left(\mathbf{s}_t, \mathbf{a}_t\right)+\alpha \mathcal{H}\left(\pi_\theta\left(\cdot \mid \mathbf{s}_t\right)\right)\right],

where :math:`\alpha` controls how important the entropy term is, known as temperature parameter. The inclusion of the entropy term allows SAC to balance exploitation and exploration effectively.

Soft Q-Function and Soft Value Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The soft Q-function and soft value function are defined as follows:

1. **Soft Q-Function**:
   
   .. math::

      Q\left(s_t, a_t\right)=\mathcal{R}\left(s_t, a_t\right)+\gamma \mathbb{E}_{\pi_\theta}\left[V\left(s_{t+1}\right)\right],

   where the soft value function is given by:
   
   .. math::

     V\left(s_t\right)=\mathbb{E}_{\pi_\theta}\left[Q\left(s_t, a_t\right)-\alpha \log \pi_\theta\left(a_t \mid s_t\right)\right].

   Thus,

   .. math::

      Q\left(s_t, a_t\right)=\mathcal{R}\left(s_t, a_t\right)+\gamma \mathbb{E}_{\pi_\theta}\left[Q\left(s_{t+1}, a_{t+1}\right)-\alpha \log \pi_\theta\left(a_{t+1} \mid s_{t+1}\right)\right]

2. **Soft Q-function Update**: The soft Q-function parameters can be trained to minimize the soft Bellman residual,

   .. important::
      
      .. math::
         
         J_Q(w)=\mathbb{E}_{\mathcal{D}}\left[\frac{1}{2}\left(Q_w\left(s_t, a_t\right)-\left(\mathcal{R}\left(s_t, a_t\right)+\gamma \mathbb{E}_{\pi_\theta}\left[V_{\bar{\psi}}\left(s_{t+1}\right)\right]\right)\right)^2\right],

with gradient,

.. math::

   \begin{aligned}
	\nabla _wJ_Q(w)=&\nabla _wQ_w\left( s_t,a_t \right)\\
	&\left( Q_w\left( s_t,a_t \right) -\left( \mathcal{R} \left( s_t,a_t \right) +\gamma \left( Q_{\bar{w}}\left( s_{t+1},a_{t+1} \right) -\alpha \log \left( \pi _{\theta}\left( a_{t+1}\mid s_{t+1} \right) \right) \right) \right) \right) ,
	\end{aligned}

where :math:`\bar{\psi}` and :math:`\bar{w}` are the target state-value function and action-value function which are the exponential moving average.

3. **Policy Objective**: The policy is updated to minimize the KL-divergence between the policy distribution and the exponentiated soft Q-function:

   .. important::
      
      .. math::
	\begin{aligned}
	\pi_{\text {new }} & =\arg \min _{\pi^{\prime} \in \Pi} D_{\mathrm{KL}}\left(\pi^{\prime}\left(. \mid s_t\right) \| \frac{\exp \left(Q^{\pi_{\text {old }}}\left(s_t, .\right)\right)}{Z^{\pi_{\text {old }}}\left(s_t\right)}\right) \\
	& =\arg \min _{\pi^{\prime} \in \Pi} D_{\mathrm{KL}}\left(\pi^{\prime}\left(. \mid s_t\right) \| \exp \left(Q^{\pi_{\text {old }}}\left(s_t, .\right)-\log Z^{\pi_{\text {old }}}\left(s_t\right)\right)\right)
	\end{aligned}
         
with gradient,	

.. math::

	\begin{aligned}
	J_\pi(\theta) & =\nabla_\theta D_{\mathrm{KL}}\left(\pi_\theta\left(\cdot \mid s_t\right) \| \exp \left(Q_w\left(s_t, .\right)-\log Z_w\left(s_t\right)\right)\right) \\
	& =\mathbb{E}_{a_t \sim \pi}\left[-\log \left(\frac{\exp \left(Q_w\left(s_t, a_t\right)-\log Z_w\left(s_t\right)\right)}{\pi_\theta\left(a_t \mid s_t\right)}\right)\right] \\
	& =\mathbb{E}_{a_t \sim \pi}\left[\log \pi_\theta\left(a_t \mid s_t\right)-Q_w\left(s_t, a_t\right)+\log Z_w\left(s_t\right)\right],
	\end{aligned}	

where :math:`Z^{\pi_{\text {old }}}` is the partition function to normalize the distribution and :math:`\Pi` denotes a set of policies that can be readily tractable.

Algorithmic flow
^^^^^^^^^^^^^^^^^^^

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
