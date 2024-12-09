Proof of Policy Gradient Descent
=================================

The policy gradient method is a key technique in reinforcement learning, aiming to optimize the policy parameters by following the gradient of expected reward. Here, we present the derivation and proof of the policy gradient descent.

Let the objective be the maximization of the expected return:

.. math::
    J(\theta)=\sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s)=\sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) Q^\pi(s, a),

where :math:`d^\pi(s)` is the stationary distribution of Markov chain. For simplicity, all subsequent funcitons related to :math:`\pi` eliminate the subscript of :math:`\theta`.

Policy Gradient Theorem
--------------------------------------

.. important:: 
    :math:`\nabla_\theta J(\theta)=\sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \mid s)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\right]`.


Proof of Policy Gradient Theorem
--------------------------------------

Start with state-value function,

.. math::
    \begin{aligned}
    \nabla _{\theta}V^{\pi}(s)\\
    =&\nabla _{\theta}\left( \sum_{a\in \mathcal{A}}{\pi _{\theta}}(a\mid s)Q^{\pi}(s,a) \right)\\
    =&\sum_{a\in \mathcal{A}}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\pi _{\theta}(a\mid s)\nabla _{\theta}Q^{\pi}(s,a) \right)}\\
    =&\sum_{a\in \mathcal{A}}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\pi _{\theta}(a\mid s)\nabla _{\theta}\left( \mathcal{R} _{s}^{a}+\gamma\sum_{s^{\prime}}{P}\left( s^{\prime}\mid s,a \right) V^{\pi}\left( s^{\prime} \right) \right) \right)}\\
    =&\sum_{a\in \mathcal{A}}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\gamma\pi _{\theta}(a\mid s)\sum_{s^{\prime}}{P}\left( s^{\prime}\mid s,a \right) \nabla _{\theta}V^{\pi}\left( s^{\prime} \right) \right)}.\\
    \end{aligned}

Then, we can get,

.. important::
    
    .. math::
        \nabla_\theta V^\pi(s)=\sum_{a \in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a)+\gamma\pi_\theta(a \mid s) \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) \nabla_\theta V^\pi\left(s^{\prime}\right)\right).

Define that

.. math::
    \rho^\pi\left(s \rightarrow s^{\prime}, k=1\right)=\sum_a \pi_\theta(a \mid s) P\left(s^{\prime} \mid s, a\right).

Let :math:`\phi(s)=\sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a)`, we have,

.. math::
    \begin{aligned}
    	\nabla _{\theta}V^{\pi}(s)\\
    	=&\phi (s)+\gamma \sum_a{\pi _{\theta}}(a\mid s)\sum_{s^{\prime}}{P}\left( s^{\prime}\mid s,a \right) \nabla _{\theta}V^{\pi}\left( s^{\prime} \right)\\
    	=&\phi (s)+\gamma \sum_{s^{\prime}}{\sum_a{\pi _{\theta}}}(a\mid s)P\left( s^{\prime}\mid s,a \right) \nabla _{\theta}V^{\pi}\left( s^{\prime} \right)\\
    	=&\phi (s)+\gamma \sum_{s^{\prime}}{\rho ^{\pi}}\left( s\rightarrow s^{\prime},1 \right) \nabla _{\theta}V^{\pi}\left( s^{\prime} \right)\\
    	=&\phi (s)+\gamma \sum_{s^{\prime}}{\rho ^{\pi}}\left( s\rightarrow s^{\prime},1 \right) \left[ \phi \left( s^{\prime} \right) +\gamma \sum_{s^{\prime\prime}}{\rho ^{\pi}}\left( s^{\prime}\rightarrow s^{\prime\prime},1 \right) \nabla _{\theta}V^{\pi}\left( s^{\prime\prime} \right) \right]\\
    	=&\phi (s)+\gamma \sum_{s^{\prime}}{\rho ^{\pi}}\left( s\rightarrow s^{\prime},1 \right) \phi \left( s^{\prime} \right) +\gamma ^2\sum_{s^{\prime\prime}}{\rho ^{\pi}}\left( s\rightarrow s^{\prime\prime},2 \right) \nabla _{\theta}V^{\pi}\left( s^{\prime\prime} \right)\\
    	=&\dots\\
    	=&\sum_{x\in \mathcal{S}}{\sum_{k=0}^{\infty}{\gamma ^k\rho ^{\pi}}}(s\rightarrow x,k)\phi (x).\\
    \end{aligned}

By putting it into the objective function, we can obtain,

.. math::
    \begin{aligned}
    	\nabla _{\theta}J(\theta )&=\nabla _{\theta}\sum_{s_0}\rho_0\left( s_0 \right)V^{\pi}\left( s_0 \right)=\sum_{s_0}\sum_s{\sum_{k=0}^{\infty}{\gamma ^k\rho_0\left( s_0 \right)\rho ^{\pi}}}\left( s_0\rightarrow s,k \right) \phi (s)\\
    	&=\sum_s{\eta}(s)\phi (s)\\
    	&=\left( \sum_s{\eta}(s) \right) \sum_s{\frac{\eta (s)}{\sum_s{\eta}(s)}}\phi (s)\\
    	&\propto \sum_s{\frac{\eta (s)}{\sum_s{\eta}(s)}}\phi (s)\\
    	&=\sum_s{d^{\pi}}(s)\sum_a{\nabla _{\theta}}\pi _{\theta}(a\mid s)Q^{\pi}(s,a),\\
    \end{aligned}

where :math:`\eta(s)=\sum_{s_0}\sum_{k=0}^{\infty} \gamma ^k\rho_0\left( s_0 \right)\rho^\pi\left(s_0 \rightarrow s, k\right)` and :math:`\rho_0\left( s_0 \right)` denotes initial state probability distribution.

Finally,

.. math::
    \begin{aligned}
    \nabla_\theta J(\theta) & \propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \mid s) \\
    & =\sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} \\
    & =\mathbb{E}_\pi\left[Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \mid s)\right] \quad \textbf{Q.E.D.}
    \end{aligned}

The policy gradient method is a powerful approach for reinforcement learning, as it directly optimizes the policy by following the gradient of expected return. The derived proof provides the foundation for many policy-based methods in the field of reinforcement learning.
