Proof of Policy Gradient Descent
=================================

The policy gradient method is a key technique in reinforcement learning, aiming to optimize the policy parameters by following the gradient of expected reward. Here, we present the derivation and proof of the policy gradient descent.

Let the objective be the maximization of the expected return:

.. math::
    J(\theta)=\sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s)=\sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) Q^\pi(s, a)

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




The policy gradient method is a powerful approach for reinforcement learning, as it directly optimizes the policy by following the gradient of expected return. The derived proof provides the foundation for many policy-based methods in the field of reinforcement learning.
