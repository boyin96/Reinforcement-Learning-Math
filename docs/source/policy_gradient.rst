Policy Gradient
=================================

The policy gradient method is a key technique in reinforcement learning, aiming to optimize the policy parameters by following the gradient of expected reward. Here, we present the derivation and proof of the policy gradient.

Let the objective be the maximization of the expected return:

.. math::
    J(\theta)=\sum_{s } \rho^\pi(s) V^\pi(s)=\sum_{s} \rho^\pi(s) \sum_{a} \pi_\theta(a \mid s) Q^\pi(s, a),

where :math:`\rho^\pi(s)` is the discounted-aggregate state-visitation measure (stationary distribution with initial state probability) of Markov chain. For simplicity, all subsequent funcitons related to :math:`\pi` eliminate the subscript of :math:`\theta`.

Policy Gradient Theorem
--------------------------------------

.. important:: 
    :math:`\nabla_\theta J(\theta)=\sum_{s } \rho^\pi(s) \sum_{a } \nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\right]`.


Proof of Policy Gradient Theorem 1
--------------------------------------

Start with state-value function,

.. math::
    \begin{aligned}
    \nabla _{\theta}V^{\pi}(s)\\
    =&\nabla _{\theta}\left( \sum_{a}{\pi _{\theta}}(a\mid s)Q^{\pi}(s,a) \right)\\
    =&\sum_{a}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\pi _{\theta}(a\mid s)\nabla _{\theta}Q^{\pi}(s,a) \right)}\\
    =&\sum_{a}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\pi _{\theta}(a\mid s)\nabla _{\theta}\left( \mathcal{R} _{s}^{a}+\gamma\sum_{s^{\prime}}{P}\left( s^{\prime}\mid s,a \right) V^{\pi}\left( s^{\prime} \right) \right) \right)}\\
    =&\sum_{a}{\left( \nabla _{\theta}\pi _{\theta}(a\mid s)Q^{\pi}(s,a)+\gamma\pi _{\theta}(a\mid s)\sum_{s^{\prime}}{P}\left( s^{\prime}\mid s,a \right) \nabla _{\theta}V^{\pi}\left( s^{\prime} \right) \right)}.\\
    \end{aligned}

Then, we can get,

.. important::
    
    .. math::
        \nabla_\theta V^\pi(s)=\sum_{a}\left(\nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a)+\gamma\pi_\theta(a \mid s) \sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) \nabla_\theta V^\pi\left(s^{\prime}\right)\right).

Define that

.. math::
    \rho^\pi\left(s \rightarrow s^{\prime}, k=1\right)=\sum_a \pi_\theta(a \mid s) P\left(s^{\prime} \mid s, a\right).

Let :math:`\phi(s)=\sum_{a} \nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a)`, we have,

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
    	\nabla _{\theta}J(\theta )&=\sum_{s_0}\rho_0\left( s_0 \right)\nabla _{\theta}V^{\pi}\left( s_0 \right)=\sum_{s_0}\sum_s{\sum_{k=0}^{\infty}{\gamma ^k\rho_0\left( s_0 \right)\rho ^{\pi}}}\left( s_0\rightarrow s,k \right) \phi (s)\\
    	&=\sum_s\rho^\pi(s)\phi (s)\\
    	&=\sum_s{\rho^{\pi}}(s)\sum_a{\nabla _{\theta}}\pi _{\theta}(a\mid s)Q^{\pi}(s,a),\\
    \end{aligned}

where :math:`\rho^\pi(s)=\sum_{s_0}\sum_{k=0}^{\infty} \gamma ^k\rho_0\left( s_0 \right)\rho^\pi\left(s_0 \rightarrow s, k\right)` and :math:`\rho_0\left( s \right)` denotes initial state probability distribution.

Finally,

.. important::
    
    .. math::
        \begin{aligned}
        \nabla_\theta J(\theta) & = \sum_{s} \rho^\pi(s) \sum_{a} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \mid s) \\
        & =\sum_{s} \rho^\pi(s) \sum_{a} \pi_\theta(a \mid s) Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} \\
        & =\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s)Q^{\pi_\theta}(s, a) \right] \quad \textbf{Q.E.D.}
        \end{aligned}

The policy gradient method is a powerful approach for reinforcement learning, as it directly optimizes the policy by following the gradient of expected return. The derived proof provides the foundation for many policy-based methods in the field of reinforcement learning.

Proof of Policy Gradient Theorem 2
--------------------------------------

Start with another type of objective function,

.. math::
    J(\theta)=E_{\tau \sim \pi}[R(\tau)] = \sum_{\tau} P(\tau ; \theta) R(\tau),

where :math:`R(\tau)` is a return from an arbitrary trajectory :math:`\tau`, and :math:`P(\tau ; \theta)` is the probability of each possible trajectory which depends on the :math:`\theta`.

Then, we have,

..math::
    \nabla_\theta J(\theta)=\nabla_\theta \sum_\tau P(\tau ; \theta) R(\tau) =\sum_\tau \nabla_\theta(P(\tau ; \theta) R(\tau))=\sum_\tau \nabla_\theta P(\tau ; \theta) R(\tau)，

since :math:`R(\tau)` is not dependent on :math:`\theta`. By using the derivative log trick, we can get

..math::
    \nabla_\theta J(\theta)=\sum_\tau P(\tau ; \theta) \nabla_\theta \log P(\tau ; \theta) R(\tau).

Thanks for this new formula, we can estimate the gradient using trajectory samples to remove the probability term :math:`P(\tau ; \theta)`, i.e.,

..math::
    \nabla_\theta J(\theta)= m^1 \sum_{i=1}^m \nabla_\theta \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right).

Now we need to simplify the term :math:`P\left(\tau^{(i)} ; \theta\right)`. According to the MDP, the trajectory can be represented as 

..math:: 
    P(\tau ; \theta)=\rho^\pi(s_0)\prod_{t=0} P\left(s_{t+1} \mid s_t, a_t\right) \pi_\theta\left(a_t \mid s_t\right).

Then, we have 

..math::
\begin{aligned}
    \nabla_\theta \log P\left(\tau^{(i)} ; \theta\right)=&
    \nabla_\theta \log \left[\mu\left(s_0\right) \prod_{t=0}^H P\left(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}\right) \pi_\theta\left(a_t^{(i)} \mid  s_t^{(i)}\right)\right] = &
\nabla_\theta\left[\log \mu\left(s_0\right)+\sum_{t=0}^H \log P\left(s_{t+1}^{(i)} \mid s_t^{(i)} a_t^{(i)}\right)+\sum_{t=0}^H \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)\right]
\end{aligned}



References
--------------------------------------

- https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- https://web.stanford.edu/class/cme241/lecture_slides/PolicyGradient.pdf
- https://huggingface.co/learn/deep-rl-course/unit4/pg-theorem
