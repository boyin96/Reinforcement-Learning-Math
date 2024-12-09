Proof of Policy Gradient Descent
=================================

The policy gradient method is a key technique in reinforcement learning, aiming to optimize the policy parameters by following the gradient of expected reward. Here, we present the derivation and proof of the policy gradient descent.

Let the objective be the maximization of the expected return:

.. math::
    J(\theta)=\sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s)=\sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) Q^\pi(s, a)

where :math:`d^\pi(s)` is the stationary distribution of Markov chain. For simplicity, all subsequent expressions related to :math:`\pi` eliminate the subscript of :math:`\theta`.

Policy Gradient Theorem
--------------------------------------

.. important:: 
    \nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\right]

Proof of Policy Gradient Theorem
--------------------------------------

We want to find the gradient of \( J(\theta) \) with respect to the policy parameters \( \theta \). Applying the score function gradient:

.. math::
    \nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\pi_\theta}[R(\tau)]

Using the definition of expectation, we have:

.. math::
    \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(a_t | s_t) R_t\right]

This is a direct application of the score function method. Here, \( \nabla_\theta \log \pi_\theta(a_t | s_t) \) is the gradient of the log-probability of taking action \( a_t \) in state \( s_t \), and \( R_t \) is the total reward at time step \( t \).

### Step 2: The Gradient of the Log-Policy

The next step is to expand the gradient of the log-policy \( \log \pi_\theta(a_t | s_t) \). The policy \( \pi_\theta(a_t | s_t) \) represents the probability of choosing action \( a_t \) at state \( s_t \), which is parameterized by \( \theta \).

The gradient of the log-policy can be expressed as:

.. math::
    \nabla_\theta \log \pi_\theta(a_t | s_t) = \frac{\nabla_\theta \pi_\theta(a_t | s_t)}{\pi_\theta(a_t | s_t)}

This gradient shows how the log-probability of taking action \( a_t \) at state \( s_t \) changes with respect to the parameters \( \theta \).

### Step 3: Expectation and Sampling

Since the expectation is taken over the trajectory distribution, we can approximate it using Monte Carlo sampling. We can sample trajectories \( \tau \) from the policy \( \pi_\theta \) and compute the gradient for each trajectory:

.. math::
    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log \pi_\theta(a_t | s_t) R_t

where \( N \) is the number of sampled trajectories. This approximation is widely used in policy gradient methods such as REINFORCE.

### Step 4: The Policy Gradient Descent Update

Once we have the gradient \( \nabla_\theta J(\theta) \), we can update the policy parameters using gradient descent:

.. math::
    \theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)

where \( \alpha \) is the learning rate. This update rule guides the policy towards higher expected rewards by adjusting the parameters in the direction of the gradient.

### Conclusion

The policy gradient method is a powerful approach for reinforcement learning, as it directly optimizes the policy by following the gradient of expected return. The derived proof provides the foundation for many policy-based methods in the field of reinforcement learning.
