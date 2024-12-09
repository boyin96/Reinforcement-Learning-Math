===============================
Policy Gradient Theorem in DRL
===============================

In this document, we will provide a detailed derivation of the **Policy Gradient Theorem** used in **Deep Reinforcement Learning (DRL)**. We begin by reviewing the reinforcement learning setup and the goal of policy optimization.

1. **Reinforcement Learning Setup**
   ================================

   In reinforcement learning, an agent interacts with an environment in discrete time steps. The environment is modeled as a **Markov Decision Process (MDP)** defined by the tuple:

   .. math::

      \mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, \mathcal{R}, \gamma \rangle

   where:
   - \( \mathcal{S} \) is the set of states,
   - \( \mathcal{A} \) is the set of actions,
   - \( P(s'|s, a) \) is the state transition probability,
   - \( \mathcal{R}(s, a) \) is the reward function,
   - \( \gamma \in [0, 1] \) is the discount factor.

   An agent follows a policy \( \pi(a|s) \), which is a distribution over actions given states. The goal is to optimize this policy in order to maximize the expected return, which is the sum of discounted rewards.

2. **Expected Return (Objective Function)**
   =========================================

   The objective is to maximize the expected return. The return from time step \( t \) is defined as:

   .. math::

      G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}

   where \( R_{t+k+1} \) represents the reward at time step \( t+k+1 \), and \( \gamma \) is the discount factor.

   The **state-value function** \( V^\pi(s) \) and **action-value function** \( Q^\pi(s, a) \) represent the expected returns from state \( s \) under policy \( \pi \), and from state \( s \) and action \( a \), respectively. 

   The goal is to find a policy that maximizes the expected return:

   .. math::

      J(\pi) = \mathbb{E}_\pi \left[ G_t \right]

   where \( J(\pi) \) is the expected return under policy \( \pi \).

3. **The Policy Gradient Theorem**
   ===============================

   To optimize the policy, we seek the gradient of \( J(\pi) \) with respect to the policy parameters. The key idea is to compute the gradient of the expected return by differentiating under the expectation.

   The objective function \( J(\pi) \) is:

   .. math::

      J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \right]

   Now, the gradient of \( J(\pi) \) with respect to the policy \( \pi \) is:

   .. math::

      \nabla_\theta J(\pi_\theta) = \nabla_\theta \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \right]

   Using the **log-derivative trick**, we can express the gradient as:

   .. math::

      \nabla_\theta J(\pi_\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right]

   This is the **policy gradient** formula, where:

   - \( \log \pi_\theta(a_t | s_t) \) is the log of the probability of taking action \( a_t \) in state \( s_t \) under policy \( \pi_\theta \),
   - \( \nabla_\theta \log \pi_\theta(a_t | s_t) \) is the gradient of the log-probability with respect to the policy parameters \( \theta \).

4. **Estimating the Gradient: Monte Carlo and Temporal Difference**
   ===============================================================

   To estimate the gradient in practice, we can use two main methods:
   
   - **Monte Carlo estimation**: This method uses the complete return \( G_t \) from each trajectory to estimate the gradient.
   - **Temporal Difference (TD) estimation**: Instead of using the full return, TD methods estimate the return based on a bootstrapped estimate of the value function.

   The **Monte Carlo estimate** of the gradient is:

   .. math::

      \hat{\nabla}_\theta J(\pi_\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \gamma^t R_{t+1} \nabla_\theta \log \pi_\theta(a_t | s_t)

   The **TD estimate** of the gradient, where we use a value function approximation \( V^\pi(s_t) \), is:

   .. math::

      \hat{\nabla}_\theta J(\pi_\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \delta_t \nabla_\theta \log \pi_\theta(a_t | s_t)

   where \( \delta_t \) is the **TD error**, defined as:

   .. math::

      \delta_t = R_{t+1} + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)

5. **Conclusion**
   ==============

   The policy gradient theorem provides a way to compute the gradient of the expected return with respect to the parameters of a policy. This allows us to optimize the policy using gradient ascent methods. Both Monte Carlo and Temporal Difference methods can be used to estimate the gradient in practice, with Temporal Difference often being more efficient in real-world applications.

   The key takeaway is that policy gradient methods allow us to directly optimize the policy by using the gradient of the objective function, making them a powerful tool in deep reinforcement learning.

