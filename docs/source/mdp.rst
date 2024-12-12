Markov Decision Processes
===========================

Introduction
------------------
Markov Decision Processes (MDPs) provide a mathematical framework for modeling decision-making in environments where outcomes are partly random and partly under the control of a decision maker. This framework is fundamental in fields such as reinforcement learning, operations research, and economics. 

This document explores key concepts of MDPs, starting with Markov Processes (MP), then Markov Reward Processes (MRP), and finally MDPs. Each section provides definitions and the corresponding Bellman equations.

Markov Processes
------------------
A Markov Process (MP) is a stochastic process characterized by the Markov property, which states that the future state depends only on the current state and not on the sequence of states that preceded it. A state :math:`S_t` is Markov (or with Markov property) if and only if

.. math::
   P\left[S_{t+1} \mid S_t\right]=P\left[S_{t+1} \mid S_1, \ldots, S_t\right]

Definition
^^^^^^^^^^^^^

.. note::
   A Markov Process (or Markov Chain) is a tuple :math:`\langle\mathcal{S}, P\rangle`
  
   - :math:`\mathcal{S}` is a (finite) set of states.
   - :math:`P` is a state transition probability matrix, :math:`P=P\left[S_{t+1}=s^{\prime} \mid S_t=s\right]`

Bellman Equation
^^^^^^^^^^^^^^^^^^

.. tip::
   For a Markov Process, the Bellman equation describes the recursive relationship of state transition probabilities,

   .. math::
      P(s^{\prime}) = \sum_{s} P(s^{\prime} \mid s) P(s)

Markov Reward Processes
--------------------------------
A Markov Reward Process (MRP) extends an MP by associating rewards (values) with state transitions.

Definition
^^^^^^^^^^^^^

.. note::
   An MRP is a tuple :math:`\langle\mathcal{S}, P,\mathcal{R},\gamma\rangle`
  
   - :math:`\mathcal{S}` is a (finite) set of states.
   - :math:`P` is a state transition probability matrix, :math:`P=P\left[S_{t+1}=s^{\prime} \mid S_t=s\right]`
   - :math:`\mathcal{R}_s` is a reward function.
   - :math:`\gamma` is a discount factor.

Bellman Equation
^^^^^^^^^^^^^^^^^^
The state value function :math:`V(s)` of an MRP is the expected return starting from state :math:`s`,

.. math::
   V(s)=\mathbb{E}\left[G_t \mid S_t=s\right]

The Bellman equation for the value function :math:`V(s)` is,

.. math::
   \begin{aligned}
   V(s) & =\mathbb{E}\left[G_t \mid S_t=s\right] \\
   & =\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\ldots \mid S_t=s\right] \\
   & =\mathbb{E}\left[R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\ldots\right) \mid S_t=s\right] \\
   & =\mathbb{E}\left[R_{t+1}+\gamma G_{t+1} \mid S_t=s\right] \\
   & =\mathbb{E}\left[R_{t+1}+\gamma V\left(S_{t+1}\right) \mid S_t=s\right]
   \end{aligned}

.. tip::

   .. math::
      V(s)=\mathcal{R}_s+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)   

Markov Decision Processes
-------------------------------
**Definition**

A Markov Decision Process (MDP) introduces decision-making into an MRP. It is defined by the tuple :math:`(\mathcal{S}, \mathcal{A}, P, R, \gamma)` where:

- :math:`\mathcal{S}` is a finite set of states.
- :math:`\mathcal{A}` is a finite set of actions.
- :math:`P(s' \mid s, a)` is the transition probability given action :math:`a`.
- :math:`R(s, a)` is the expected reward for taking action :math:`a` in state :math:`s`.
- :math:`\gamma \in [0, 1]` is the discount factor.

**Bellman Equation**

The Bellman equation for the optimal value function :math:`V^*(s)` is:

.. math::
   V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]

**Bellman Optimality Equation**

For the optimal action-value function :math:`Q^*(s, a)`:

.. math::
   Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')

References
----------------

- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- https://www.davidsilver.uk/teaching/

