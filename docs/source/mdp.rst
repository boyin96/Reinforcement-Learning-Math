Markov Decision Processes
===========================

Introduction
------------------
Markov Decision Processes (MDPs) provide a mathematical framework for modeling decision-making in environments where outcomes are partly random and partly under the control of a decision maker. This framework is fundamental in fields such as reinforcement learning, operations research, and economics. 

This document explores key concepts of MDPs, starting with Markov Processes (MP), then Markov Reward Processes (MRP), and finally MDPs. Each section provides definitions and the corresponding Bellman equations. References are listed at the end for further reading.

Markov Processes
------------------
**Definition**

A Markov Process (MP) is a stochastic process characterized by the Markov property, which states that the future state depends only on the current state and not on the sequence of states that preceded it. Formally:

.. math::
   P(s_{t+1} \mid s_t, s_{t-1}, \ldots, s_0) = P(s_{t+1} \mid s_t)

Where:

- :math:`s_t` is the state at time :math:`t`.
- :math:`P` is the transition probability.

**Bellman Equation**

For a Markov Process, the Bellman equation describes the recursive relationship of state transition probabilities:

.. math::
   P(s_{t+1}) = \sum_{s_t} P(s_{t+1} \mid s_t) P(s_t)

Markov Reward Processes
--------------------------------
**Definition**

A Markov Reward Process (MRP) extends an MP by associating rewards with state transitions. An MRP is defined by the tuple :math:`(\mathcal{S}, P, R, \gamma)` where:

- :math:`\mathcal{S}` is a finite set of states.
- :math:`P` is the state transition probability matrix.
- :math:`R(s)` is the expected reward for state :math:`s`.
- :math:`\gamma \in [0, 1]` is the discount factor.

**Bellman Equation**

The Bellman equation for the value function :math:`V(s)` is:

.. math::
   V(s) = R(s) + \gamma \sum_{s'} P(s' \mid s) V(s')

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
1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.

