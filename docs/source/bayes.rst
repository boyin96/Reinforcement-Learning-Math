Bayes Theorem
=============

Introduction
------------
Bayes theorem, named after the Reverend Thomas Bayes, is a fundamental concept in probability theory and statistics. It provides a mathematical framework for updating the probability of a hypothesis based on new evidence. This theorem is widely used in various fields, such as machine learning, medical diagnosis, and decision-making under uncertainty.

Mathematical Formulation
------------------------
Bayes theorem is expressed as:

.. math::

   P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}

where:

- :math:`P(H|E)` is the posterior probability: the probability of the hypothesis :math:`H` given the evidence :math:`E`.
- :math:`P(E|H)` is the likelihood: the probability of observing the evidence :math:`E` given that :math:`H` is true.
- :math:`P(H)` is the prior probability: the initial probability of the hypothesis :math:`H` before observing the evidence.
- :math:`P(E)` is the marginal probability: the total probability of the evidence :math:`E` under all possible hypotheses.

Proof
-----
Bayes theorem can be derived using the definition of conditional probability. For two events :math:`H` and :math:`E`:

.. math::

   P(H \cap E) = P(H|E) \cdot P(E) = P(E|H) \cdot P(H)

Rearranging this equality gives:

.. math::

   P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}

Thus, Bayes theorem is established.

Intuitive Explanation
---------------------
Bayes theorem allows us to update our belief about a hypothesis when new evidence is introduced. Consider the following analogy:

- Hypothesis (:math:`H`): A bag contains mostly red balls.
- Evidence (:math:`E`): You draw a red ball from the bag.

Initially, you may assign a prior probability to the hypothesis based on your knowledge or assumptions. When you observe the evidence (a red ball), you use the likelihood (:math:`P(E|H)`) to update your belief about the hypothesis. The posterior probability (:math:`P(H|E)`) reflects this updated belief.

Key Insights
------------
1. **Dynamic Updating:** Bayes theorem provides a dynamic way to revise probabilities as new evidence is introduced.
2. **Interplay of Prior and Evidence:** The posterior probability depends on both the prior and the likelihood. Strong prior beliefs can dominate unless the evidence is overwhelming.
3. **Normalization:** The marginal probability (:math:`P(E)`) ensures that the posterior probabilities across all hypotheses sum to 1, maintaining a coherent probability distribution.
4. **Applications:** From spam email detection to medical testing, Bayes theorem underpins many probabilistic models and inference techniques.

Conclusion
----------
Bayes theorem is a powerful tool for reasoning under uncertainty. By combining prior knowledge with observed evidence, it allows for a rational and systematic update of beliefs. Its significance extends beyond theoretical probability, impacting practical applications across diverse domains.

