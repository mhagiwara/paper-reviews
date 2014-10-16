Logic
=====

* Hoifung Poon and Pedro Domingos. Joint Inference in Information Extraction. AAAI 2007. http://www.aaai.org/Papers/AAAI/2007/AAAI07-145.pdf
  - Joint segmentation and entity resolution of citation matching (from CiteSeer and Cora)
  - Markov logic (Richardson & Domingos 2006) Alchemy package, MC-SAT algorithm ("slice sampling" MCMC) + voted perceptron algorithm
  - Citation matching: title, author, venue: MLN: token(t-token, i-pos, c-citation), InField, SameCitation
  - Segmentation: stickey HMM model with explicit punctuation modeling
  - SimilarTitle, SimilarVenue, SimilarAuthor "bandwidth" between segmentation and entity matching
  - Result: CiteSeer cluster recall ~96%, segmentation F1: ~95%

* Matthew Richardson and Pedro Domingos. Markov Logic Networks. http://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf
  - Markov network (MRF) - defines joint distribution over (X1, X2, ..., Xn) decomposed over a set of cliques (or log-linear form)
  - MLN - (formula, weight)+constants
  - One binary node for each possible grounding of each predict in MLN. One binary feature for each possible grounding of each formula in MLN.
  - Inference: What is the probability that formula F1 holds given that formula F2 does? -> Gibbs sampling with rejection

* Dan Garrette et al. Integrating Logical Representations with Probabilistic Information using Markov Logic. IWCS 2011 http://www.cs.utexas.edu/~dhg/papers/garrette_iwcs_2011.pdf
  - First order logical forms: high precision at the cost of low recall.
  - Implicativity and factivity, word meaning, and coreference.
  - MLN: weights for word mianing rules are computed from the distributional model -> injected into MLN. Rules -> given infinite weight (hard constraints)
  - Boxer output (nested DRS) -> flattened (introducing new labels to sub-DRSs)
  - Assign similarity-based probability compuated by (Erk and Pado 2010) to MLN rules (with context)
  - "bat" -> animal or artifact? Check whether it's close to "club" (instead of "artifact")
  - Evaluation based on small examples
