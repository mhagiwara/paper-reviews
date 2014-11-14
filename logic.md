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

* Xin Luna Dong et al. Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion. KDD 2014. https://www.cs.cmu.edu/~nlao/publication/2014.kdd.pdf
  - 71% of people in Freebase have no known place of birth.
  - Knowledge vault (KV) - separates facts about the world from their lexical representation.
  - Extract facts from a large variety of sources of Web data, including free text, HTML DOM trees (this one has the highest AUC score), HTML Web tables, and human annotations of Web pages. -> classifier per predicate
  - Evaluation: random separation of (s, p, o) (edge) from a graph into train/test, and how well KV can predict missing edges.
  - Graph-based priors: path ranking algorithm (use random-walk paths to predict prediates) and neural network model (Socher et al. NIPS 2012)
  - Future work: modelling mutual exclusion between facts (but simple exclusion doesn't work e.g., Honolulu -> Hawaii), numerical correlation (DoB of a person and DoB of children)

* Shangpu Jiang et al. Learning to Refine an Automatically Extracted Knowledge Base using Markov Logic. ICDM 2012 http://ix.cs.uoregon.edu/~lowd/icdm12jiang.pdf
  - Ontological constraints from the original IE system serve as hard constraints (e.g., subsumption among categories, mutually exclusive categories and relations, domain range)
  - IE: simple logistic regression model for each category or relation to predict the truth (features: co-occurrence with patterns)
  - Use of logistic regression to approximate weight learning (MLN is equivalent to logistic regression without hard constraint)
  - MLN forms clusters of "domains"
  - Evaluation using AUC, P/R/F on randomly sampled 4,511 relations -> Models with patterns, human label training, and ontological constraints performed better
