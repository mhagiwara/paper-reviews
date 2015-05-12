Parsing
=======

Constituency Parsing
--------------------

* Michael Collins and Brian Roark. Incremental Parsing with the Perceptron Algorithm. ACL 2004. http://www.aclweb.org/anthology/P04-1015
  - Generic linear models for NLP: Given a training instance, decodes the training instance given the current model (features and weights), update the parameters when errors are made.
  - Perceptron guarantee: if training samples are separable, the parameters converge within a finite number of iterations
  - Grammar G include allowable chains (path from a word w to starting symbol S) and allowable triples (allowable non-terminals Z after Y under parent X)
  - Caching past hypothesis and reuse them to update weights (for efficiency), early update (if partial analysis of gold standard doesn't exist in the hypothesis set, exit the decoding and update parameters immediately)
  - Experiment: 1.5 points lower than generative models, but competitive when including punctuation features in Penn Treebank

* Dan Klein and Chris Manning. Accurate Unlexicalized Parsing. ACL 2003 http://www.cs.berkeley.edu/~klein/papers/unlexicalized-parsing.pdf
  - Success of lexicalized PCFG models (e.g., Charniak 1997, Collins 1999)
  - Johnson (1998) unlexicalized PCFG could be improved by annotating each node by its parent category (e.g., subject NP is 8.7 times more likely than an object NP to expand as just a pronoun)
  - Collins (1999) hand engineerred subcategorizations e.g., differenciating base-NPs, sentences with empty subjects
  - Generalization of parent annotation and markovization - only the past v vertical ancestors matter, only the previous h horizontal ancestors matter -> best results when v = 3 and h ≤ 2
  - Annotation: UNARY-INTERNAL marks any nonterminal node which has only one child. UNARY-DT marks splits, TAG-PA marks all preterminals with their their parent category
  - Unlexicailzed grammar - e.g., functional words and content words - PP[to] -> OK, NP[stocks] -> NG
  - Annotations already in the treebank - PP-LOC or ADVP-TMP have negative utility, TMP-NP positive
  - Head annotation e.g., SPLIT-VP annotates all VP nodes with their head tag.

* Liang Huang and Kenji Sagae. Dynamic Programming for Linear-Time Incremental Parsing. ACL 2010 https://www.aclweb.org/anthology/P/P10/P10-1110.pdf
  - Greedy search e.g., shift-reduce parsers -> fast (linear) and psychologically motivated, but severe search errors
  - Can output a forest encoding exponentially many trees
  - Shift-reduce dependency parsing: shift, reduce(left) and reduce(right)
  - Graph-structured stack and deduction: two states are equivalent if both agree on features and queue top. Maintain explicit predictor states which work as backpointers in graphs
  - Experiments: DP with beam=16 achieves the same quality as non-DB with beam=64, while 5x faster

* Mi Haitao and Liang Huang. Shift-Reduce Constituency Parsing with Dynamic Programming and POS Tag Lattice. NAACL 2015 http://acl.cs.qc.edu/~lhuang/papers/dp-constituency.pdf
    - Extension of (Huang and Sagae 2010) to CFG constituency parsing
    - Introduce odd step: choose from un (apply unary rule) or st (do nothing). At even steps, choose among shift/reduceR/reduceL.
    - Merge equivalent states:
    - Tag lattice: for shift action, split states based on the PoS tag of n - k-th token (k is context length)
    - Experiments: English (PTB) and Chinese (CTB), trained using 'max-violation perceptron', evaluated by labeled precision (LP) and labeled recall (LR) and bracketing F1. Faster convergence and better performance than non-DP parsers. Comparable to Wang (2014)'s joint PoS+parsing model in bracketing F1.
