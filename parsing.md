Parsing
=======

Constituency Parsing
--------------------

* Dan Klein and Chris Manning. Accurate Unlexicalized Parsing. ACL 2003 http://www.cs.berkeley.edu/~klein/papers/unlexicalized-parsing.pdf
  - Success of lexicalized PCFG models (e.g., Charniak 1997, Collins 1999)
  - Johnson (1998) unlexicalized PCFG could be improved by annotating each node by its parent category (e.g., subject NP is 8.7 times more likely than an object NP to expand as just a pronoun)
  - Collins (1999) hand engineerred subcategorizations e.g., differenciating base-NPs, sentences with empty subjects
  - Generalization of parent annotation and markovization - only the past v vertical ancestors matter, only the previous h horizontal ancestors matter -> best results when v = 3 and h ≤ 2
  - Annotation: UNARY-INTERNAL marks any nonterminal node which has only one child. UNARY-DT marks splits, TAG-PA marks all preterminals with their their parent category
  - Unlexicailzed grammar - e.g., functional words and content words - PP[to] -> OK, NP[stocks] -> NG
  - Annotations already in the treebank - PP-LOC or ADVP-TMP have negative utility, TMP-NP positive
  - Head annotation e.g., SPLIT-VP annotates all VP nodes with their head tag.

* Mi Haitao and Liang Huang. Shift-Reduce Constituency Parsing with Dynamic Programming and POS Tag Lattice. NAACL 2015 http://acl.cs.qc.edu/~lhuang/papers/dp-constituency.pdf
    - Extension of (Huang and Sagae 2010) to CFG constituency parsing
    - Introduce odd step: choose from un (apply unary rule) or st (do nothing). At even steps, choose among shift/reduceR/reduceL.
    - Merge equivalent states:
    - Tag lattice: for shift action, split states based on the PoS tag of n - k-th token (k is context length)
    - Experiments: English (PTB) and Chinese (CTB), trained using 'max-violation perceptron', evaluated by labeled precision (LP) and labeled recall (LR) and bracketing F1. Faster convergence and better performance than non-DP parsers. Comparable to Wang (2014)'s joint PoS+parsing model in bracketing F1.
 
