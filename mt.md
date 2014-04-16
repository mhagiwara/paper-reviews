Machine Translation
===================

Bilingual Term Extraction
-------------------------

* [Fung and Yee COLING 1998] An IR Approach for Translating New Words from Nonparallel Comparable Texts http://acl.ldc.upenn.edu/P/P98/P98-1069.pdf
** Assumption: words which appear in the context of a word and its translation should be similar to each other
** Used word pairs from bilingual lexicon as seed words to "bridge" words in context
** Used tf.idf and confidence (rank in a lexicon) weighting and cosine-like and Dice coefficient similarity measures for ranking translation candidates
** High precision in the top ranked candidates. Similarity combination (Cosine times weighted Dice) performed the best

