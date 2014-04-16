Machine Translation
===================

Bilingual Term Extraction
-------------------------

* Pascale Fung and Lo Yuan Yee. An IR Approach for Translating New Words from Nonparallel Comparable Texts, COLING 1998. http://acl.ldc.upenn.edu/P/P98/P98-1069.pdf
    - Assumption: words which appear in the context of a word and its translation should be similar to each other
    - Used word pairs from bilingual lexicon as seed words to "bridge" words in context
    - Used tf.idf and confidence (rank in a lexicon) weighting and cosine-like and Dice coefficient similarity measures for ranking translation candidates
    - High precision in the top ranked candidates. Similarity combination (Cosine times weighted Dice) performed the best


* Reinhard Rapp. Automatic identification of word translations from unrelated English and German corpora, ACL 1999. http://acl.ldc.upenn.edu/P/P99/P99-1067.pdf
    - Assumption ``If, in a text of one language two words A and B co-occur more often than expected by chance, then in a text of another language those words that are translations of A and B should also co-occur more frequently than expected.``
    - Create word vector of context words considering relative positions (window size of 3), used log likelihood ratio for weighting, and city-block (Manhattan distance) as the similarity measure
    - Experiment: finding English translations for an English input word, from two unrelated corpora of news articles
    - Achieved top1 accuracy of 72%. Ambiguity (e.g., "weiß" for "know" and "white") is an issue. 
