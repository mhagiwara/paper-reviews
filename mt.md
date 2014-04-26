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

Pivot Approaches
----------------
* Hua Wu and Haifeng Wang. Pivot Language Approach for Phrase-Based Statistical Machine Translation. ACL 2007. http://acl.ldc.upenn.edu/P/P07/P07-1108.pdf
    - Improve (or build from scratch) Lf-Le translation model using language pairs Lf-Lp and Lp-Le, where bilingual corpora exist
    - Calculate phrase translation prob. of Lf-Le from Lf-Lp and Lp-Le by summing over possible p. Lexical weight from induced alignment between Lf-Le and co-occurrence counts in phrases.
    - Interpolate phrase translation prob. and lexical weights. 
    - 22.13% BLEU improvement over standard model with 5K parallel sentences, when using two pivots (En+De). (Lf-En, En-Le, Lf-De, De-Le each has ~700K sents)

* Masao Utiyama and Hitoshi Isahara. A Comparison of Pivot Methods for Phrase-based Statistical Machine Translation. NAACL 2007. http://acl.ldc.upenn.edu/N/N07/N07-1061.pdf
    - Phrase translation: similar to (Wu and Wang 2007), convolution (summing over) common pivot phrase e, lexical translation prob. = defined by the average of maximum likelihood estimation (aligned counts)
    - Sentence translation: translate source to n pivot sentences and then translate each to n target sentences, choose the one with the highest score (based on MERT weights)
    - Experiment on Europarl (Es, De, Fr): Direct > PhraseTrans > SntTrans15 (with n = 15) ~ SntTrans1
    - Phrase table size of PhraseTrans x10 times bigger than Direct, recall of phrases is more important

Optimization
------------

* Franz Och. Minimum Error Rate Training in Statistical Machine Translation. ACL 2003. http://acl.ldc.upenn.edu/acl2003/main/pdfs/Och.pdf
    - Optimize weights of log-linear models so that it directly maximize translation quality criteria (WER, BLEU, etc.)
    - Powells algorithm combined with a grid-based line optimization method. Efficient update achieved on the optimization on a piecewise linear function merged over all sentences in the corpus
    - BLEU ~ +6 point improvement over MMI if trained on BLEU (Chinese to English standard phrase-based SMT system)
