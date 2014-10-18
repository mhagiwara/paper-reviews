Machine Translation
===================

General (Translation Model)
---------------------------
* Christopher Dyer, et al. Generalizing Word Lattice Translation. ACL 2008. http://aclweb.org/anthology//P/P08/P08-1115.pdf
  - Word lattice on the source side (e.g., morph analysis for Arabic, word segmentation for Chinese) and decode through noisier channel (maximum entropy model over (e, f', o))
  - SCFG based decoder - parser over word lattice on deductive proof system
  - phrase-based decoder - keep track of nodes translated, use the length of the shortest path between nodes for distortion models
  - Experiments on Chinese (NIST MT06), and Arabic (NIST MT 08) both improved BLEU scores

Transliteration
---------------

* Tarek Sherif and Grzegorz Kondrak. Substring-Based Transliteration. ACL 2007. http://aclweb.org/anthology//P/P07/P07-1119.pdf
  - Apply phrase-based translation methods to transliteration
  - Two searches 1) Viterbi dynamic programming approach (monotone decoding) and 2) substring-based transducer (can incorporate word unigram model)
  - Evaluation: seen (test set in the LM) and unseen (not in the LM). Substring trans. good for seen test set (70% top-1 accuracy) while Viterbi substring good for unseen test sets

* Ulf Hermjakob et al. Name Translation in Statistical Machine Translation Learning When to Transliterate. ACL 2008. http://aclweb.org/anthology//P/P08/P08-1045.pdf
  - "BLEU do not encourage researchers to improve name translation ... names are vastly outnumbered by prepositions, articles, adjectives, common nouns etc."
  - NEWA (named entity weak accuracy): what percentage of source-language named entities are translated correctly? -> Result: even better than some human translation
  - Transliteration model: find in an English candidate list based on scoring. (include "style flags" for e.g., French words
  - Train "transliterate-me" tagger by word alignment and transliteration detection (Prec 92%, Rec 96%)


Bilingual Term Extraction
-------------------------

* Audrey Laroche and Philippe Langlais. Revisiting Context-based Projection Methods for Term-Translation Spotting in Comparable Corpora. COLING 2010. http://olst.ling.umontreal.ca/pdf/LarocheLanglais2010.pdf
  - Significant gains can be obtained by using (that is rarely used in practice), compared to likelihood score (most popular)
  - Projection-based variants: context - translate - similarity
  - Cognate heuristics - e.g., orthographic features in (Haghighi et al. 2008). In this paper, two words are cognates <=> first four letters are identical
  - LO (log-odds ratio) is significantly superior to the others in every variant
  - Sntence-wide context is more appropriate for autoatic bilingual lexicon construction

* Yun-Chuang Chiao and Pierre Zweigenbaum. Looking for candidate translational equivalents in specialized, comparable corpora. COLING 2002. http://www.aclweb.org/anthology/C02-2020
  - French-English translation candidates from medical domain
  - Seed lexicon: French-English medical lexicon
  - Find the target words that have the most similar distributions with a given source word
  - Weighting = raw, tf.idf, log likelihood, similarity measure = Jaccard, cos
  - Precision boost by applying the same model in the reverse direction

* Philipp Koehn and Kevin Knight. Learning a Translation Lexicon from Monolingual Corpora. ACL 2002. http://homepages.inf.ed.ac.uk/pkoehn/publications/learnlex2002.pdf
  - Identical words - exact same spelling between English and German, 976 such words with 88% accuracy. The longer the words, the more accurate. German->English conversion rule, extra 363 pairs.
  - Similar spelling (cognates) - longest common subsequence ratio (24% accuracy)
  - Similar context - context vector translated (bootstrapped) by previously extracted translations
  - Preserving word similarity - peripheral tokens
  - Frequency - the same concepts should be used with similar frequencies in comparable corpora. Combining clues yields significantly better resutls

* Haghighi et al. Learning Bilingual Lexicons from Monolingual Corpora. ACL 2008. http://anthology.aclweb.org//P/P08/P08-1088.pdf
  - Extract bilingual lexicons (pairs) from (possibly unrelated) bilingual lexicons, (possibly) without seed data
  - Hard EM-like algorithm iterates between (E) find the maximum weighted bipartite matching and (M) Update model parameters by CCA (Canonical Correlation Analysis)
  - Bootstrapping - increase the number of edges gradually
  - Precision at 0.33 89.0% (compared with 61.1% edit distance baseline) for ES-EN
  - Lower precision for other languages, e.g., EN-CH 26.8% @ p0.33 and EN-AR @ 31.1 @ p0.33

* Pascale Fung and Lo Yuan Yee. An IR Approach for Translating New Words from Nonparallel Comparable Texts, COLING 1998. http://acl.ldc.upenn.edu/P/P98/P98-1069.pdf
  - Assumption: words which appear in the context of a word and its translation should be similar to each other
  - Used word pairs from bilingual lexicon as seed words to "bridge" words in context
  - Used tf.idf and confidence (rank in a lexicon) weighting and cosine-like and Dice coefficient similarity measures for ranking translation candidates
  - High precision in the top ranked candidates. Similarity combination (Cosine times weighted Dice) performed the best

* Robert C. Moore. Learning Translations of Named-Entity Phrases from Parallel Corpora. EACL 2003. http://www.aclweb.org/anthology-new/E/E03/E03-1035.pdf
  - Learning translations of multiword phrases (phrases already identified in the source side)
  - Heuristic: exploit exactly the same phrases in the target language -> 17% identical (En-Fr)
  - Model 1: log-likelihood-ratio of sentence-level co-occurrence, and combination of inside + ouside scores, supplemented with capitalization scores
  - Model 2: whole-phrase-based inside scores, interpolated with word-based ones
  - Model 3: counting co-occurrence of the source phrase and its highest scoring candidate translation in a sentence
  - Result: over 80% up to 99% coverage

* Reinhard Rapp. Automatic identification of word translations from unrelated English and German corpora, ACL 1999. http://acl.ldc.upenn.edu/P/P99/P99-1067.pdf
  - Assumption ``If, in a text of one language two words A and B co-occur more often than expected by chance, then in a text of another language those words that are translations of A and B should also co-occur more frequently than expected.``
  - Create word vector of context words considering relative positions (window size of 3), used log likelihood ratio for weighting, and city-block (Manhattan distance) as the similarity measure
  - Experiment: finding English translations for an English input word, from two unrelated corpora of news articles
  - Achieved top1 accuracy of 72%. Ambiguity (e.g., "weiß" for "know" and "white") is an issue.

* Jiajun Zhang, et al. Bilingually-constrained Phrase Embeddings for Machine Translation. ACL 2014. http://nlpr-web.ia.ac.cn/cip/ZongPublications/2014/2014_ACL_Regular_Oral,ZhangJJ,PP111-121.pdf
  - Minimie the semantic distance of translation equivalents and maximizes the semantic distance of non-translation (by randomly replacing words) pairs simultaneously
  - Co-training style algorithm: pre-training with standard recursive auto-encoder, fine-tuning with bilingual constraint
  - Tries to minimize the weighted sum of reconstruction error and sematic error (by linear mapping s->t and t->s)
  - Experiments on phrase table pruning and decoding with phrasal semantic similarities, can remove 72% of the phrase table with only 0.06 BLEU loss

* Dragos Munteanu and Daniel Marcu. Improving Machine Translation Performance by Exploiting Non-Parallel Corpora. Computational Linguistics 2005. http://dl.acm.org/citation.cfm?id=1110828
  - Parallel sentence discovery from comparable corpora. Good quality MT system can be built from very little parallel data and a large amount of comparable, non-parallel data
  - Maximum entropy classifier (given a sentence, output if the sentence is parallel or not): length, word overlap, alignment features (fertility) etc.
  - Bilingual dictionary (for filtering) + training classifiers -> from parallel sentences
  - Article selection: sentence overlap + date window overlap, sentence selection: length, half the words have translation in the other sentence

* Pacale Fung and Percy Cheung: Mining Very-Non-Parallel Corpora: Parallel Sentence and Lexicon Extraction via Bootstrapping and EM. EMNLP 2004 http://www.aclweb.org/anthology/W04-3208
  - Very-Non-Parallel Corpora - even including off-topic sentences
  - "find-one-get-more" - document found to contain one pair of parallel sentences must contain others even if the documents are judged to be of low similarity
  - cf. "find-topic-exact-sentence" - find similar document pairs then look for exact parallel sentences
  - IBM Model 4 EM to learn to find bilingual lexicon, EM initialized by parallel corpora
  - Bilingual lexical matching score (bilingual pair co-occurrence normalized by each word) for parallel, comparable, and quasi-comparable corpora

* Georgios Kontonatsios et al. Combining String and Context Similarity for Bilingual Term Alignment from Comparable Corpora. EMNLP 2014. http://emnlp2014.org/papers/pdf/EMNLP2014177.pdf
  - Compositional clue -> a term translation pair consist of corresponding lexical or sub-lexical units (prefix, n-gram, etc.)
  - Corpus comparability (Li and Gaussier 2010) - the higher, the higher of the performance is.
  - Compositional clue -> binary classification by Random Forest using first-order (raw) character ngrams and second-order (co-occurrence) of ngrams
  - Hybrid model: combines both scores by a linear classifier
  - Experiments: Wikipedia comparable corpora, with 1K term testset, the performance depends on the distance of languages, hybrid a lot better

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

System Combination
------------------
* Antti-Veikko I. Rosti et al. Combining Outputs from Multiple Machine Translation Systems. NAACL 2007. http://www.mt-archive.info/NAACL-HLT-2007-Rosti.pdf
    - Sentence-level combination - select the best hypothesis out of the merged N-best lists, estimating system confidence by the logit model based on several features, aggregate over systems, then rerank (by interpolating with 5-gram langauge model)
    - Phrase-level combination - extract a new phrase translation table from each system's target-to-source alignments and re-decoding the source sentence
    - Word-level combination - built confusion network after choosing a skeleton (the hypothesis that best agrees with the other hypotheses), with system weights
    - Slight gain on the tuning set, very small gain on the test set.

* Evgeny Matusov, et al. Computing Consensus Translation from Multiple Machine Translation Systems Using Enhanced Hypotheses Alignment. EACL 2006. http://www.mt-archive.info/EACL-2006-Matusov.pdf
    - ROVER approach of (Fiscus 1997) - edit distance alignment and time information to create confusion network + voting of several ASR systems
    - Word alignment between primary hypothesis and other hypothesis - IBM Model 1 + HMM (between the same language), training corpus created by translating a test corpus
    - Word reordering - new word order is obtained by sorting words by their aligned word id (in the primary hypothesis)
    - Confusion decoding - done as in the ROVER system (sum up the probabilities of the arcs which are labeled with the same word and have the same start and the same end state)

Optimization
------------

* Franz Och. Minimum Error Rate Training in Statistical Machine Translation. ACL 2003. http://acl.ldc.upenn.edu/acl2003/main/pdfs/Och.pdf
    - Optimize weights of log-linear models so that it directly maximize translation quality criteria (WER, BLEU, etc.)
    - Powells algorithm combined with a grid-based line optimization method. Efficient update achieved on the optimization on a piecewise linear function merged over all sentences in the corpus
    - BLEU ~ +6 point improvement over MMI if trained on BLEU (Chinese to English standard phrase-based SMT system)

Post Editing
------------

* Michael Denkowski et al. Learning from Post-Editing: Online Model Adaptation for Statistical Machine Translation. EACL 2014 http://anthology.aclweb.org//E/E14/E14-1042.pdf
    - ``human translators are more productive and accurate when post-editing MT output than when translating from scratch``
    - Difficulty: update MT models after every sentence corrected -> can't wait for the full model udpate
    - Simulated post-editing: use pre-generated reference as a stand-in for actual post editing.
    - Grammar Adaptation: Lopez 2008's sampling-based pattern match grammar extraction, after forced alignment
    - Language model adaptation: hierarchical Pitman-Yor process priors ("Chinese restaurant franchise")
    - Online learning of feature weights by MIRA


Confidence Estimation
---------------------

* John Blatz, et al. Confidence Estimation for Machine Translation, 2004. http://web.eecs.umich.edu/~kulesza/pubs/confest_report04.pdf
    - Confidence Estimation (CE) - binary classification problem of assessing the correctness of output y, given input x
    - Strong/weak CE, separate or not from system, sentence/subsentence. Use CE to combine multiple systems (even a system is not statistical)
    - "Correct" -> better than a given threshould on an automatic evaluation metric (e.g., WERg, NIST), because a sentence rarely match a reference
    - Corpus: 2001/2002 NIST competition, LDC corpus
    - Sentence-level CE features: base model, Nbest, search, length, source LM, target LM, center hypothesis, syntax, ...
    - Application -> rescoring (no significant improvement)
    - Word-level CE: semantic similarity (e,f), parenthesis, occurrences, IBM model 1, etc.

* Nicola Ueffing and Hermann Ney. Word-Level Conﬁdence Estimation for Machine Translation using Phrase-Based Translation Models. HLT/EMNLP 2005. http://acl.ldc.upenn.edu/H/H05/H05-1096.pdf
    - Word-level CE: possible application: post-editing, interactive translation, system combination
    - Based on phrase-based SMT model, but does not rely on system output (N-best, word graph)
    - System based: word graph, N-best, Phrase based: LM + phrase pair score (phrase penalty and word lexcon model score)
    - Experiment: technical manural corpora, Fr-En Es-En De-En, CER (classification error rate): phrase-based < word-based < IBM-1

* Simona Gandrabur et al. Conﬁdence Estimation for NLP Applications. ACM Transactions on Speech and Language Processing, 2006. http://www.iro.umontreal.ca/~foster/papers/ce-acmtlsp06.pdf
    - NLP tasks are inherently difficult! Confidence measure for "rejection" and "reranking"
    - Confidence measure: given input x, output y, extra knowledge k, and returns i, confidence
    - Machine learning approach: neural nets (multi-layer perceptrons)
    - CE for MT: Proposed in the context of interactive translation tool TransType [Gandrabur and Foster 2003]
    - Correctness probability over predicted n-grams (n = 1 to 4). Features: intrinsic difficulty of source sentence s, how hard s is to translate, how hard s is to translate for the current model

* Radu Soricut, Abdessamad Echihabi. TrustRank: Inducing Trust in Automatic Translations via Ranking. ACL 2010. http://aclweb.org/anthology//P/P10/P10-1063.pdf
    - Example: TripAdvisor review MT: set the quality threshould and adjust the trade-off between quality and coverage
    - Document level, regression model (using Weka, e.g., M5P regression trees) using evaluation measure (BLEU) scores as training labels, rank translations such that the top one is better than the average?
    - Features: no model internal features. text-based (length), LM-based, Pseudo-reference-based (reference from different systems, reverse translation), example-based (using source similarity), training-data-based
    - PBSMT, WMT 09 data, En-Es: rAcc (ranking accuracy) = 45%, BLUE change = +5.9

ESL (English as a Second Language)
----------------------------------

* Longkai Zhang and Houfeng Wang. Go Climb a Dependency Tree and Correct the Grammatical Errors. EMNLP 2014. http://emnlp2014.org/papers/pdf/EMNLP2014033.pdf
  - The general model (correction on dependency tree, replacement model) and the special model (classification for determiners and prepositions, more suitable for insertion and deletion)
  - Long distance dependency (e.g., The book of the boy is) ...TreeNode language Model for the correctness measure - interpolated trigram LM on dep. parse paths + bottom-up decoding using DP
  - CoNLL 2013 shared task, five types of errors of non-native English speakers (determiner, preposition, noun number, subject-verb agreement, verb form)
  - Reliability: 99% of the time noun/verb replacement doesn't change the dependency tree structure.
  - Experiments: used parsed Gigaword as training data, LM->TNLM +2.1% F1 value, Local classifier -> tree classifier +4.1 F1
