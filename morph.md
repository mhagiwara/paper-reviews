Morphology (Word Segmentation & PoS Tagging)
============================================

Word Segmentation
-----------------
* Hai Zhao and Chunyu Kit. An Empirical Comparison of Goodness Measures for Unsupervised Chinese Word Segmentation with a Uniﬁed Framework. ICJNLP 2008. http://www.aclweb.org/anthology/I/I08/I08-1002.pdf
    - Comparison of unsupervised WS based on word goodness scores (FSR, DLG, AV, BE) and decoding algorithms (viterbi optimization / maximal forward matching)
    - Experiments on Bakeoff-3 (AS, CityU, CTB, MSRA corpora), DLG outperforms on 2-chars and AV/BE on 3+-chars (because two-character words are dominant in Chinese), viterbi optimization wins
    - Word candidate pruning, ensemble segmentation (taking intersection of word candidates) effective, F range 0.65 - 0.7.

* Chunyu Kit and Yorick Wilks. Unsupervised Learning of Word Boundary with Description Length Gain. CoNLL 1999. https://aclweb.org/anthology/W/W99/W99-0701.pdf
    - DLG (description length gain), following MDL (minimum description length)
    - DL = the Shannon-Fano code length of the corpus (over "tokens", characters, words, etc.), gain = difference of DL when substituting token seqs with another symbol
    - Decoding: Viterbi algorithm to maximize the sum of average DLGs over chunks
    - Experiment on the entire Brown corpus: Prec = 79%, Rec = 63%


* Mathias Creutz. Unsupervised Segmentation of Words Using Prior Distributions of Morph Length and Frequency. ACL 2003. http://acl.ldc.upenn.edu/acl2003/main/pdfs/Creutz.pdf
    - Divide words into morphs (smaller segments) from corpus assuming a generative model
    - Generate lexicon (size = uniform), morph types (gamma distribution over length), morph strings (multinomial over characters), morph frequencies (refined Zipf's formula)
    - Algorithm: greedy split until the probability converges
    - Evaluation: percentage of (aligned in the training data) recognized morphemes: approx. 40% in Finnish and 45% in English (approx. linear vs log of corpus size)

* Sharon Goldwater et al. Contextual Dependencies in Unsupervised Word Segmentation. ACL-COLING 2006. http://cocosci.berkeley.edu/tom/papers/wordseg1.pdf
    - Comparison with MBDP (Model-Based Dynamic Programming; Brent 1999) and NGS (n-gram Seggmentation; Venkataraman 2001)
    - Unigram: Dirichlet process with \alpha_0 (acts like the parameter of an infinite-dimensional symetric Dirichlet distribution) and P_0 (base distribution; unigram phoneme dist.) + Gibbs sampling + annealing (raising the probabilities of h1 and h2 to the power of 1/\gamma)
    - Bigram: hierarchical Dirichlet process, where each word w is associated with its own restaurant, which represents the distribution over words that follow w.
    - Experiments unigram -> lexicon accuracy higher but token F (~ 53.8) lower. bigram: both lexicon and token accuracy are higher than unigram (esp. tokens, F ~ 76.6)

* Teemu Ruokolainen et al. Painless Semi-Supervised Morphological Segmentation using Conditional Random Fields. EACL 2014. http://aclweb.org/anthology//E/E14/E14-4017.pdf
    - Linear chain CRF, IOB2 model on characters, with standard emission and transition features
    - Feature set augmentation, utilizing unsupervised segmentation algorithms: 1) Morfessor model (Creutz and Lagus 2007, see above), 2) LSV (letter successor variery, Coltekin 2007)
    - Experiments: Morph Challenge 2009/2010, English, Finnish, Turkish. F - upper 80s (CRF + Morphessor CatMAP+Harris) or even 90s (with 1,000 training instances)


PoS Tagging
-----------

* Dipanjan Das and Slav Petrov. Unsupervised Part-of-Speech Tagging with Bilingual Graph-Based Projections. ACL 2011. http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/37071.pdf
    - Build a PoS tagger (on universal PoS tag set) for resource scarce languages (using parallel data and English supervised PoS tagger)
    - Projection and label propagation on a bilingual graph of trigrams (in foreign language) and word types (in English), using similarity statistics and word alignment
    - PoS tagger was trained on feature-based HMM (Berg-Kirkpatrick et al. 2010) using tag distributions as features
    - Experiment: used 8 Indo-European languages from Europarl and ODS UN dataset
    - Full model achieved 10.4 point increase from the state-of-the art and 16.7% from vanilla HMM

* Shay B. Cohen, et al. Unsupervised Structure Prediction with Non-Parallel Multilingual Guidance. EMNLP 2011. http://www.cs.columbia.edu/~scohen/emnlp11multilingual.pdf
    - Learn models of the target language using annotated data from helper languages, without any form of parallel data.
    - Probabilisitc models based on ultinomial distributions, such as HMM and DMV (dependency model with valence)
    - Use helper languages for initialization, then unsupervised learning to learn the each contribution


Multi-Word Expressions
----------------------

* Katerina Frantz, et al. Automatic Recognition of Multi-Word Terms: the C-value/NC-value Method, International Journal on Digital Libraries, 2000. http://personalpages.manchester.ac.uk/staff/sophia.ananiadou/ijodl2000.pdf
    - C-value: "termhood" - nested multi-word terms. row freq minus (normalized) freqs of words which contain the term, weighted by length.
    - NC-value incorporates context information (term context words which appear in the vicinity of terms), weighted by importance (relative freq. of appearance with terms)
    - From 810K-word eye-pathology medical record, extracted terms with 75% precision (top 40) using NC-value (5 point increase over C-value)

* Youngja Park, et al. Automatic Glossary Extraction: Beyond Terminology Identification. COLING 2002. https://aclweb.org/anthology/C/C02/C02-1142.pdf
    - Extraction of glossary items (e.g., words/phrases to include published glossaries)
    - Recognize phrases by PoS tag sequences, finite state transducers
    - Calculate domain specificity and association of pre-modifiers and remove generic ones
    - Variant aggregation: symbolic variants, compounding variants, inflectional variants, misspelling variants, abbreviations
    - Ranking based on term domain-specificity and term cohesion
    - Many domain specific terms in Top300 and less in Bottom300, compared to LLR and MI
