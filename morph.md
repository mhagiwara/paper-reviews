Morphology (Word Segmentation & PoS Tagging)
============================================

Word Segmentation
-----------------
* Nobuhiro Kaji et al. Efficient Staggered Decoding for Sequence Labeling. ACL 2010. http://anthology.aclweb.org//P/P10/P10-1050.pdf
    - Viterbi decoding slow - O(NL^2) where L is the # of labels
    - Staggererd decoding: group labels into degenerate labels, activate labels by their relative frequency, continues until the best path doesn't go through degenerate nodes, transition scores = max of degenerate nodes
    - Pruning based on score lower bound and maximum for each label
    - PoS tagging, PoS tagging and chunking, and supertagging: several order of magnitute faster than Viterbi, CarpeDiem, comparable with beam search with similar performance.

* Manabu Sassano. Deterministic Word Segmentation Using Maximum Matching with Fully Lexicalized Rules. EACL 2014. http://anthology.aclweb.org//E/E14/E14-4016.pdf
    - Maximum algorithm with transformation rules (sequence of characters -> sequences of morphemes)
    - Learning transformation rules - by transformation-based learning (TBL; Brill 1995) - if system makes an error, learn a rule [word sequence with error] -> [correct sequence]. Sort the rules by scores and accept the rule if it gives positive
    - F measure almost 0.96 after training on Web corpus, 4 times faster than MeCab. Higher performance by post-processing.

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
    - Probabilisitc models based on multinomial distributions (maximum likelihood in each helper language), such as HMM and DMV (dependency model with valence)
    - Use helper languages for initialization, then unsupervised learning to learn the each contribution
    - Mixture coefficient \beta learnd via EM algorithm
    - Re-parametrize \theta by feature-rich unsupervised model (Berg-Kirkpatrick et al. 2011) and beta (to ignore simplex constaints)

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

Word Embeddings
---------------

* Richard Socher et al. Parsing Natural Scenes and Natural Language with Recursive Neural Networks. ICML 2011. http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Socher_125.pdf
    - RNN computes (i) a score that is higher when neighboring regions should be merged into a larger region, (ii) a new semantic feature representation for this larger region, and (iii) its class label.
    - Input for sentence parsing: word representation from neural language models
    - We want the score of the highest scoring correct tree to be larger up to a margin defined by the loss (for sentence, is the sum over incorrect spans in the treee)
    - Training by subgradient method, calculate the derivative by using backpropagation through structure
    - Final F measure 90.29% (Berkley Parser 91.63%), but found interesting sentence level paraphrases

* Tomas Mikolov, et al. Exploiting Similarities among Languages for Machine Translation. arXiv 2013. http://arxiv.org/pdf/1309.4168v1.pdf
    - Word representation (Skip-gram) with a linear projection between the languages
    - "In practice, Skip-gram gives better word representations when the monolingual data is small. CBOW [Continuous Bag-of-Words] however is faster and more suitable for larger datasets"
    - Translation matrix - linear transform Wxi -> z. Train by stochastic gradient descent so as to minimize the sum of L2 error between the two.
    - Used Google Translate to create gold dictionaries (for training and testing). En<->En 90% precision @ 5.
    - Weighted combination of edit distance based similarity and matrix based similarity
    - Thresholding on the confidence score, defined by the maximum cosine similarity over possible candidates in the target language

* Tomas Mikolov, et al. Distributed Representations of Words and Phrases and their Compositionality NIPS 2013. http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    - Skip-gram model: predict the surrounding words from the center word. Objective = maximize the average of log prob of p (w_context|c_center), defined by SoftMax
    - Improvements: Hierarchical Softmax, Negative sampling (distinguish the target word from draws from the noise distribution, unigram distribution raised to the 3/4rd power)
    - Subsampling of the frequent words improves the training speed several times and maek the word representations significantly more accurate
    - Phrase identification: bigram_count(wi, wj) / (unigram_count(wi) x unigram_count(wj)), 2-4 passes over the training data to allow phrases 2+ word long
    - Why additive compositionality? the sum of two word vectors is related to the product of the two context distributions

* Tomas Mikolov, et al. ICLR 2013. Efficient Estimation of Word Representations in Vector Space. http://arxiv.org/pdf/1301.3781.pdf
    - Comparison of four models: Feedforward Neural Net Language Model (NNLM), Recurrent Neural Net Language Model (RNNLM), Continuous Bag-of-Words Model, and Continuous Skip-gram Model
    - Experiment on word similarity task (e.g., What is the word that is similar to "small" in the same sense as "biggest" is similar to "big"?)
    - Have to increase both vector dimensionality and the amount of the training data (e.g. 600 dim for 783M training words)
    - NNLM better than RNNLM, CBOW and Skip-gram better than the other two
    - Microsoft Research Sentence Completion Challenge: Skip-gram + RNNLMs (weighted combination) the best

* Joseph Turian, Lev Ratinov, and Yoshua Bengio. Word representations: A simple and general method for semi-supervised learning. ACL 2010. http://www.newdesign.aclweb.org/anthology/P/P10/P10-1040.pdf
    - Comprehensive review of distributuional word representation methods, including: distributional similarity, SOM, LSA, LDA, Hyperspace Analogue to Language (HAL), random projection
    - Distributed representation (aka word embeddings; not to confused with distributional representation) - Collobert and Weston (2008) embeddings and hierarchical log-bilinear (HLBL) model (Mnih and Hinton 2007)
    - Tasks: Chunking (CoNLL-2000 shared task) and NER (CoNLL 2003 and MUC7), following Ratinov and Roth (2009)
    - Scaling of word embeddings: reasonable choice of scale factor is such that the embeddings have a standard deviation of 0.1.
    - Performance increase by combining different types of word representations, Brown clustering tops (good representation for rare words). C&W embeddings better than HLBL.

* Alexandre Klementiev and et al. Inducing Crosslingual Distributed Representations of Words. COLING 2012. http://ivan-titov.org/papers/coling12distr.pdf
    - Treat as a multitask learning problem (Cavallanti et al. 2010) where each task corresponds to a single word, with prior knowledge encoded in matrix A
    - Neural language models of Bengio et al. (2003) predicting P(w | history), trained by maximizing likelihood by stochasitc gradient descent.
    - Document classification between English-German. Beat baselines (glossing, MT). 70-80 classification accuracy.


* Karl Moritz Hermann and Phil Blunson. Multilingual Models for Compositional Distributed Semantics. ACL 2014. http://arxiv.org/pdf/1404.4641.pdf
    - Only requires sentence aligned parallel corpus (no word alignment)
    - CVM (compositional vector model): compute semantic representations of sentences and documents. Vector Addition (ADD) and bigram (BI), recursive application to documents
    - Minimize the difference between aligned sentences while ensures the margins between non-aligned sentences
    - Trained on Europerl corpus v7 and TED corpus for IWSLT 2013, evaluated for multilingual document classification using Reuters corpora
    - Compared with MT baseline (slightly better), BI performing better than ADD

* Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations. CoNLL 2014 http://levyomer.files.wordpress.com/2014/04/linguistic-regularities-in-sparse-and-explicit-word-representations-conll-2014.pdf
    - Improved method of recovering relational similarities (aka linguistic regularities)
    - "Neural embedding process is not discovering novel patterns, but rather is doing a remarkable job at preserving the patterns inherent in the word-context co-occurrence matrix."
    - Compared with context based sparse vector representation (PMI weighting + distinguishing different positions)
    - 3CosAdd model: arg max sim(b*, b - a + a*) = arg max (cos (b*, b) - cos(b*, a) + cos(b*, a*))
    - 3CosMul model: arg max cos(b*, b) cos(b*, a*) / (cos(b*, a) + eps) -> performing better than state-of-the-art
    - Most common error cause: "default behavior" -> some very frequent words act as "hubs" and confuse the model

* Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. ICML 2014. http://cs.stanford.edu/~quocle/paragraph_vector.pdf
  - Paragraph vector PV-DM (distributed memory) version -> shared by all contexts generated from the same paragraph but not across paragraphs.
  - PV-DBOW (distributed bag-of-words) -> predict words randomly sampled from the paragraph in the output.
  - Usually PV-DM performs better, but combining PV-DM and PV-DBOW performed even better
  - Prediction: fix the word vectors and infer paragraph vectors
  - The proposed PV performed better than recursive networks even though they don't use parsing

* Rami Al-Rfou et al. Polyglot: Distributed Word Representations for Multilingual NLP. CoNLL 2013 http://arxiv.org/pdf/1307.1662.pdf
  - Used neural language model (SENNA; Collobert and Weston 2008) - state-of-the-art performance in Danish, Enlish, Swedish
  - Trained word representations in 100+ languages using their Wikipedias
  - Train PoS taggers using only word embeddings model
  - Avoid excessive normalization, e.g., "Apple" (IT company) vs "apple" (fruits)

* Marco Baroni et al. Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors. ACL 2014 http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf
  - Empirical comparison of context-counting (DistSim) and contex-predicting (word embeddings) methods
  - Most effective parameter set for count-based vectors: narrow context window, positive PMI, SVD reduction -> no compression was better
  - Predict-based models are a clear winner
  - Collobert and Weston out-of-the-box embeddings didn't work so well

* Lauly et al. Learning Multilingual Word Representations using a Bag-of-Words Autoencoder. NIPS 2013 workshop on deep learning.
  - learning multilingual auto-encoder with bitext (without word alignment)
  - autoencoder - input bag-of-words -> sum of embeddings -> non-linear decoder -> bag of words (hierachical softmax-like encoding)
  - 1) lang x -> lang y 2) lang y -> lang x 3) x -> x 4) y -> y
  - Experiments: crosslingual classification (classifier trained in lang x to classify lang y based on sum of word representations)
  - Competitive performance compared with Klementiev embeddings

* Richard Socher, et al. Semantic Compositionality through Recursive Matrix-Vector Spaces. EMNLP 2012 http://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf
  - Vector - inherent meanng of word and constituent, matrix - how it changes the meaning of neighboring words and phrases
  - e.g., if a word lacks operator semantics, matrix -> identity matrix; if word acts as an operator -> vector close to zero
  - initialize all word vectors x with representation of Collobert and Weston (2008)
  - add softmax classifier on top of each parent node to predict a class distribution
  - Experiment 1 : adverb-adjective pairs from movie review, predict number of starts, evaluate in terms of KL-divergence (multinomial over stars)
  - Accuracy of movie review polarity classification: higher than Tree-CRF, RAE, Linear MVR

* Richard Socher, et al. Parsing with Compositional Vector Grammars. ACL 2013. http://nlp.stanford.edu/pubs/SocherBauerManningNg_ACL2013.pdf
  - Jointly learns how to parse and how to represent phrases (as both discrete categories and continuous vectors)
  - Structured margin loss -> the number of nodes with an incorrect span, max-margin, structure prediction objective (highest scoring tree -> correct tree)
  - Parent vector = f (W [b c]) (f: non-linear mapping, W: conditioned on the syntactic categories). Final score: sum of vector based score and log of PCFG score.
  - Two pass decoding: 1) CKY DP for PCFG (for top k = 200) and 2) beam search with the full CVG model.
  - best performance boost comes from PP attachement, vectors learn a soft notion of head words
