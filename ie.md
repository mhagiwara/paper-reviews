Information Extraction
======================

General
-------

* Dayne Freitag et al. Information Extraction with HMMs and Shrinkage. AAAI 1999. http://people.cs.umass.edu/~mccallum/papers/ieshrink-aaaiws99.pdf
    - Build each HMM to extract just one type of fileds (e.g., purchasing price)
    - Shrinkage: build a HMM node hierarchy (sth like fallback e.g., suffix -> context -> uniform), estimate the interpolation weight by an EM-like algorithm
    - Experiments: seminar announcements and corporate acquisitions. "Global" shrinkage model performs the best (50% to 90%, depending on NE types)

Set Expansion
-------------

* Richard C. Wang. Language-Independent Class Instance Extraction Using the Web. Ph.D. Thesis 2009. http://www.cs.cmu.edu/~rcwang/papers/phd-thesis.pdf
    - Bilingual SEAL: Iteratively run two instances of iSEAL alternating source/target languages, checking if the new seed's translation exists in the other language set
    - Named entity translation: find frequently co-occurring chunks in the search engine snippet in the target language, ranked by snippet/excerpt frequency and inverse distance.
    - Relational SEAL: seed = pairs, wrappers = (left, middle, right), with a "reverse" flag, can be used for translation pair extraction, ~80% mean average precision

Named Entity Recognition (General) 
----------------------------------

* Jenny Rose Finkel, et al. Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling. ACL 2005. http://nlp.stanford.edu/manning/papers/gibbscrf3.pdf
    - Enforce (non-local) label consistency in NER
    - Gibbs sampling on CRF (markov network with transition cliques) with simulated annealing
    - Penalty model: penalty weight ^ # of violation, combined with the baseline model
    - CoNLL consistency model: penalize consistent entities with empirical bayes estimates from the training data. Seminar announcement: inconsistency of start/end time and speaker

* Andrew Arnold, et al. Exploiting Feature Hierarchy for Transfer Learning in Named Entity Recognition. ACL 2008. http://www.cs.cmu.edu/~wcohen/postscript/acl-2008.pdf
    - Significance of some generalized classes of features retain their importance across domains (Minkov et al. 2005)
    - Two subproblems of transfer learning: domain adaptation and multi-task learning.
    - CRF with Gaussian priors (mu's and sigma's for each feature) to regularize it towards the source domain (Celba and Acero, 2004), fallback to N(0, 1)
    - HIER: Fallback to the parent node, sharing a same tree T across different domains, with hyperparameters shared by the parent nodes
    - Approximate model: Back-off in the tree until we had a large enough sample of prior data (M, number of subtrees)


Multilingual NER
----------------

* Alexander E. Richman and Patrick Schone. Mining Wiki Resources for Multilingual Named Entity Recognition, ACL 2008. http://www.mt-archive.info/ACL-2008-Richman.pdf
    - Use categories to classifiy English pages, and map NE types via interwiki links. Use this as the type of a wikilink (intra-language link to another article). Foreign language knowledge is not requried.
    - If a foreign page doesn't have a link to its English translation, use cateogory translation to classify types
    - Trained a NER system (PhoenixIDF), tested on ACE 2007 (newswire) and Wikipedia held-out data, with approx. 0.82 ~ 0.84 F-value in Spanish, French, and other languages

* Sungchul Kim, Kristina Toutanova, and Hwanjo Yu. Multilingual Named Entity Recognition using Parallel Data and Metadata, ACL 2012. http://www.newdesign.aclweb.org/anthology-new/P/P12/P12-1073.pdf
    - Wiki-tagger: Built local and global taggers based on article categorization (as in Richman and Schone 2008)
    - Mapping-based taggger: a ranking (log-linear) model from English entities to foreign entity spans (Feng et al. 2004), trained on 100 parallel sentences with NEs and alignments
    - Combination of these two via semi-Markov CRF (Sarawagi, Cohen 2005) outperforms individual models (~91% F-value in Korean and Bulgarian)

* Oscar Täckström et al. Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure. NAACL 2012 http://soda.swedish-ict.se/5251/1/paper.pdf
    - Use of word cluster features (Uszkoreit and Brants 2008) for (monolingual, non-transfer) parsing and NER
    - Delexicalized direct transfer method (McDonald et al. 2011 and Zeman and Resnik 2008): trained on the source language without lexicalization (but de-lexicalization degraded performance for NER)
    - NER performance (F-value) on CoNLL 2002/2003 test set: baseline: 39.1, with X-LINGUAL clusters: 52.7

* Kareem Darwish. Named Entity Recognition using Cross-lingual Resources: Arabic as an Example. ACL 2013. http://aclweb.org/anthology//P/P13/P13-1153.pdf
    - Positive effect of cross-lingual features (especially on recall)
    - 1. cross-lingual capitalization (through true-cased phrase table), 2. transliteration (surrogate of capitalization since transliteration is a strong clue for transliteration), 3. DBpedia classification (e.g., B-Organization) via interwiki links
    - Overall, 20.5% F-value increase on the TWEETS data

* David Burkett et al. Learning Better Monolingual Models with Unannotated Bilingual Text. CoNLL 2010. http://www.cs.berkeley.edu/~dburkett/papers/burkett10-bilingual_multiview.pdf
    - Supervised monolingual models -> improve using bilingual parallel corpus
    - Multiview bilingual model: parametrize using one-to-one matching between nodes (named entities, node in parse tree). Example: ambiguous PP attachement in EN but not in ZH
    - Introduce the output of deliberately weakened monolingual models as features in the bilingual view
    - Retrain monolingual model -> useful when lacking anotated data but bitexts are plentiful

Unsupervised NER
----------------

* Oren Etzioni et al. Unsupervised named-entity extraction from the web: An experimental study. Artificial Intelligence, 2005. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.124.8829&rep=rep1&type=pdf
    - Extraction of unary and binary relation using Hearst-style lexico-syntactic patterns (LSPs; e.g., NP1 such as NPList2) generated from class names. Then validation by PMI co-occurence on the Web.
    - Recall improvement: Learning patterns from seeds and use them as discriminators, subclass extraction (chemists or bilologists as opposed to scientists) again by LSPs, and list extractor (exactly as in SEAL)
    - list extractor's extraction rate was the greatest (more than x40 increase)


* Alessandro Cucchiarelli and Paola Velardi. Unsupervised named entity recognition using syntactic and semantic contextual evidence. Computational Linguistics, 2001. http://acl.ldc.upenn.edu/J/J01/J01-1005.pdf
    - Learn typical syntax/semantic context from a corpus to expand gazetteers
    - Syntactic info: elementary syntactic link (esl): Subject-Object, Noun-Preposition-Noun, etc.
    - Unknown proper noun classification: to the maximum evidence, defined by the relative plausibility of each detected esl., augmented by WordNet similarity
    - Test on Italian (Sole 24Ore) corpus and WSJ corpus: Prec and Recall both up by ~10%, 5 point increase on F-measure by WordNet context generalization

* D Nadeau et al. Unsupervised Named-Entity Recognition: Generating Gazetteers and Resolving Ambiguity. 2006. http://brown.cl.uni-heidelberg.de/~sourjiko/NER_Literatur/NER_Turney.pdf
    - Extraction (generating gazetteers): Wrapper based set expansion from few seeds, repeat retrieving Web pages and applying Web wrapper
    - Disambiguation: entity-noun ambiguity (capitalization heuristics), entity boundary detection (longest match, merge consecutive entries), entity type ambiguity (use clues in the same document, alias clues)
    - MUC 7 evaluation: Generated lists provide higher recall but lower precision.
    - Car brand evaluation: generated a list of 5,701 brands and recognition F ~ 86 
    
Product Information Extraction
------------------------------

* Rayid Ghani et al. Text Mining for Product Attribute Extraction. SIGKDD 2006. http://www.accenturehighperformancedelivered.com/SiteCollectionDocuments/PDF/sigkdd06.pdf
    - Implicit semantic attributes: manually define semantic attributes (e.g., age group, functionality, price point, etc.) and build supervised/semi-supervised text classifier
    - Explicit attributes (color, size, etc.): generate seeds (by a PMI-like cooccurrence measure), classify phrases into attribute/values/neither, then link them by dependencies
    - Precision (fully correct: around 30 to 50 percent, partially correct: more than 90 percent), recall (75% by co-EM)
    - Applications: recommender systems, copywriters marketing, store profiling & assortment comparison

* Anton Bakalov and Ariel Fuxman. SCAD: Collective Discovery of Attribute Values. WWW 2011. http://talukdar.net/papers/scad_www2011.pdf
    - Tag attribute values from a) schema, b) a small set of seed entities + attribute/values c) Web pages 
    - Density estimation from ground truth attributes
    - Global optimization for decoding, including entity-level consensus (one attribute constraint, no value conflict, etc.), category-level constraints (Gaussian kernel density of the value from the ground truth values)
    - Local model (assigning snippets to attributes): logistic regression using attribute names, words, etc.

* D. Putthividhya and J. Hu. Bootstrapped Named Entity Recognition for Product Attribute Extraction. EMNLP 2011. http://aclweb.org/anthology//D/D11/D11-1144.pdf
    - Attribute extraction from listing titles (long, little grammatical structure, errors, lack of context) by NER
    - Clothing and shoes listings from eBay, brand, garment, type/style, size, color
    - Supervised (SVM, MaxEnt) and sequential labeling by HMM (1,000 annotated items, 93.35% label classification accuracy with CRF)
    - Bootstrapping (MaxEnt, supervised) using context starting from initial seeds, with 90% precision, brand name normalization (using n-gram similarity and Jaro-Winkler distance)

* K. Mauge et al. Structuring E-Commerce Inventory. ACL 2012. http://aclweb.org/anthology//P/P12/P12-1085.pdf
    - Pattern-based property extraction (e.g., color : light blue)
    - Synonym discovery: supervised (maximum entropy) classifier over defined features with popularity (defined by # of sellers which use the attribute) and graph partitioning
    - Experiment: one-year worth (several billion) desc. of eBay listings. Prec = almost 100% up to rank 18, then drops. Synonym discovery: Prec. = 91.8% and Recall = 51%.

* K. Shinzato, S. Sekine. Unsupervised Extraction of Attributes and Their Values from Product Description. IJCNLP 2013. https://www.aclweb.org/anthology/I/I13/I13-1190.pdf
    - Build knowledge base (attribute and values) from unstructured product description, using table headers (for attribute labels) and manually crafted patterns (for values)
    - Attribute synonym discovery: pairs sharing popular values and never appear in the same structured data
    - Annotate product pages using KB (from the same genre) values, calculate the ratio MF (in desc.) and MF (in structured data) and removed high score sents.
    - Correct attribute labels = 77.6%, synonym detection purity/inv. purity = 0.920 and 0.813, extraction model (CRF) micro-averaged F = 58.15%

