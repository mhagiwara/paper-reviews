Information Extraction
======================

General
-------

* Dayne Freitag et al. Information Extraction with HMMs and Shrinkage. AAAI 1999. http://people.cs.umass.edu/~mccallum/papers/ieshrink-aaaiws99.pdf
    - Build each HMM to extract just one type of fileds (e.g., purchasing price)
    - Shrinkage: build a HMM node hierarchy (sth like fallback e.g., suffix -> context -> uniform), estimate the interpolation weight by an EM-like algorithm
    - Experiments: seminar announcements and corporate acquisitions. "Global" shrinkage model performs the best (50% to 90%, depending on NE types)

* Tim Dawborn and James R. Curran. docrep: A lightweight and efficient document representation framework. http://aclweb.org/anthology/C/C14/C14-1072.pdf
    - Cf. heavy-weight DRF (Document representation framework), such as GATE or UIMA. Cf. Stanford NLP Pipeline and CURATOR
    - Support APIs for Python (2.7, 3.3), C++ (C++11), Java (Java 6)
    - Serialization: BSON, MessagePack, ProtocolBuffers, Thrift. BSON worse, MessagePack good in terms of speed and size, and self-describing.
    - Case Study using OntoNote 5 corpus: compared conversion+serialization time and size on disk using UIMA and docrep.

* Denny Vrandečić and Markus Krötzsch. Wikidata: A Free Collaborative Knowledge Base. CACM 2014. http://korrekt.org/papers/Wikidata-CACM-2014.pdf
  - Issues on FreeBase: Einstein is classified as musician (to have his voice recordings)
  - Lua was introduced as a scripting language to Wikipedia (March 2013)
  - First task: move interwiki links from Wikipedia to Wikidata (now they are served from Wikidata)
  - Simple Data: properties (with types) and values (the item Rome has a property population with value 2,777,979).
  - Qualifiers: Rome - has_population - 2.7M (subordinate property-values: as of 2010, method: estimation) DataModel: https://www.mediawiki.org/wiki/Wikibase/DataModel
  - Launched October 2012, the most edited Wikipedia project, 14M items as of Feb 2014, reconciling external IDs
  - Prettify browser: http://tools.wmflabs.org/reasonator/

Set Expansion
-------------

* Richard C. Wang. Language-Independent Class Instance Extraction Using the Web. Ph.D. Thesis 2009. http://www.cs.cmu.edu/~rcwang/papers/phd-thesis.pdf
    - Bilingual SEAL: Iteratively run two instances of iSEAL alternating source/target languages, checking if the new seed's translation exists in the other language set
    - Named entity translation: find frequently co-occurring chunks in the search engine snippet in the target language, ranked by snippet/excerpt frequency and inverse distance.
    - Relational SEAL: seed = pairs, wrappers = (left, middle, right), with a "reverse" flag, can be used for translation pair extraction, ~80% mean average precision

Named Entity Recognition (General)
----------------------------------

* Chris Manning. Doing Named Entity Recognition? Don't optimize for F1. http://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html
    - Error types: tps and tns + false negatives (fn), false positives (fp) + labeling error (le), boundary error (be), label-boundary error (lbe)
    - First 4 types -> 1 demerit (either prec or recall). Other three -> 2 demerits (both prec an recall)
    - F1 encourages not tagging at all rather than making these three errors, which are actually not as bad in practice


* Maksim Tkachenko and Andrey Simanovsky. Named Entity Recognition: Exploring Features. KONVENS 2012. http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
    - Comparison of various features for supervised NER
    - Local knowledge: large context window led to worse F1 measure
    - External knowledge: PoS tags (even high-quality POS tags lead to decreased F1), word/phrase clustering, Encyclopedic knowledge (Wikipedia/DBPedia are still useful, especially with disambiguation info.)

* Vijay Krishnan and Christopher D. Manning. An Effective Two-Stage Model for Exploiting Non-Local Dependencies in Named Entity Recognition. ACL 2006. http://nlp.stanford.edu/manning/papers/Vijay-NER-2pass.pdf
    - Global soft constraint - encourage all the occurrences of the token sequence to the same entity type (with actual stats from CoNLL 2003 data), not so strong for subsequence case (e.g., The China Daily and China)
    - Additional features for the second stage CRF (trained on the training data with 10-fold cross validation): token-majority features, Entity-majority features, Superentity-majority features
    - CoNLL 2003 English data - 12.6% relative error reduction, with inter-document non-local features helped

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

* Scott Miller, et al. Name Tagging with Word Clusters and Discriminative Training. NAACL 2004. http://oldsite.aclweb.org/anthology-new/N/N04/N04-1043.pdf
    - Use discriminative model (averaged perceptron) to use statistically-correlated hierarchical word clusters
    - Add Brown clustering hierarchical IDs (up to leaf levels, where each word is a cluster)

* Arvind Neelakantan and Michael Collins. Learning Dictionaries for Named Entity Recognition using Minimal Supervision. EACL 2014. https://people.cs.umass.edu/~arvind/eacl_final.pdf
    - Construction of NER dictionaries using seeds (positive + negative) and large amounts of unlabeled data
    - Learning a classifier (binary SVM for NE / not NE) using the lower dimensional, real-valued CCA
    - Obtained high recall, low precision candidate phrases by simple patterns and noun phrases
    - Dictionary based NER (gazeteers) F-value increased over candidate list and co-training
    - CRF-based NER: CCA (proposed) F-value increase larger when there are fewer labeled sentences to train

* Charles Sutton and Andrew McCallum. Collective Segmentation and Labeling of Distant Entities in Information Extraction. 2004. http://homepages.inf.ed.ac.uk/csutton/publications/tr-04-49.pdf
    - Skip-chain CRF - connect labels of pairs of similar (identical capitalized) words
    - Exact inference intractable: Approximate inference using loopy belief propagation called tree reparameterization (TRP). Parameter learning by L-BFGS
    - Features for skip chan: combine information (disjunction/or) from the neighborhood of both words
    - Experiments on CMU seminar announcement data. Dramatic decrease in inconsistently mislabeled tokens from 30.2 to 4.8 on the "speaker" field (but no improvement on other fields)


Multilingual NER
----------------

* Mengqiu Wang and Christopher D. Manning. Cross-lingual Projected Expectation Regularization for Weakly Supervised Learning. TACL 2013. http://www-nlp.stanford.edu/mengqiu/publication/tacl13.pdf
    - Project expectation (marginalized posterior) over labels (not explicit labels), work as soft constraints (loss function = difference between expectation and constraints, optimized by L-BFGS)
    - "It is more efficient to label features than examples when the budget is limited (Druck et al 2009)"
    - Projected (hard) labels can be too noisy to be used as directly used as training signals
    - Source side noise model by confusion matrix
    - Outperforms project-then-train CRF training scheme. F1 of 60-64% without training data (equivalent to 12K labeled sentences).

* Mengqiu Wang, Wanxiang Che, Christopher D. Manning. Effective Bilingual Constraints for Semi-supervised Learning of Named Entity Recognizers. AAAI 2013 http://www-nlp.stanford.edu/mengqiu/publication/aaai13.pdf
    - Bilingual NER Constraints: Hard agreement constraints (indicator function if labels match), Soft agreement constraints (replace by PMI of automatically tagged bitext), alignment uncertainty (raising to the power of alignment prob.), global constraint as in (Finkel, Grenager, Manning 2005) + enhansing recall
    - Gibbs sampling with monolingual Viterbi initialization and simulated annealing
    - Relative error reduction of 10.8% (CN) and 4.5% (EN), with huge increase by "rewards" parameter for the global constraints

* Mengqiu Wang, Wanxiang Che, and Christopher D. Manning. Joint Word Alignment and Bilingual Named Entity Recognition Using Dual Decomposition. ACL 2013. http://www-nlp.stanford.edu/mengqiu/publication/acl13.pdf
    - Bilingual NER by Agreement - minimize CRF prob. while agreeing on the labels y(e) and y(f), solved by dual decomposition. Soft agreement - introduce a factor between word positions (i, j), raised to the power of PMI(i, j)
    - Joint alignment and NER decoding: joint optimization of alignment variable (both en->cn and cn->en) and edge potential (i, j)
    - Absolute F1 improvement of 6.7% in Chinese over monolingual baselines. Joint alignment and NER: 10.8% error reduction in AER (word alignemnt) and better NER performance.

* Fei Huang and Stephan Vogel. Improved Named Entity Translation and Bilingual Named Entity Extraction. 2002. http://isl.anthropomatik.kit.edu/downloads/icmi2002_huang.pdf
    - Iteration between NE alignment and NE dictionary updates
    - Translation prob. of NE pair = IBM model1. Sentence level NE alignment = minimum of the sum of the translation cost of aligned NEs (using competitive linking algorithm)
    - NE dictionary probability = normalized alignment frequencies over a corpus
    - Augmented NE alignment cost = linear interpolation between translation prob. and alignment prob.
    - F-value (and the NE dictionary) improved over iterations


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
    - Retrain monolingual model -> useful when lacking annotated data but bitexts are plentiful

* Ryan McDonald et al. Multi-Source Transfer of Delexicalizaed Dependency Parsers. EMNLP 2011. http://www.petrovi.de/data/emnlp11a.pdf
  - Delexicalized dep parser trained in English (only rely on PoS tags, universal PoS tagset). Romance languages tend to transfer to one another.
  - Seed a perceptron learner (direct transfer, "re-lexialize", parse parallel sentences on both sides, choose a parse which aligns best with Englishs)
  - Arc-eager transition parser trained with Averaged perceptron with beam search (used MSTParser with similar results)
  - Direct transfer UAS ~ 55% with gold-PoS (target langs.: Indo-European languages with significant parallel data), after projection ~ 60%
  - Multi-source languages (concatination of the all the training data) -> 63.8% UAS
  - Comparison with USR (universal syntactic rules; Naseem et al. 2010), PGI (phylogenetic grammar induction; Berg-Kirkpatrick 2010), PR (posterior regularization; Ganchev et al. 2009)
  - UAS 39.9/41.7/43.3% for Arabic, Chinese, and Japanese using multi-source direct transfer


* Oscar Täckström et al. Target Language Adaptation of Discriminative Transfer Parsers. NAACL 2013. http://soda.swedish-ict.se/5501/1/paper.pdf
  - Ambiguity-aware self-training - consider arc marginals and cut by a threshold
  - Graph-based parser (discriminative model, trained by L-BFGS, decoded by Eisner's algorithm)
  - Bare features -> delexicalized features minus direction dependent ones -> works better for Basque, Hungarian, and Japanese
  - Shared directional features conjoined with WALS (World Atlas of Language Structures) feature
  - "Family" model shares features among the same language families (UAS 57.4% -> 62.0%)
  - Ambiguity-aware ensemble-training: union of NBG parser (Naseem et al 2012) and base parser ambiguity sets

* Mo Yu et al. Cross-lingual Projections between Languages from Different Families. ACL 2013. http://www.asiteof.me/acl2013.pdf
   - Direct projection, or projection based on word alignment?
   - Brown clusters based word representations on the target side and noise removal (re-implementation of cluster projectino of Tackstrom 2012)
   - Noise removal: based on the whole corpus marginals on words, brown clusters, and bigrams
   - Experiments: NER En->Zh: F = 33.91 (baseline - direct projection), 56.60 (word alignment projection), 62.53 (after noise removal)

* Greg Durett et al. Syntactic Transfer Using a Bilingual Lexicon. EMNLP 2012 http://www.eecs.berkeley.edu/~gdurrett/DurPauKle_emnlp12.pdf
   - Dependency parsing (MSTParser): UAS increase of ~2 points
   - Transfer type-level information (instead of token level) -> Proj. features: queries (parent word -> child word, direction, distance), signature (pos -> pos, a flag specifying if they are lexicalized)
   - Query value estimation: using (probabilistic) bilingual lexicon (weights for marginalizing out target translation)
   - AUTOMATIC lexicon (collected from word alignment) better than MANUAL lexicon (because of lower coverage)

* Daniel Zeman and Philip Resnik. Cross-Language Parser Adaptation between Related Languages. IJCNLP 2008. http://www.aclweb.org/anthology/I08-3008
   - Building a parser of a language without a treebank
   - Danish -> Swedish (closely related, almost dialects). Result: 1,500 Swedish parse trees would be required for training
   - Treebank normalization (normalizing two different sets of annotation schemes), mapping tags
   - Bridging the gap between languages: glossing (word alignments), delexialization (by only using PoS tags),
   - Evaluated by evalb http://nlp.cs.nyu.edu/evalb/ (by Sekine and Collins)

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

* Winston Lin et al. Bootstrapped Learning of Semantic Classes from Positive and Negative Examples. ICML 2003. http://www.cs.nyu.edu/roman/Papers/2003-icml-nomen.pdf
  - Thelen and Rillof (2002) and Yangarber et al. (2002) found that performance was improved when multiple semantic clases were learned simultaneously.
  - Process: generate patterns -> evaluate pattens (positive / negative / unknown examples) -> acquire patterns (scoring by accuracy and confidence) -> apply patterns -> acquire names.
  - Experiments (lexicon generation) specialized corpus (ProMED) based on a recall list and a precision list.
  - Target classes < + Other category < split the other category into six competing categories.

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

* Wayne Xin Zhao, et al. Group based Self Training for E-Commerce Product Record Linkage. COLING 2014. http://aclweb.org/anthology/C/C14/C14-1124.pdf
    - Usually recast as the record pair classification problem -> correlation (one product is only linked to one another)
    - Products are described by sets of attributes. Candidate record pairs share the same set of attributes. Assume one-to-one mapping between databases.
    - Confidence defined: product of (score for positive pair) and (negative score for negative pairs in conflicting links) -> self training
    - Experiments: linking Jingdong with eTao: group based self training achieved best F1 value compared with simpler classifier and self training

* Michael Wiegand and Dietrich Klakow. Separating Brands from Types: an Investigation of Different Features for the Food Domain. COLING 2014. http://aclweb.org/anthology/C/C14/C14-1216.pdf
    - Distinction between brands and food types (sometimes blurred e.g., sprite)
    - Simple baseline: use of coordination starting from a few seeds
    - Intrinsic property: desirable brand name characteristics (e.g., onomatopoeia, word-initial plosives, word length, etc.) -> not effective excpt for length (brands tend to be shorter)
    - Ranking problem directly using features: length, NER, diversification (e.g., affix "light"), commerce cues (e.g., buy, purchase, customer), graph clustering, Food Guide Pyramid, Wikipedia classification, vector space model
    - Data: German food domain. Proposed F1-value ~73%
