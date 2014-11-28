Machine Learning
================

General
-------

* Pedro Domingos. A Few Useful Things to Know about Machine Learning. CACM 2012. http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
    - Explaining and communicating "black art" "folk knowledge" of machine learning
    - Learning = Representation + Evaluation (distinguish good classifiers from bad ones) + Optimization (search among the classifiers). The latter two are often overlooked
    - It's generalization that counts: separation of train / test, holding out is important. We don't have access to the objective function, but the surrogate can be sub-optimal
    - Data alone is not enough: we can't learn every function, but possible functions are distributed not uniformly
    - Overfitting: bias and variance. strong false assumptions (e.g., Naive Bayes) can be better than weak true ones (e.g., decision tree) because a learner with the latter needs more data to avoid overfitting.
    - Dimensionality: Generalizing correctly becomes exponentially harder as dimensioality grows. Nearest neighbor, mass of Gaussian distribution, etc. are counter intuitive
    - Beware of theoretical guarantees (not a criterion for practical decisions) - they are loose, need exponentially large examples vs dimensionality. So is "Asymptopia"
    - Feature Engineering is the Key: ML is an iterative process of learning (the quickest), analysis, modification.
    - More Data Beats a Cleverer Algorithm: New the bottleneck is often the time. As a rule, try the simplest learners first.
    - Learn Many Models, Not Just One: model ensembles (bagging, boosting, stacking) is now standard. In Netflix prize, the winner was stack ensembles of over 100 learners.

* Greg Welch and Gary Bishop. An Introduction to the Kalman Filter. 2006. http://www.cfar.umd.edu/~fer/cmsc828/classes/kalman_intro.pdf
    - State space representation: time update (x_t-1 -> x_t) and measurement (x_t -> z_t), each has white noise N(0, Q) and N(0, R)
    - Filtering: repeat between 1) time update and 2) measurement update
    - Time update: update covariance prediction, and state prediction
    - Measurement update: Compute Kalman gain K_t, estimate a posteriori state, update error covariance
    - Parameter estimation via EM-like algorithm

* Yucheng Low et al. Distributed GraphLab: A Framework for Machine Learning and Data Mining in the Cloud. VLDB 2012  http://vldb.org/pvldb/vol5/p716_yuchenglow_vldb2012.pdf
    - Sequential shared memory abstraction - user defined functions read & write the data in the "scope" - current node + adjacent edges and nodes
    - Optimization: often a large number of parameters converge quickly while others very slowly
    - Users can associate data with nodes AND edges
    - NER example: bipartite graph of noun phrases and context - propagate labels from seeds

* Oren Anava et al. Online Learning for Time Series Prediction. JMLR 2013. http://jmlr.org/proceedings/papers/v30/Anava13.pdf
  - An online learning approach that allows the noise to be arbitrarily or even adversarially generated
  - Goal: minimize the sum of losses over a predefined number of iterations T
  - Applied to loss function (squared loss) and achieved regret (hindersight difference of loss and optimal case loss) bound of O(log^2 (T))
  - We can have a regret as low as the best ARMA(k, q) movel, using only an AR(m+k) model.

* Kiri L. Wagstaff. Machine Learning that Matters. ICML 2012. http://arxiv.org/pdf/1206.4656.pdf
  - Quantitative improvements in performance are rarely accompanied by an assessment of whether those gains matter to the world
  - what is the [machine learning] field's objective function?
  - Most of the papers evaluate only on synthetic or UCI dataset, provide no meaningful interpretation in the field.
  - Performance metrics such as F-measure tell nothing about real world impact (e.g., 99% for poisonous mushroom classification?)
  - Meaningful evaluation metric - dollars saved, lives preserved, time conserved, effort reduced, quality of life increased
  - Ambitious and meaningful challenges (Carbonell 1992) e.g., Outperforming a hand-built NLP system on a task such as translation.
  - Impact Challenges - e.g., A conflict between nations averted through high-quality translation provided by an ML system
  - Obstacles: jargons (e.g., feature extraction -> representation), risks (e.g., concern when relying on machines for decision), complexity (e.g., "ML solutions come packaged in a PhD")

* John Duchi et al. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR 2011.
  - Stochastic optimization which employ proximal functions to control the gradient steps of the algorithm.
  - Infrequent features are highly informative -> give frequently occurring features very low learning rates and infrequent features high learning rates.
  - Specialize to the diagonal case (in the update)
  - Guranteed to suffer asymptotically sub-linear regret
  - Composite mirror descent -> tradeoff between current gradient, regularization, and staying close to the current parameter -> use time-dependent Mahalanobis distance

Domain Adaptation & Transfer Learning
-------------------------------------

* Bollegala et al. Learning to Predict Distributions of Words Across Domains. ACL 2014. http://cgi.csc.liv.ac.uk/~danushka/papers/ACL_2014.pdf
    - Predict distributional representation of words in the target domain
    - Compute SVD lower dimension representation in each domain
    - Compute the mapping (matrix) using Partial Least Squares Regression (PLSR), using "pivots" --- words appearing in the both domains
    - Note: Positive Pointwise Mutual Information (PPMI) (Lin, 1998, Bullinaria and Levy, 2007)

* Mikhail Kozhevnikov and Ivan Titov. Cross-lingual Model Transfer Using Feature Representation Projection. ACL 2014 http://ivan-titov.org/papers/acl14.pdf
    - Learn compact feature representations in either language, then learn mapping (e.g., mapping from prepositions to morphological suffix) from bitext
    - Word representation (Mikolov et al., 2013a), trained by word2vec. Mapping & classifiers learned by PyLearn2 (Goodfellow et al. 2013)
    - Baseline: direct transfer (Universal POS tags and cross-lingual word clusters) and annotation projection
    - Experiments: CONLL 2009 semantic role labeling English->French, English->Czech, result competitive with annotation projection and direct transfer

* Sinno Jialin Pan and Qiang Yang. A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering. http://www1.i2r.a-star.edu.sg/~jspan/publications/TLsurvey_0822.pdf
    - Transfer learning allows the domains, tasks, and distributions used in training and testing to be different
    - Different domains implies either the feature space (e.g., document classification written in different languages) or the distribution is different
    - Inductive TL = the same domains, different tasks
    - Transductive TL = different domains, the same tasks
    - What to transfer: relational-knowledge-transfer: some relationship among the data in the source and target domains is similar

* Hal Daumé III. Frastratingly Easy Domain Adaptation. ACL 2007.  http://aclweb.org/anthology//P/P07/P07-1033.pdf
    - Fully supervised setting: large source domain annotation and small target domain annotation.
    - Baselines: SrcOnly, TgtOnly, All, Weighted, Pred (use source prediction as a feature in target), LinInt (Linear interpolation). Strong ones: Prior (Chela and Acero 2004) and EM (Daume III and Marcu 2006) which learns three separate models (source specific, target specific, and general)
    - Augment features: src_map(x) = <x, x, 0>, tgt_map(x) = <x, 0, x> + kernelized version
    - Experiments in sequence labelins tasks: Outperformed baselines all but brown corpus domain, where SrcOnly wins TgtOnly

Clustering
----------

* Moses Charikar et al. Incremental Clustering and Dynamic Information Retrieval. STOC 1997. http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/papers/Clustering/1997-Incremental-clustering.pdf
    - Incremental clustering: a sequence of points from a metric space is presented: maintain a clustering so as to minimize the maximum cluster diameter (maximum inter-point distance)
    - Doubling algorithm: Step1 merge: build a t-threshold graph and merge a node with its neighbors. Step2 update: assign a new point in one of the current clusters, or to a new cluster
    - Clique Partition algorithm: Step1 merge: build a d-threshold graph and compute minimum clique partition. Step2 update.
    - Dual clustering problem: for a sequence of points, cover each point with a unit ball so as to minimize the number of balls.

* Martin Ester, et al. Incremental Clustering for Mining in a Data Warehousing Environment. VLDB 1998 http://pdf.aminer.org/000/302/472/online_hierarchical_clustering_in_a_data_warehouse_environment.pdf
    - "A data warehouse is a collection of data from multiple sources, integrated into a common repository and extended by summary information (such as aggregate views) for the purpose of analysis"
    - a cluster: density-connected objects which is maximal wrt. density-reachability. noise: objects not contained in any cluster
    - DBSCAN: starts with an arbitrary object p and retrieve all objects density-reachable from p, create a cluster. then visit the next object
    - Incremental DBSCAN: affected objects: Eps-neighborhood and all the objects density reachable from objects in the neighborhood
    - It is sufficient to reapply DBSCAN to the set of affected objects

Classification
--------------

* Naoki Yoshinaga and Masaru Kitsuregawa. A Self-adaptive Classifier for Efficient Text-stream Processing. COLING 2014. http://aclweb.org/anthology/C/C14/C14-1103.pdf
    - Tweets data from 2011 Great East Japan Earthquake using base-phrase chunker (Sassano, 2008) and dependency parser (Sassano, 2004)
    - Reuse the past classification results and their scores. Pre-compute common classification problems and reuse them.
    - Proposed: keep updating common classification problems (with limited number) online
    - Defined two functions for trimming useless problems: least frequently used and least recently used (cf caching)
    - Experiments on chunking and parsing, achieved 3.2 and 5.7 speed-up, respectively.

* Ofer Dekel et al. Large Margin Hierarchical Classification. ICML 2004. http://u.cs.biu.ac.il/~jkeshet/papers/DekelKeSi04.pdf
  - Associate a prototype (vector) to each label, and classify instances according to their similarity (argmax of inner product) to the various prototypes
  - Decompose prototype vector into differences between the parent and the current nodes
  - We require that the margin between the correct and each incorrect labels to be at least the sqrt(dist between them) -> instead minimize convex hinge loss function
  - Batch version: selecting y (system output) to be the label which maximize the hinge loss function + take the average of prototypes.
  - Experiments: Web pages (ODP/DMOZ) speech phoneme classification: always achieved better tree induced errors than "flattened" counterparts.

Neural Networks
---------------

* Lei Jimmy Ba and Rich Caruana. Do Deep Nets Really Need to be Deep? ICLR 2014. http://arxiv.org/pdf/1312.6184v5.pdf
  - Shallow networks (using model compression to mimic) can learn the complex functions previously learned by deep nets and achieve accuracies previously only achievable with deep models.
  - Raw shallow neural net with the same number of parameters 2% less accurate than the DNN
  - SSN-MIMIC - directly trained on the log probability values before the softmax activation.
  - Introducing a bottleneck linear layer with k hidden units between the input and the hidden layer sped up learning
  - Model compression usually works best when the unlabeled data set is much larger than the original train set

Crowdsourcing
-------------

* Rion Snow et al. Cheap and Fast --- But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks. EMNLP 2008. http://web.stanford.edu/~jurafsky/amt.pdf
  - Affect recognition (given a headline, give numeric judgement for six emotions), word similarity, recognizing textual entailment, event temporal ordering, and word sense disambiguatin.
  - AMT allows a requester to restrict which workers are allowed to annotate a task by requiring set of qualifications
  - ITA (inter annotator aggreement) is quite close between experts and turks. Minimum of 4 turks averaging required to beat expert.
  - word similarity: achieved correlation = 0.952 with 10 turns averaging   RTE: achieved 89.7% accuracy by majority vote (expert = 91%)
  - Turk accuracies vary, and a few turks do a large portion of the task. -> bias correction using a small gold standard labels (like weighted voting by each workers log likelihood ratio)
  - bias correction -> RTE +4.0% accuracy increase
