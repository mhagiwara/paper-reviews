Machine Learning
================

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

* Sinno Jialin Pan and Qiang Yang. A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering. http://www1.i2r.a-star.edu.sg/~jspan/publications/TLsurvey_0822.pdf
    - Transfer learning allows the domains, tasks, and distributions used in training and testing to be different
    - Different domains implies either the feature space (e.g., document classification written in different languages) or the distribution is different
    - Inductive TL = the same domains, different tasks
    - Transductive TL = different domains, the same tasks
    - What to transfer: relational-knowledge-transfer: some relationship among the data in the source and target domains is similar

* Hal Daumé III. Frastratingly Easy Domain Adaptation. ACL 2007.  http://aclweb.org/anthology//P/P07/P07-1033.pdf
    - Fully supervised setting: large source domain annotation and small target domain annotation.
    - Baselines: SrcOnly, TgtOnly, All, Weighted, Pred (use source prediction as a feature in target), LinInt (Linear interpolation). Strong ones: Prior (Chela and Acero 2004) and EM (Daume III and Marcu 2006) which learns three separate models (source specific, target specific, and general)

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
