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
