Machine Learning
================

* Greg Welch and Gary Bishop. An Introduction to the Kalman Filter. 2006.
    - State space representation: time update (x_t-1 -> x_t) and measurement (x_t -> z_t), each has white noise N(0, Q) and N(0, R)
    - Filtering: repeat between 1) time update and 2) measurement update
    - Time update: update covariance prediction, and state prediction
    - Measurement update: Compute Kalman gain K_t, estimate a posteriori state, update error covariance