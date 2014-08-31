Logic
=====

* Hoifung Poon and Pedro Domingos. Joint Inference in Information Extraction. AAAI 2007. http://www.aaai.org/Papers/AAAI/2007/AAAI07-145.pdf
    - Joint segmentation and entity resolution of citation matching (from CiteSeer and Cora)
    - Markov logic (Richardson & Domingos 2006) Alchemy package, MC-SAT algorithm ("slice sampling" MCMC) + voted perceptron algorithm
    - Citation matching: title, author, venue: MLN: token(t-token, i-pos, c-citation), InField, SameCitation
    - Segmentation: stickey HMM model with explicit punctuation modeling
    - SimilarTitle, SimilarVenue, SimilarAuthor "bandwidth" between segmentation and entity matching
    - Result: CiteSeer cluster recall ~96%, segmentation F1: ~95%
