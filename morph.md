Morphology (Word Segmentation & PoS Tagging)
============================================

PoS Tagging
-----------

* Dipanjan Das and Slav Petrov. Unsupervised Part-of-Speech Tagging with Bilingual Graph-Based Projections http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/37071.pdf
    - Build a PoS tagger (on universal PoS tag set) for resource scarce languages (using parallel data and English supervised PoS tagger)
    - Projection and label propagation on a bilingual graph of trigrams (in foreign language) and word types (in English), using similarity statistics and word alignment
    - PoS tagger was trained on feature-based HMM (Berg-Kirkpatrick et al. 2010) using tag distributions as features
    - Experiment: used 8 Indo-European languages from Europarl and ODS UN dataset
    - Full model achieved 10.4 point increase from the state-of-the art and 16.7% from vanilla HMM
