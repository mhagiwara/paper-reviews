Named Entity Recognition
========================


* Alexander E. Richman and Patrick Schone. Mining Wiki Resources for Multilingual Named Entity Recognition, ACL 2008. http://www.mt-archive.info/ACL-2008-Richman.pdf
    - Use categories to classifiy English pages, and map NE types via interwiki links. Use this as the type of a wikilink (intra-language link to another article). Foreign language knowledge is not requried.
    - If a foreign page doesn't have a link to its English translation, use cateogory translation to classify types
    - Trained a NER system (PhoenixIDF), tested on ACE 2007 (newswire) and Wikipedia held-out data, with approx. 0.82 ~ 0.84 F-value in Spanish, French, and other languages

* Sungchul Kim, Kristina Toutanova, and Hwanjo Yu. Multilingual Named Entity Recognition using Parallel Data and Metadata, ACL 2012. http://www.newdesign.aclweb.org/anthology-new/P/P12/P12-1073.pdf
    - Wiki-tagger: Built local and global taggers based on article categorization (as in Richman and Schone 2008)
    - Mapping-based taggger: a ranking (log-linear) model from English entities to foreign entity spans (Feng et al. 2004), trained on 100 parallel sentences with NEs and alignments
    - Combination of these two via semi-Markov CRF (Sarawagi, Cohen 2005) outperforms individual models (~91% F-value in Korean and Bulgarian)
