Human Learning
==============

General
-------

* K. Andres Ericsson et al. The Role of Deliberate Practice in the Acquisition of Expert Performance. Psychological Review. 1993. http://www.mockingbirdeducation.net/uploads/5/4/0/7/5407628/ericsson_1993.pdf
  - Innate (genetic) influences had been viewed as deterministic and limiting factors for expert performance
  - Relation of IQ to exceptional performance (music, chess) as well as occupational success, is rather weak (0.19 on average)
  - unique environmental conditions and parental support, rather than talent, determins the initial onset of training and ultimate performance.
  - Sports (and other skill) levels improved in the past century e.g., Tchaikovsky's violin concert refused by two violinist but now standard
  - Maximum level of performance is not attained automatically as function of extended experience, but as a result of deliberate efforts to improve.
  - condition for improvement: subjects' motivation to attend to the task and exert effort to improve their performance.
  - in the absence of adequate feedback, efficient learning is impossible and improvement only minimal even for highly motivated subjects
  - Deliberate practice: (ideally individual) explicit instruction and supervision by a teacher should be given to assure effective learning
  - work - far from optimal in terms of learning opportunities (discourages learning and acquisition of new and possibly better methods)
  - Difference between experts and less skilled subject reflects qualitative differences in the organization of knowledge and its representation
  - Excerpt: https://psy.fsu.edu/faculty/ericsson/ericsson.exp.perf.html

* Jeffrey D Karpicke and Henry L Roediger III. The Critical Importance of Retrieval for Learning. http://learninglab.psych.purdue.edu/downloads/2008_Karpicke_Roediger_Science.pdf
  - Learn foreign language vocabulary pairs (e.g., Swahili-English mashua-boat), -> +study+test, -study+test, +study-test, -study-test -> final test
  - Proportion recalled = +study+test (80%), -study+test (80%), +study-test (36%), -study-test (33%)
  - Repeated testing (retrieval of learned information) produced a large positive effect, while repeated studying has no benefit.
  - Wrong conventional wisdom: once information can be recalled from memory it can be dropped from further practice
  - See Testing effect http://en.wikipedia.org/wiki/Testing_effect


* Allen Newell and Paul S. Rosenbloom. Machanisms of Skill Acquisition and the Law of Practice. Cognitive Skills and their Acquisition. 1980 http://repository.cmu.edu/cgi/viewcontent.cgi?article=3420&context=compsci&sei-redir=1&referer=http%3A%2F%2Fwww.google.com%2Furl%3Fq%3Dhttp%253A%252F%252Frepository.cmu.edu%252Fcgi%252Fviewcontent.cgi%253Farticle%253D3420%2526context%253Dcompsci%26sa%3DD%26sntz%3D1%26usg%3DAFQjCNEIrb1-kozWZZnflhBUQsjFa_1lVw#search=%22http%3A%2F%2Frepository.cmu.edu%2Fcgi%2Fviewcontent.cgi%3Farticle%3D3420%26context%3Dcompsci%22
  - The ubiquitous "power law of practice" (log-log linear learning law) - holds for practice learning of all kinds
    - Show 12 examples, e.g., including mirror-tracing (Snoddy 1926) - time + errors decrease as trials increase ( T = BN^(-alpha) where alpha < 1 is the learning rate)
  - Power law, exponential, and generalized curves (T = B*exp(-alpha * N ^ (1 - beta)))
  - The chunking theory of learning (e.g., Seibel's light-button resopnse task, chunk = span of multiple lights) -> larger chunks are exponentially less frequent
  - Data fits the family of generalized power law (not the simple power law)


- Hao Cen et al. Learning Factor Analysis - A General Method for Cognitive Model Evaluation and Improvement.  Intelligent Tutoring Systems 8th International Conference 2005. http://www.ml.cmu.edu/research/dap-papers/cen_kddpaper2.pdf
  - Complex skills do not show smooth, declining learning curves (Corbett and Anderson 1995) -> decompose into two skills with smoother learning curves
  - Learning Factor Analysis (LFA) - intercept for each student, slope parameter for each skill (not on students), intercept for each skill
  - Difficulty factors - hidden property of a problem that causes some difficulty
  - A* search for best model - searches the best knowledge representation by splitting skills (in original skills x difficulty factors) based on AIC and BIC
  - Experiment with area unit of the Geometry Tutor - analysis of intercept and slope of learning curves
  - Splitting and merging of skills based on AIC, BIC

- Kenneth Koedinger et al. Using data-driven discovery of better student models to improve student learning. AIED 2013 http://www.learnlab.org/research/wiki/images/e/eb/Koedinger-et-al-aied2013.pdf
  - Use learning curves (flat and rigged) to identify poorly taught KCs (e.g., "compose-by-addition" -> +decompose, +subtract)
  - Added scafolding questions


- Tanja Täsker, et al. Different parameters - same prediction: An analysis of learning curves. EDM 2014 http://educationaldatamining.org/EDM2014/uploads/procs2014/long%20papers/52_EDM-2014-Full.pdf
  - AFM (additive factor model) - a logistic regression model to fit student proficiency, skill difficulty, learning rate - suffers from student attrition (well performing students need few opportunities to master a skill, while weaker students remain)
  - LG (learning gain model) - normalizes trial to [0, 1] (mastery)
  - DIS (disaggregated model) - group students by the number of opportunities needed to first master a skill.
  - BKT (Bayesian knowledge tracing) - HMM (hidden state = mastery, emission = correct/incorrect)
  - Experiment with Calcularis (ITS) - residual analysis: overestimate the outcome of badly performing student and underestimate the outcome for well performing students
  - Re-test prediction - AFM and LG comparable in terms of RMSE and AUC
  - Generalization to new students - LG > AFM - prediction accuracy is quite similar

Language Learning
-----------------
- Sze-Meng Jojo Wong and Mark Dras. Parser Features for Sentence Grammaticality Classification. 2010 http://www.aclweb.org/anthology/U10-1011
  - Existing approach: parser byproduct (probability, number of partial parse trees, parsing duration)
  - Ungrammatical corpora obtained from noisy version of WSJ created by Foster (2007) (Noisy WSJ and Nosier WSJ)
  - Trained three Stanford Parser models - PureParser, NoisyParser, and MixedParser (mixing Pure WSJ and Noisy WSJ)

- Gwen Frishkoff et al. A Model-based Approach to Second-Language Learning of Grammatical Constructions. Proceedings of the 30th Annual Meeting of the Cognitive Science Society (2008) http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.490.7689&rep=rep1&type=pdf
  - Dative alternation (give someone the book vs give the book to someone)
  - Four factor model (givenness, definiteness, pronominality of the recipient noun phrase, pronominality of the theme), obtained by 14-factor PCA analysis
  - Examples with easy/prototypical factors are learned faster
  - Explicit presentation of the linguistic rules boosted performance

- Trude Heift and Mathias Shulze. Student Modeling and ab initio Language Learning, System, 2003. http://arts.uwaterloo.ca/~mschulze/papers/system_2003.pdf
  - German tutoring system with a parser and a grammar to analyzes sentences from students and detects grammatical errors, with error specific messages (e.g., determiner / noun aggreement)
  - Student model keeps track of performance on "phrase descriptor" - grammatical phenomena (e.g., agreement, modals, verb particles) represented as [descriptor, correct/error/absent]
  - Show 'groups' of phrase descriptors, strong and weak skills, provide practice for weak skills
