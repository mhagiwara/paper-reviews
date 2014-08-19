Finance
=======

* Allan Borodin et al. Can We Learn to Beat the Best Stock. Journal of Artificial Intelligence Research. 2004. http://arxiv.org/pdf/1107.0036.pdf
    - (Borodin et al. 2000) prediction component of Lempel-Ziv (LZ) losless compression algorithm
    - ANTICOR: money i->i if the growth rate for stock i exceeds that of stock j, and stock j will start to emulate the past growth of stock i in the near future
    - For most of the window sizes and buy-and-hold(ANTICORE_w), it beats the best stocks and markets
    - Using ANTICORE to actively trade ANTICORE_w's (better performance), clearly beats the market when the transaction fee is less than 0.4%


* Helen Susannah Moat et al. Quantifying Wikipedia Usage Patterns Before Stock Market Moves, Scientific Reports, 2013. http://www.nature.com/srep/2013/130508/srep01801/pdf/srep01801.pdf
    - Used views and edits on financially related Wikipedia pages, weekly changes over past N week average
    - Sell (DJIA) on the first trading day of a week if the change is negative, buy oterwise, and close the position on the following week.
    - Views on DJIA companies page -> +0.5 return (std. dev. of random strategy), views on financial topics (+1.10 return), views on actors & filmmakers (no significant difference)

* Tobias Preis et al. Quantifying Trading Behavior in Financial Markets Using Google Trends, Scientific Reports, 2013. http://www.nature.com/srep/2013/130425/srep01684/pdf/srep01684.pdf
    - Financial market drops are preceded by periods of invester concern, i.e., Internet search for market information
    - Used weekly relative search volume change, sell DJIA if the change is positive and cloase the position one week later. Buy otherwise.
    - Strategy using the term "debt" -> 326% return (2.31 std. dev. of random strategy), "stocks" (2.21 std. dev.), portfolio (1.69) (mean of delta = 1..6 weeks to average). The more finacially related, the better the performance (compared with relative frequency to Financial Times)

* Hyunyoung Choi and Hal Varian. Predicting the Present with Google Trends. 2009. http://static.googleusercontent.com/media/www.google.com/en/us/googleblogs/pdfs/google_predicting_the_present.pdf
    - Predict the present (metrics which have delays until announcements) using Google Trends (Index: percentage deviation from January 1 2004)
    - Ford sales: seasonal autoregressive (AR) model, including the sales 12 months ago, with query index for 'Ford.' -> MAE reduction 3%
    - Automotive sales: included Google trends category index, makes
    - Home sales: "Real Estate Agencies" -> best predictor of house sales
    - Travel: Visits to Hong Kong from other countries, based on 'Hong Kong' subcategory under Vacation Destinations. A dummy variable to explain Beijing Olympics negative effect.

* Steven L. Scott, Hal Varian. Bayesian Variable Selection for Nowcasting Economic Time Series. 2012. http://people.ischool.berkeley.edu/~hal/Papers/2012/fat.pdf
    - "Fat regressio" - the number of possible predictors exceeds the number of observations
    - Kalman filter for time-series prediction: "Log-linear trend model with regression" (level + trend + regressors + error)
    - Spike and slab variable selection (for the details, better to follow George and McCulloch 1997: http://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf) + Bayesian model averaging
    - Experiments: Nowcasting consumer sentiment based on 151 Google Search Insights (inclusion prob. Financial.Planning, Investing, etc.), one step ahead prediction MAE = 4.5% + gun sales

* Heeyoung Lee, et al. On the Importance of Text Analysis for Stock Price Prediction. LREC 2014. http://web.stanford.edu/~jurafsky/pubs/lrec2014_stocks.pdf
    - 8-K financial report (filled by public company when major events occur) (corpus is public: http://nlp.stanford.edu/pubs/stock-event.html), from 2002 to 2012
    - Stock price: relative change from S&P500 index during the time period (between the document release and market open), three class (UP/DOWN/STAY) classification
    - Features: Earnings surprise (diff. of consensus EPS and actual EPS), recent movements, VIX, event category, unigram, NMF vector of unigrams
    - Classified by RandomForest. Earnings suprise the single most effective feature.
    - "Stock market is highly sensitive to company reports in the short term, but more sensitive to third-party perspectives in the longer term."

* Kogan et al. Predicting Risk from Financial Reports with Regression. NAACL 2009. http://www.cs.cmu.edu/~nasmith/papers/kogan+levin+routledge+sagi+smith.naacl09.pdf
  - Simple bag of ngrams representation can (when combined) beat baseline (prior volatility is a very good predictor of future volatility)
  - "It is, by now, received wisdom in the field of economics that predicting a stock's performance, based on easily accessible public information, is difficult."
  - Support Vector Regression (SVR)
  - Form 10-K from SEC Website. "Management's discussion and analysis of financial conditions and results of operations (MD&A)"
  - Feature: LOG1P (log of 1+tf)
  - Temporal changes in financial reporting make training data selection non-trivial. (More accurate after 2002 Sarbanes-Oxley Act)
  - Company delisting prediction from 10-K report: 75% precision and 10% recall.

Economics
=========

* Archak et al. Show me the Money! Deriving the Pricing Power of Product Features by Mining Consumer Reviews. KDD 2007. http://pages.stern.nyu.edu/~aghose/kdd2007.pdf
  - Incorporate qualitative features in a hedonic (aggregation of individual characteristics)-like framework
  - Collect adjectives (good, amazing, bad) + nouns (lens, image quality)
  - Tensor product of evaluation and feature spaces. Linear regression of log of demand = product factor + log of price + error
  - Reduce the dimensionality of by rank-1 approximation of the above product
  - Collected reviews and ranking data from Amazon, converted ranking to demand levels
  - Seemingly positive evaluations like "decent quality" "nice/fine camera" hurt sales -> positive reviews are superlatives. Product got bad reviews disappear quickly.
  - "In general, the reviews that appear on Amazon are positive, especially for products with large number of posted reviews."

* Clinton P. McCully et al. Comparing the Consumer Price Index and the Personal Consumption Expenditures Price Index. 2007. http://www.bea.gov/scb/pdf/2007/11%20November/1107_cpipce.pdf
  - Comparison of CPI (by BLS - Bureau of Labor Statistics) and PCE (by BEA - Bureau of Economic Analysis). CPI grew 0.4 point / year than PCE
  - Formula effect: CPI (modified Laspeyres formula) vs PCE (Fisher-Ideal formula)
  - Weight effect: CPI (based on household survey) vs PCE (business survey)
  - Scope effect: Difference in scope items: CPI (out-of-pocket expenditures of all urban households) vs PCE (services purchased by households and nonprofit institutions)
  - Other effect

* Anindya Ghose and Arun Sundararajan. Evaluating Pricing Strategy Using e-Commerce Data: Evidence and Estimation Challenges. http://arxiv.org/pdf/math/0609170.pdf Statistical Science 2006
  - Price-discriminate strategies: nonlinear pricing, bundling, mixed bundling, versioning, succeesive generations, ...
  - Data: 330 products on Amazon randomly selected from busines and productivity, security and utilities, graphics and development and operation systems
  - Demand estimation from rank - ranks follow Pareto distribution (i.e., power law) log[Q] = \alpha + \beta log[rank] -> detect "spikes" of hourly sales rank
  - Own-price elasticity = diff demand per diff own price, cross-price elasticity = diff demand per diff another product's price.
  - Cost estimation using Hausman (1994)
  - Optimality of pricing = partial derivative of profit wrt price = 0
  - Consumer demand for Microsoft Office Professional is very sensitive to the price of Microsoft Office Standard
