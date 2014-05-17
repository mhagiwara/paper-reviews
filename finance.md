Finance
=======

* Helen Susannah Moat et al. Quantifying Wikipedia Usage Patterns Before Stock Market Moves, Scientific Reports, 2013. http://www.nature.com/srep/2013/130508/srep01801/pdf/srep01801.pdf
    - Used views and edits on financially related Wikipedia pages, weekly changes over past N week average
    - Sell (DJIA) on the first trading day of a week if the change is negative, buy oterwise, and close the position on the following week.
    - Views on DJIA companies page -> +0.5 return (std. dev. of random strategy), views on financial topics (+1.10 return), views on actors & filmmakers (no significant difference)

* Tobias Preis et al. Quantifying Trading Behavior in Financial Markets Using Google Trends, Scientific Reports, 2013. http://www.nature.com/srep/2013/130425/srep01684/pdf/srep01684.pdf
    - Financial market drops are preceded by periods of invester concern, i.e., Internet search for market information
    - Used weekly relative search volume change, sell DJIA if the change is positive and cloase the position one week later. Buy otherwise.
    - Strategy using the term "debt" -> 326% return (2.31 std. dev. of random strategy), "stocks" (2.21 std. dev.), portfolio (1.69) (mean of delta = 1..6 weeks to average). The more finacially related, the better the performance (compared with relative frequency to Financial Times)

* Clinton P. McCully et al. Comparing the Consumer Price Index and the Personal Consumption Expenditures Price Index. 2007. http://www.bea.gov/scb/pdf/2007/11%20November/1107_cpipce.pdf
    - Comparison of CPI (by BLS - Bureau of Labor Statistics) and PCE (by BEA - Bureau of Economic Analysis). CPI grew 0.4 point / year than PCE
    - Formula effect: CPI (modified Laspeyres formula) vs PCE (Fisher-Ideal formula)
    - Weight effect: CPI (based on household survey) vs PCE (business survey)
    - Scope effect: Difference in scope items: CPI (out-of-pocket expenditures of all urban households) vs PCE (services purchased by households and nonprofit institutions)
    - Other effect

* Hyunyoung Choi and Hal Varian. Predicting the Present with Google Trends. 2009. http://static.googleusercontent.com/media/www.google.com/en/us/googleblogs/pdfs/google_predicting_the_present.pdf
    - Predict the present (metrics which have delays until announcements) using Google Trends (Index: percentage deviation from January 1 2004)
    - Ford sales: seasonal autoregressive (AR) model, including the sales 12 months ago, with query index for 'Ford.' -> MAE reduction 3%
    - Automotive sales: included Google trends category index, makes
    - Home sales: "Real Estate Agencies" -> best predictor of house sales
    - Travel: Visits to Hong Kong from other countreis, based on 'Hong Kong' subcategory under Vacation Destinations. A dummy variable to explain Beijing Olympics negative effect.

* Steven L. Scott, Hal Varian. Bayesian Variable Selection for Nowcasting Economic Time Series. 2012. http://people.ischool.berkeley.edu/~hal/Papers/2012/fat.pdf
    - "Fat regressio" - the number of possible predictors exceeds the number of observations
    - Kalman filter for time-series prediction: "Log-linear trend model with regression" (level + trend + regressors + error)
    - Spike and slab variable selection (for the details, better to follow George and McCulloch 1997: http://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf) + Bayesian model averaging
    - Experiments: Nowcasting consumer sentiment based on 151 Google Search Insights (inclusion prob. Financial.Planning, Investing, etc.), one step ahead prediction MAE = 4.5% + gun sales
