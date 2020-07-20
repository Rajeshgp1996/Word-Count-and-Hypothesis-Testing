# Word-Count-and-Hypothesis-Testing
Task Requirements: Your objective is to compute the correlation between each of the 1,000 most common words (case insensitive) across all reviews with the rating score for the reviews, controlling for whether the review was verified or not.


First you must figure out which of all the possible words are the most common. You should consider anything matched by the following regular expression as a word:
  r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))'


Then, you must figure out how common each of the 1k words occurs in each review. Record the relative frequency = (total count of word) / (total number of words in review). Note most words will occur 0 times in most reviews.


Finally, compute the relationship. Each review represents an observation, and each of the 1,000 words is essentially a hypothesis. Thus, you will have over 1k linear regressions to run representing 1,000 hypotheses to test. Further, you will need to run the tests without and using "verified" as a control (simply including it as an additional covariate in your linear regression as either 0 or 1).  You must use Spark such that each of these correlations (i.e. standardized linear regression) can be run in parallel -- organize the data such that each record contains all data needed for a single word (i.e. all relative frequencies as well as corresponding ratings and verified indicators for each review), and then use a map to compute the correlation values for each.
You don't have to worry about duplicate reviews for this one. Assume each review is a separate review.


You must choose how to handle the outcome and control data effectively. You must implement standardized multiple linear regression yourself -- it is just a line or two of matrix operations (using Numpy is fine).  Finally, you must compute p values for each of the top 20 most positively and negatively correlated words and apply the Bonferroni multi-test correction.All together, your code should run in less than 8 minutes on the provided data. Your solution should be scalable, such that one simply needs to add more nodes to the cluster to handle 10x or 100x the data size.


Other than the above, you are free to design what you feel is the most efficient and effective solution. Based on feedback, the instructor may add or modify restrictions (in minor ways) up to 3 days before the submission.  You are free to use broadcast or aggregator variables in ways that make sense and fit in memory -- typically 1 row or 1 column by itself will fit in memory but not an entire matrix (at least for the larger dataset).


Output: Your code should output four lists of results. For each word, output the triple: (“word”, beta_value, multi-test corrected (for 1000 hypothesis) p-value)

1) The top 20 word positively correlated with rating

2) The top 20 word negatively correlated  with rating

3) The top 20 words positively related to rating, controlling for verified

4) The top 20 words negatively related to rating, controlling for verified
