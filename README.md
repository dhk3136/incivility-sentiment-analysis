# Incivility in Public Language Discourse: A Sentiment Analysis
#### Daniel Kim

![fighting wallabies](img/toy_boxing_unsplash.jpg)

## Overview:

Sentiment Analysis within Natural Language Processing (NLP) has come a long way since its earlier days both in terms of its methodology and the innovations available to Data Scientists.

However, one problem--a sizeable one--persists. Unlike other classification problems, Sentiment Analysis requires extensive labeling (by humans) so that predictions have an actual target. Human bias and politics are inclusive of this problem, and especially so when the topic is polarizing, negative, and potentially offense. In the case of the organization from where I got my dataset, they conduct occasional spot-checking and follow-ups with self-reported surveys in order to remove as much subjectivity they can from annotated labels.

## Purpose:

To predict the classification of reader comments accompanying online articles from the *New York Times*. This is a multi-categorical classification task using six different "toxic" categories:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

A look at the initial data:  
![initial dataframe examples](img/toxic_preview.png)  


## Technologies
- Tensorflow, Keras, Scikit-Learn, pandas, Numpy, NLTK, Matplotlib, Seaborn

## Preprocessing
The dataset needed a substantial amount of cleaning. Aside from commenters' use of poor syntax and spelling, items such as IP addresses, names and extensions of files and images, (occasional) usernames, multiple escape characters, strangely placed symbols (e.g., =, @, . ., "") appeared for no discernable reason, or in the least, out of place, and several places in the text displayed words stuck together with no white space in between. NaNs were present as were 0 float values for no apparent reason. However, I was glad to see an equal number of rows for the dataset's features.

The dataset initially came with eight features, and I selected one, "toxic," to predict its classification (1 = hit, 0 = miss). Across similar sentiment-based commenter datasets, negative sentiment generally is expressed at a highly reduced frequency compared to its counterparts in the minority class, and this dataset was no exception.

As a result, the dataset was imbalanced. In order to rectify this problem, the original dataset was randomly sampled to produce a more even split between majority and minority classes.

NLP Tasks:
- stopwords: 
I used NLTK and Sklearn's stopword lists for removing standard and common words detracting from word frequency counts and analysis
- stemming
I took the root of each word to better generalize tokens to the data and so that the algorithm could better find contextual similarity  
List of common contractions
- this list changes contractions to their formalized roots, e.g., doesn't = does not; again this helps the algorithm to discern cleanly tokenized words and sub-words

n-grams (2)

More preprocessing:
IP address check:
- Leaving identifiable information intact with the comments easily causes data leakage if those IPs in the train set match those in the test set
- Same goes for any identifiable information
- So I conducted checks for IPs and usernames, and it turned out both were marginal in effect (e.g., only six overlapping IPs)
Regex Condition:


---
Results:
Class: toxic  

Precision
0.86      

Recall
0.74       

f1-score  
0.79  


__Redux, Next Steps__
- First, choose another dataset--or at least one that's far more balance in terms of classes
- Pay more attention to term frequency importance and relationship to predictive score
- Understand and know how to troubleshoot your dataset when your accuracy score is close to perfect
- Continue work on the project as something to show others as a way of showing NLP potential projects