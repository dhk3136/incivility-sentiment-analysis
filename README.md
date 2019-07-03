# Incivility in Public Language Discourse: A Sentiment Analysis
#### Daniel Kim

## Overview:

Sentiment Analysis within Natural Language Processing (NLP) has come a long way since its earlier days both in terms of its methodology and the innovations available to Data Scientists.

However, one problem--a sizeable one--persists. Unlike other classification problems, Sentiment Analysis requires extensive labeling (by humans) so that predictions have an actual target. Human bias and politics are inclusive of this problem, and especially so when the topic is polarizing, negative, and potentially offense. In the case of the organization from where I got my dataset, they conduct occasional spot-checking and follow-ups with self-reported surveys in order to remove as much subjectivity they can from annotated labels.

## Purpose:

To predict the classification of a "toxic" comment from ConversationAI's Wikipedia comments corpus.

## EDA:

The dataset needed a substantial amount of cleaning. Aside from commenters' use of poor syntax and spelling, items such as IP addresses, names and extensions of files and images, (occasional) usernames, multiple escape characters, strangely placed symbols (e.g., =, @, . ., "") appeared for no discernable reason, or in the least, out of place, and several places in the text displayed words stuck together with no white space in between. NaNs were present as were 0 float values for no apparent reason. However, I was glad to see an equal number of rows for the dataset's features.

The dataset initially came with eight features, and I selected one, "toxic," to predict its classification (1 = hit, 0 = miss). Across similar sentiment-based commenter datasets, negative sentiment generally is expressed at a highly reduced frequency compared to its counterparts in the minority class, and this dataset was no exception.

As a result, the dataset was imbalanced. In order to rectify this problem, the original dataset was randomly sampled to produce a more even split between majority and minority classes.


## Method:

NLP Tasks:
- tokenizing: I used NLTK for splitting words and sentences (tokens) into a list; in addition, I created my own script, essentially a series of list comprehensions, in an attempt to clean up what the libraries did not. I also used Keras text preprocessing to see if it had an additive effect in cleaning up text. It improved the data incrementally but not enough to make a difference (although order of execution matters). In addition, I used the following parameters:  
- stemming
- lemmatizing
- stop-words
- n-grams
