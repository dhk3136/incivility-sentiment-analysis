# Incivility in Public Language Discourse: A Sentiment Analysis
#### Daniel Kim

![fighting wallabies](img/toy_boxing_unsplash.jpg)

## Overview:

Sentiment Analysis within Natural Language Processing (NLP) has come a long way since its earlier days both in terms of its state-of-the-art leaps and bounds within this past year with regard to new network architectures (e.g., (transformers)[https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), [transfer learning](https://github.com/huggingface/transformers), and along with these innovations, the ability to perform sentence classification with unparalleled accuracy.  

However, one problem--a sizeable one--persists. Unlike other classification problems, Sentiment Analysis requires extensive labeling (by humans, and now, [ML annotation](https://arxiv.org/pdf/1903.06765.pdf) for the ostensible purpose of anthropomorphic inter-coder agreement. Other machine learning disciplines, of course, face the annotation problem, but it's far more prevalent in NLP (think how summarization might be annotated!) For example, in a Computer Vision classification task, it's a bit easier to identify / fail-to-identify something like a set of objects, cats/dogs, numerical readability or illegibility. Image ID can be bound by boxes--or not. These are all more ocular-centric, but everyday, commonsensical readability tasks. Human bias and politics are inclusive of many problems, NLP or otherwise, and especially so when the topic is polarizing, negative, and potentially offense. Regardless of the data science task, most annotations, especially within larger datasets, are crowdsource using various mechanisms such as Amazon Web Services or Mechanical Turk or self-reported annotation as with the Reddit comments marked by commenters as TL;DR for summarization and what OpenAI utilized to pretrain its infamous GPT-2. The organization publishing this [dataset](https://jigsaw.google.com) conveys that they conduct occasional spot-checking and follow-ups with self-reported surveys in order to remove as much subjectivity they can from annotated labels.

## Purpose:

To predict the classification of reader comments accompanying online articles from Wikipedia's Talk Page site.  

This problem is a multi-categorical classification task containing __six toxic categories__:
```python
Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate
```  
  
## Data
A look at the initial data:  
```python
Total comments = 159571
Total clean comments = 143346
Remaining = 16225
Tagged = 35098
```  

![toxic_class_distribution](img/toxic_distribution.png)  

Got imbalance?  

But wait, why the discrepancy between remaining comments (16225) and those that are tagged (35098)? Why aren't they the same? In a word: multi-tagging. The label annotators followed a methodology in which they were instructed to tag as many categories as fit the criteria. This makes for a more robust data and simultaneously a pain in the butt dataset. Across similar sentiment-based commenter datasets, negative sentiment generally is expressed at a highly reduced frequency compared to its counterparts in the minority class, and this dataset was no exception.

The existing class imbalance is made worse by constraining `clean` comments to unique counts while the rest can double, triple-dip, etc. across the classes. If the same criteria was applied to the `clean` class, the imbalance would be far greater. Thankfully, null values were not present in the data. The first step simply was to add an additional 'clean' feature column for the large amount of untagged text.

There is nearly a 10:1 majority/minority imbalance between clean/toxic comments! Thus, I'll attempt to address the imbalance by either oversampling the minority class or undersampling the aggregate majority class.

![initial dataframe examples](img/toxic_preview.png)  

## Technologies
- Tensorflow, Keras, Scikit-Learn, pandas, Numpy, NLTK, Matplotlib, Seaborn

## Preprocessing
The dataset needed a substantial amount of cleaning. Aside from commenters' use of poor syntax and spelling, items such as IP addresses, usernames, and extensions of files and images, multiple escape characters, strangely placed symbols (e.g., =, @, . ., "") appearing for no discernable reason, or in the least, out of place, and several passages in the text displayed words stuck together without whitespace.  

NLP Tasks:
- Stopwords: 
 - I used NLTK and Sklearn's stopword lists for removing standard and common words detracting from word frequency counts and analysis
- Stemming
 - I took the root of each word to better generalize tokens to the data and so that the algorithm could better find contextual similarity  
- Common contractions
 - this list converts contractions to their formalized roots, similar to stemming for the specific case of contractions without redacting words (e.g., doesn't = does not); again this helps the algorithm to discern cleanly tokenized words and sub-words
- Regex
 - combined [regex](https://regexr.com) searches with the other cleaning tasks into a function

Before:  
```python
hay bitch 

thank you kindly for your advice on my vandalism but if your the dick who removed the thing abouth Berties make up costs Thats true... so ah FUCK  YOU
```

After:
```python
hay bitch thank kindly advice vandalism dick remove thing abouth berties make cost thats true ... ah fuck
```

Not perfect, but inspected samples showed good stopword er, stoppage, [contraction conversion](https://www.kaggle.com/jagangupta/discussion), and tokenization as well. The important thing at this point is not to study the semantic structure of the original sentence but to clean data so that approximation and contextual distance are prepped before fitting to the model. So we need additional feature engineering to address the problem.

## Feature Engineering:
IP address check:
- Leaving identifiable information intact with the comments easily causes data leakage if those IPs in the train set match those in the test set
- Same goes for any identifiable information such as duplicate usernames
After I conducted checks for IPs and usernames, and it turned out both were marginal in effect (e.g., minimal overlapping IPs)

![unique_IP_addresses](img/ip_intersection.png)

Surprising, in the case of usernames, out of a total 239, all were unique except for one duplicate.

I addressed the class imbalance by bootstrapping the minority classes, re-sampling the distribution numerous times to ensure a better total sample. A major check on the success of the bootstrapping attempt included the cross-val results. In other words, I ran the algorithm across both the original, intact data and against the re-sampled effort as a check for conspicuous leakage.  

To my surprise, result were very similar, with and without resampling. This left me suspicious of other data leakage features. As a result, more text preprocessing and feature inspection was required. 

Again, the results held, even increasing the score by small margins. The suspicion primarily arose because the accuracy scores were so high in cross-validation that they were almost unbelievable. After googling for similar classification problems regarding sentiment commentary, I noticed that many other algorithms also performed very well for similar tasks.

After extensive preprocessing and feature engineering, I tokenized my data using Keras. First, I used the sklearn `TfidfVectorizer` to convert my words into values. I also tested this against the Keras version. Once again, results were minimally different. I chose Keras because of its inter-operability with TensorFlow which in turn could increase the speed of loaded arrays as tensors while processing through the model.

## Model  
I used a vanilla LSTM model with modifications. I set my max sequence length at 256 (the length of sentences as input); per usual, any sentence not reaching the max length got padded so the arrays were of the same size.  


Here's what the model looks like:  
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 200)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 200, 128)          2560000   
_________________________________________________________________
lstm_layer (LSTM)            (None, 200, 60)           45360     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 60)                0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 60)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 50)                3050      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 306       
=================================================================
Total params: 2,608,716
Trainable params: 2,608,716
Non-trainable params: 0
_________________________________________________________________  
NB: "None" means that keras is inferring shape which is a nice feature, IMO  


And here's my detailed description of each layer in the network:  
- Start: LSTM input layer: encodes the input per length settings (e.g., max length)  
- -----> Embedding layer: projects coordinates passed from the input layer into a vector space and sets the initial contextualization and distances in space based on similarity and relevance (e.g., dog is positioned close to cat). In doing so, the layer also reduces dimensionality while determining further parameters such as `max_features` or unique words before passing these onto the next layer.  
- -----> The actual LSTM layer is next. Its primary purpose is recursion with respect to major hyperparameters. Here, we can tune for batch size, number of words as input, learning rate, etc. It accepts variables as each word in a sentence. It can take two options--to take the final recurrent output--or to 'roll out' the LSTM so that previous output is passed back into the LSTM layer with each recurrent pass (i.e., total output). In our case, that would be 256 recursions. Significantly, we use TensorFlow to reduce a 3-d tensor into 2-d space to normalize and retain as much of the useful data as possible before passing data into the next layer.
- -----> Our next layer is the CNN-borrowed global [max-pooling layer](https://www.kaggle.com/sbongo/discussion). Because language tasks blow up quite easily in terms of dimension bloat, the max-pooling layer takes the aggregate of each batch to determine maximum values while dropping less relevant features, furthering the process of dimension reduction.  
- -----> Next comes the first of two dropout layers. Essentially, dropouts are meant to increase generalization of data by randomly 'dropping out' nodes in the network. Thus, the next layer must try to make sense of the updated data, kind of like a non-recursive generator-discriminator relationship as seen in GANs.  
- -----> The dropout layer feeds the next dense layer, and the important thing here is what it outputs--a ReLU activation function which in short can be describe as `Activation((Input X Weights) + Bias)`. The ReLU sums the output, and if positive, passes straight to the node input. If negative, it outputs zero. ReLU also greatly reduces the 'vanishing gradient' problem enabling higher performance from networks.
-------> Another dropout layer is fed from the previous dense layer, again to help generalize to the data.
-------> Finally, the data is passed to the final (dense) layer where the data is normalized by a sigmoid function. The takeaway here is that the sigmoid 'squishes' the data into a human-readable scale of 1's and 0's since we are engaged in a classification task.

I've defined loss with binary cross-entropy (again, works well for classification) and utilize the popular Adam optimization technique.

My metrics included accuracy, loss, f1, precision, recall  
*as of keras 2.3, only accuracy and loss are included as built-in functions
*remaining metrics are calculated by scratch within my own function


## RESULTS:
BEST OF 16 EPOCHS:

Epoch 1/10
143613/143613 [==============================] - 901s 6ms/step - loss: 0.0269 - acc: 0.9893 - f1_m: 0.8356 - precision_m: 0.8645 - recall_m: 0.8216 - val_loss: 0.0533 - val_acc: 0.9817 - val_f1_m: 0.7288 - val_precision_m: 0.7577 - val_recall_m: 0.7267

___
TEN EPOCHS:
Batch size = 64
Train on 143613 samples, validate on 15958 samples

Epoch 1/10
143613/143613 [==============================] - 901s 6ms/step - loss: 0.0269 - acc: 0.9893 - f1_m: 0.8356 - precision_m: 0.8645 - recall_m: 0.8216 - val_loss: 0.0533 - val_acc: 0.9817 - val_f1_m: 0.7288 - val_precision_m: 0.7577 - val_recall_m: 0.7267
Epoch 2/10
143613/143613 [==============================] - 889s 6ms/step - loss: 0.0243 - acc: 0.9903 - f1_m: 0.8528 - precision_m: 0.8727 - recall_m: 0.8450 - val_loss: 0.0577 - val_acc: 0.9817 - val_f1_m: 0.7146 - val_precision_m: 0.7830 - val_recall_m: 0.6791
Epoch 3/10
143613/143613 [==============================] - 880s 6ms/step - loss: 0.0218 - acc: 0.9913 - f1_m: 0.8689 - precision_m: 0.8838 - recall_m: 0.8653 - val_loss: 0.0644 - val_acc: 0.9817 - val_f1_m: 0.7237 - val_precision_m: 0.7733 - val_recall_m: 0.7030
Epoch 4/10
143613/143613 [==============================] - 885s 6ms/step - loss: 0.0192 - acc: 0.9923 - f1_m: 0.8882 - precision_m: 0.8985 - recall_m: 0.8871 - val_loss: 0.0707 - val_acc: 0.9811 - val_f1_m: 0.7088 - val_precision_m: 0.7588 - val_recall_m: 0.6881
Epoch 5/10
143613/143613 [==============================] - 891s 6ms/step - loss: 0.0174 - acc: 0.9931 - f1_m: 0.8990 - precision_m: 0.9071 - recall_m: 0.8997 - val_loss: 0.0744 - val_acc: 0.9801 - val_f1_m: 0.7063 - val_precision_m: 0.7333 - val_recall_m: 0.7073
Epoch 6/10
143613/143613 [==============================] - 883s 6ms/step - loss: 0.0158 - acc: 0.9938 - f1_m: 0.9097 - precision_m: 0.9155 - recall_m: 0.9118 - val_loss: 0.0789 - val_acc: 0.9800 - val_f1_m: 0.7056 - val_precision_m: 0.7283 - val_recall_m: 0.7117

___
FIVE EPOCHS:
Batch size = 32
Train on 143613 samples, validate on 15958 samples

Epoch 1/5
143613/143613 [==============================] - 1889s 13ms/step

TRAIN
loss: 0.0712
acc: 0.9767
f1_m: 0.5244
precision_m: 0.6338
recall_m: 0.4875

VAL
val_loss: 0.0494
val_acc: 0.9817
val_f1_m: 0.6491
val_precision_m: 0.7695
val_recall_m: 0.5995

Epoch 2/5
143613/143613 [==============================] - 1885s 13ms/step

TRAIN
loss: 0.0460
acc: 0.9829
f1_m: 0.6673
precision_m: 0.7565
recall_m: 0.6321 - 

VALs
val_loss: 0.0488
val_acc: 0.9822
val_f1_m: 0.6868
val_precision_m: 0.7388
val_recall_m: 0.6818

Epoch 3/5
143613/143613 [==============================] - 1891s 13ms/step

TRAIN
loss: 0.0409
acc: 0.9844
f1_m: 0.6995
precision_m: 0.7754
recall_m: 0.6700

VAL
val_loss: 0.0476
val_acc: 0.9825
val_f1_m: 0.6915
val_precision_m: 0.7416
val_recall_m: 0.6846

Epoch 4/5
143613/143613 [==============================] - 1863s 13ms/step

TRAIN
loss: 0.0364
acc: 0.9858
f1_m: 0.7272
precision_m: 0.7873
recall_m: 0.7056

VAL
val_loss: 0.0476
val_acc: 0.9825
val_f1_m: 0.6825
val_precision_m: 0.7554
val_recall_m: 0.6585

Epoch 5/5
143613/143613 [==============================] - 1878s 13ms/step

TRAIN
loss: 0.0321
acc: 0.9873
f1_m: 0.7597
precision_m: 0.8054
recall_m: 0.7480

VAL
val_loss: 0.0511
val_acc: 0.9823
val_f1_m: 0.6788
val_precision_m: 0.7425
val_recall_m: 0.6640

<keras.callbacks.History at 0x7f54e80a82b0>



ONE EPOCH:
Batch size = 32
Train on 143613 samples, validate on 15958 samples

Epoch 1/1
143613/143613 [==============================] - 1906s 13ms/step

TRAIN
loss: 0.0680
acc: 0.9780
f1_m: 0.5354
precision_m: 0.6264
recall_m: 0.5034

VAL
val_loss: 0.0524
val_acc: 0.9813
val_f1_m: 0.6072
val_precision_m: 0.7844
val_recall_m: 0.5287

<keras.callbacks.History at 0x7f55027e2c50>



