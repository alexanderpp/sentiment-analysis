# Sentiment Analysis

## Intro
The idea is to write different algorithms using different machine learning approaches in order to compete them in solving sentiment analysis problem.

## Data

### Dataset 1

The data ised for testing and training is part Keggle competiton called "Bag of Words Meets Bags of Popcorn".
It consists of 25,000 IMDB movie reviews, specially selected for sentiment analysis. 
The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. 
No individual movie has more than 30 reviews.
For all of the data is spit into training data and test data, using the following proportions - 70% for training and 30% for testing.

Link: https://www.kaggle.com/c/word2vec-nlp-tutorial/data


### Dataset 2

Another dataset which contains IMDB movie reviews, shared by [ACL](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib).
It contains 25,000 equally split samples for training and another 25,000 (again, equially spilt), for testing. 
Both batches of data are labeled.

Link: http://ai.stanford.edu/~amaas/data/sentiment/


## Processing of the data


### Method 1 - Term Frequency Representation
First the data is converted to indexes. All of the different words are collected into a list, where each of them is given an index.
Then after we have words and indexes, it is time to revisit every sample and create a vector for it.
The vector has the dimensions equal to the total count of words in the whole dictionary.
On every position of the vector (coresponding to the index of word) is the count of times this word is found in the current sample text.

Example:

If we these sentences:

```
"This is sentence number one."
"This sentence is the second sentence."
```

Now that we have our samples we can create our bag of words, which looks like this:

```
["this", "is", "sentence", "number", "one", "the", "second"]
```

Now we can replace the words in our sentences and make them look like vectors of integers:

```
Sentence 1: [1, 1, 1, 1, 1, 0, 0]

Sentence 2: [1, 1, 2, 0, 0, 1, 1]
```

### Method 2 - One-hot Representation

This method is very similar to the first one. Again first the data is converted to indexes. 
All of the different words are collected into a list, where each of them is given an index.
Then after we have words and indexes, it is time to revisit every sample and create a vector for it.
The vector has the dimensions equal to count of words in the sentence.
The index zero is reserved for unknown words.

Example:

If we these sentences:

```
"This is sentence number one."
"This sentence is the second sentence."
```

Now that we have our samples we can create our bag of words, which looks like this:

```
["this", "is", "sentence", "number", "one", "the", "second"]
```

Now we can replace the words in our sentences and make them look like vectors of integers:

```
Sentence 1: [1, 2, 3, 4, 5]

Sentence 2: [1, 3, 2, 6, 7, 3]
```

### Method 3 - Using Word2vec

Word2vec is a newural net which processes text. 
The input of this net is a set of text examples and the output is a set of vectors, where every word is represented as a multidimnsional vector.

![Word2vec](https://1.bp.blogspot.com/-Q7F8ulD6fC0/UgvnVCSGmXI/AAAAAAAAAbg/MCWLTYBufhs/s1600/image00.gif =300x)
*The image [article](https://opensource.googleblog.com/2013/08/learning-meaning-behind-words.html) from Google Opensource Blog*


In the current case, I am using the pretrained Word2vec model provided by Google.
It is trained using Google News.
The output is that every word is translated into a vector with 300 dimensions.
This provides a solid ground for the LSTM model to learn.

## Algorithms

### Naive Bayes Classifier

This is one of the most simple classifiers that are commonly used.
However, still it produces very good results and is able to classify with great accuracy.


Naive Bayes is part of the so called probabilistic classification algorithms (or classifiers).
It is named Bayes, because it is based on and applies Bayes' theorem which looks like ths:

```
P(A|B) = (P(A|B) P(A)) / P(B)
```

Where:
* P(A) is the probability of observing A
* P(B) is the probability of observing B
* P(A|B) is the probability of observing A given that B is present (true)
* P(B|A) is the probability of observing B given that A is present (true)


This classifier is called Naive, because the theorem is used with strong (naive) independence assumption between the fatures.


### XGBoost

... TODO ...

### Multi-Layer Perceptron Neural Network


### Long Short-Term Memory Neural Network

... TODO ...

## Current Results
<<<<<<< HEAD

| Algorithm                             | Accuracy |
| ------------------------------------- | -------- |
| Naive Bayes Classifier                |  77.00%  |
| XGBoost                               |  86.08%  |
| Multi-Layer Perceptron Neural Network |  87.23%  |
| Long Short-Term Memory Neural Network |  83.45%  |
=======
>>>>>>> 41aeca1f318d8c27f974af9612ad97f5a3d3d5cb

### Results using the first dataset

| Algorithm                              | Accuracy |
| -------------------------------------- | -------- |
| Naive Bayes Classifier                 |  77.00%  |
| XGBoost                                |  86.08%  |
| Multi-Layer Perceptron Neural Network  |  87.23%  |
| Long Short-Term Memory Neural Network* |  83.45%  |


*\*The data was represented using One-hot encoding*


### Results using the second dataset

| Algorithm                              | Accuracy |
| -------------------------------------- | -------- |
| Naive Bayes Classifier                 |    |
| XGBoost                                |    |
| Multi-Layer Perceptron Neural Network  |    |
| Long Short-Term Memory Neural Network* |    |

## Libraries

* Scikit-learn - Multifunctional machine learning library for Python (http://scikit-learn.org/)
* Keras - Deep Learning library for Python (https://keras.io/)
* XGBoost - Scalable and Flexible Gradient Boosting (https://xgboost.readthedocs.io/)
* NLTK - Natural Language Toolkit for Python (http://www.nltk.org/)
* stop-words - Repository of common stop words (https://pypi.python.org/pypi/stop-words)


## Useful Links

* https://deeplearning4j.org/ - Although this is focused on machine learning with java, here you can find very good explanation of the way most of the machine learning algorithms work.