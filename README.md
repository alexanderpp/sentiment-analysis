# Sentiment Analysis

## Intro
The idea is to write different algorithms using different machine learning approaches in order to compete them in solving sentiment analysis problem.

## Data

The data ised for testing and training is part Keggle competiton called "Bag of Words Meets Bags of Popcorn".
It consists of 25,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews.
For all of the data is spit into training data and test data, using the following proportions - 70% for training and 30% for testing.

Link: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

## Algorithms

* Naive Bayes Classifier
* XGBoost
* Multi-Layer Perceptron Neural Network
* Long Short-Term Memory Neural Network

## Current Results
| Algorithm                             | Accuracy |
| ------------------------------------- | -------- |
| Naive Bayes Classifier                |  77.00%  |
| XGBoost                               |  86.08%  |
| Multi-Layer Perceptron Neural Network |  87.23%  |
| Long Short-Term Memory Neural Network |  83.45%  |

## Libraries

* Scikit-learn - Multifunctional machine learning library for Python (http://scikit-learn.org/)
* Keras - Deep Learning library for Python (https://keras.io/)
* XGBoost - Scalable and Flexible Gradient Boosting (https://xgboost.readthedocs.io/)
* NLTK - Natural Language Toolkit for Python (http://www.nltk.org/)
* stop-words - Repository of common stop words (https://pypi.python.org/pypi/stop-words)


## Useful Links

* https://deeplearning4j.org/ - Although this is focused on machine learning with java, here you can find very good explanation of the way most of the machine learning algorithms work.