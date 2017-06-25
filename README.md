# Sentiment Analysis

## Motivation
The idea is to write different algorithms using different machine learning approaches in order to compete them in solving sentiment analysis problem.
After playing around with the alogrithms, one of the will be chosen as the best one. It will be traind and tweaked for the best results.
As a result this model will be saved and demonstrated.

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

![Word2vec](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/Word2vec.gif)

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
* **P(A)** is the probability of observing ***A***;
* **P(B)** is the probability of observing ***B***;
* **P(A|B)** is the probability of observing ***A*** given that ***B*** is true;
* **P(B|A)** is the probability of observing ***B*** given that ***A*** is true.

If we apply this theorem to an actial problem, it will look like this:

```
P(h|D) = (P(D|h) P(h)) / P(D)
```

Where:
* **P(h)** - ** *a priori* probability** - the probability that the hypothesis ***h*** is true, before even taking a look at the training data. If we have no ** *a priori* knowledge**, we can use the same probability for all;
* **P(D)** - the ** *a priori* probability** that we will see the data ***D***. In other words this is the probability that we will see ***D*** without knowing which hypothesis is correct;
* **P(D|h)** - the expected probability for seeing the data ***D***, given that the hypothesis ***h*** is true;
* **P(h|D)** - ** *a posteriori* probability* - the probability that ***h*** is true, given that we have the training data D.

In the Bayes Classifier, we can compute the most probable classifiaction of new example by combining the predictions of all the hypothesis, weighed by their ** *a posteriori* probabilities** 

The Naive Bayes classifier is a method for Bayes learning, which has proven its usablity in numerous tasks. 
It is called **naive** because it uses strong (naive) independence assumptions. 
To be more specific - it assumes that the values of the attributes are independent in the scope of a given classification of the example. 
Even if this assumption is not met, the classification method is still very effective.

### XGBoost

[XGBoost](http://xgboost.readthedocs.io/en/latest/) is short for “Extreme Gradient Boosting”,
where the term “[Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)” is proposed in the paper Greedy Function Approximation:
A Gradient Boosting Machine, by Friedman. XGBoost is based on this original model.
The GBM (boosted trees) has been around for really a while, and there are a lot of materials on the topic.

XGBoost is an optimized distributed gradient boosting [library](https://github.com/dmlc/xgboost) designed to be highly efficient, flexible and portable.
It implements machine learning algorithms under the Gradient Boosting framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

### Multi-Layer Perceptron Neural Network

The idea behind the Neural Networks is to replicate the way the human brain works.
In order to achieve that they create a big network of so called ```perceptrons``` - specailly designed mathematical models that sumulate biological neurons.

Basically a percepron looks like this:

![Perceptron](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/Perceptron.PNG)

You can see that it is consisted of several parts, which are:
* Input - a perceptron takes multiple inputs which, in the picture they are enumerated as x<sub>0</sub>, x<sub>1</sub> to x<sub>n</sub>. Another way to represent the input is as the vector x.
* Weights - weights are the metric that describes how meaninful is each input for the final classification.
* Bias - a constant, the purpose of which is to tilt the activation function in order to help it fit better the training data
* Sum - the sum of the Inputs, each multiplied by its weight, plus the bias.
* Activation function - the function that produces the final result.

The forward pass of the perceptron, meaning the transition form the inputs to the output is described by this formula:


*output = g(XW + b)*


Where:

* *X = x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub> ... x<sub>n</sub> - the input of the perceptron*;
* *W = w<sub>0</sub>, w<sub>1</sub>, w<sub>2</sub> ... w<sub>n</sub> - the weights for each input*;
* *b - the bias*;
* *g() - the activation function*.



In order to make a Neural Network we need to combine multiple layers of perceptrons.
Imagine that we have one perceptron which receives some input and processes some otput in the form of a vector (it can be with the same dimensions as the input or different).
Now we can take the output of our first perceptron and pass it to the second one.
This process can be repeated as much as we want to.
Every perceptron feeding form the input of another one is part of a hidden layer.
To get better idea of the situation, check the picture below. Note that the links between the input and the output are repleced by X for simplicity.


![NeuralNetwork](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/NeuralNetwork.PNG)

### Long Short-Term Memory Neural Network

Befor talking about LSTM it is only natural that we first mention something about Recurrent Neural Networks in general.

A recurent neural netowrk is verry similar to normal neural network.
In fact if looked from the outmost perspective it looks exacly the same - input layer, some hidden layers and an output layer.
However it is very differeny in the way the hidden layers behave.

![RNN_Hidden_Layer](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/RNN_Hidden_Layer.PNG)

Where:

* W, U - weight matrices;
* x<sub>0</sub> - vector representing the first word;
* s<sub>0</sub> - cell state at t = 0;
* s<sub>1</sub> - cell state at t = 1 (s<sub>1</sub> = *tanh*(W x<sub>0</sub> + U s<sub>0</sub>)).

This basically means that Recurrent Neural Networs are aware of their previous state.
You can olso say that they remember a context, or that they are context aware.

Another way to represent the RNN is if we unfold it through time:

![Unfolded_RNN](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/Unfolded_RNN.PNG)

LSTM is basically a Reacurrent Neural Network that uses several steps of logic gates, which controls the flow of input through it.
The purpose of this gates is to decide what information flows through and what information gets multiplied by the weights and activations.

How LSTM works:

* At first it forgets the usless (irrelevant) parts of the previous state;
* Not all cell values are updated. The update process is done selectively. Again irrelevant subjects don't update the cell state;
* Output certain parts of the cell state that are considered rellevant.

This way all the irellevant information is forgotton to give priority to remembering only what is rellevant.

## Current Results

### Results using the first dataset

| Algorithm                              | Accuracy |
| -------------------------------------- | -------- |
| Naive Bayes Classifier                 |  77.00%  |
| XGBoost                                |  86.08%  |
| Multi-Layer Perceptron Neural Network  |  87.23%  |
| Long Short-Term Memory Neural Network* |  83.45%  |


*\*For this model, the data is represented using One-hot encoding*


### Results using the second dataset

| Algorithm                              | Accuracy |
| -------------------------------------- | -------- |
| Naive Bayes Classifier                 |  76.69%  |
| XGBoost                                |  85.96%  |
| Multi-Layer Perceptron Neural Network  |  86.42%  |
| Long Short-Term Memory Neural Network* |  89.20%  |


*\*This time, the data was represented using word2vec*

As you can see, the results using this datasets are a little bil lower for most of the algorithms (for all without LSTM).
There are two reasons for this to hapen:
* With this dataset we have more data to train, but also a lot more data to test with. Therefor it is normal that the results will be a little lower
* All the algorithms except for LSTM were tweaked to work good with the first dataset. A little bit of tweaking the parameters can lead to better results with the second dataset, but this is not hte scope of the current project.

### Lest take a further look at the LSTM scoring the best resutls



#### Model

The implementation is fairly simple. There are only three layers in the model:
* LSTM layer
* Droput layer - Dropout is a technique which aims to reduce the overfitting in a Neural netowrk. Its tries to prevent the the adaptation to the training data. 
* Dense layer - The final layer is a simple dens layer which presents the output as a single value.

#### Learning process

The learning process usign LSTM is slow.
It becomes even slower, when we add the transformation of every single word into verctors (using word2vec).
The fact that this processing takes time is not the only thing that is slow, when every word is transformed in 300 dimensional vector it takes a lot of time for LSTM to process it.

Here you can take a look into the accuracy during different steps of the learning process:

![Accuracy](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/Accuracy.png)

As you can already see from the figure above, the process took 12 epochs to achieve its best results.

For completeness, here is the chart displaying the loss, during the same learning process:

![Loss](https://gitlab.com/university-projects/sentiment-analysis/raw/master/assets/Loss.png)


## Conclusion

In these project we observed four machine learning algorithms and competed them to solving sentiment analysis problem.
Each gave there results and finally, it was concluded that from the chosen alogrithms it was LSTM that solved the problem the best way.
That is not a surprace, because a lot of research lead to the fact that Recurrent Neural Networks are the best at solving problems of this matter.
LSTM is a concept that has gone even further and provided a brilliant sollution.

## Libraries

* Scikit-learn - Multifunctional machine learning library for Python (http://scikit-learn.org/)
* Keras - Deep Learning library for Python (https://keras.io/)
* XGBoost - Scalable and Flexible Gradient Boosting (https://xgboost.readthedocs.io/)
* NLTK - Natural Language Toolkit for Python (http://www.nltk.org/)
* stop-words - Repository of common stop words (https://pypi.python.org/pypi/stop-words)


## Useful Links

* https://deeplearning4j.org/ - Although this is focused on machine learning with java, here you can find very good explanation of the way most of the machine learning algorithms work.