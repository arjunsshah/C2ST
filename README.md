# Classifier Based Two Sample Testing
In two-sample testing, we want to determine if two selected samples come from the same or different distributions. C2ST aims to use neural network classifiers to help determine this. 

## Process
Two samples, $S_P = \{x_1,...,x_n\}$ and $S_Q = \{y_1, ..., y_m\}$

1) Construct our dataset

$D = \{(x_i,0)\}^n_{i=1} \cup \{(y_i,0)\}^n_{i=1} = \{z_i,l_i\}^{2n}_{i=1}$

2) Split dataset

Shuffle into train-test split

3) Train binary classifier

4) Return a classification accuracy on $D_{te}$:

$\hat t = \frac{1}{n_{te}} \Sigma_{(z_i, l_i) \in D_{te}} I[I(f(z_i) > \frac{1}{2}) = l_i]$


5) Accept/Reject $H_0$: A significance threshold is written up below where if our c2st statistics is less than it, we accept the null hypothesis, otherwise we reject.

## How to use this notebook?
I will outline the steps needed to understand and use this notebook to perform classifier two-sample testing on any datasets

Below I have outlined a c2st function that takes in both samples, X and Y, along with other parameters that can be edited. 

For example, here is a classifier two-sample test between a student t-distribution and gaussian distribution:



```
student = scale(np.random.standard_t(20, size = 1000)).reshape(-10,10) # X
gaussian = scale(np.random.normal(0,1,1000)).reshape(-10,10) # Y

c2st(student, gaussian)
```
The last line will return the c2st score for the two samples. Currently, the default parameter for the machine learning model used as our classifier is a RandomForestClassifier but can be changed as a parameter:



```
c2st(student, gaussian, clf=MLPClassifier(activation='tanh', hidden_layer_sizes=(10,10,10), max_iter=600))
```
Here, you can edit the number of hidden layers, activation functions, etc for our classifier. Here is a [link](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) to the sckit-learn MLPClassifer model for specifics.

### Accepting/Rejecting $H_0$ and Power

I have created two functions, significance_threshold and power, in order to determine when we accept/reject our null hypothesis. Both of these formulas come straight from Lopez-Paz's paper in the appendix. Here is an example of using the significance_threshold function:


```
significance_threshold(0.05, 100)
```
The first parameter is the significance level and the second parameter is the size of our test set when we do the train test split. 

The power function is similar in nature except it has an additional parameter representing the distance of our c2st score from 0.5 (the null hypothesis). Here is an example of how this can be calculated using our student vs gaussian example:


```
c2stScore = c2st(student, gaussian)
epsilon = c2stScore - 0.5
power(0.05, 1000, epsilon)
```
