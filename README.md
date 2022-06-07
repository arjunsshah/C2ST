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
