# Prototype Neural Network Implementation

In this first sprint I have implemented a prototype neural network and tested it with some fake data that I have generated.

The first implementation is of the simplest architecture, a feed-forward network with one hidden layer, shown below:

## Synthetic Test Data

The first layer consists of 8 synthetic inputs. These are data I have generated myself for initial testing. They are designed to have a very simple relationship between the data features (simulating quiz grades) and the output (passing or failing the course). Each data item is generated as a set of features between 0 (0% grade) and 1 (100% grade), using Python's built in Rand library in the code below:

```python
[code snippet]
```

The output is then calculated by computing an average of the feature values. If the result is greater than 0.5 the output is set to 1 (pass) if not it is set to -1 (fail).

```python
[code snippet]
```

The above calculation is repeated 2000 times to generate a moderately large dataset, which is saved to a CSV file.

## Neural Network Architecture

The following code creates the network architecture shown in the diagram above using the TensorFlow library in Python. The first step is to create a TensorFlow graph:

```python
[code snippet]
```

The next step creates an input layer of 8 units and saves a handle to it that can later be used to input the test data:

```python
[code snippet]
```

The hidden layer is the most complex. Each unit $j$ must have a weight $W_{ij}$ for each input $i$ plus a bias term $b_j$. They are used to calculate the output $o_j$ from the input values $a_i$, using the following formula, which performs a weighted sum of the inputs:

$$o_j = \sum_{i=0}^{32} W_{ij} \cdot a_i + b_j$$

The weights and biases are parameters whose values are set by learning and are initially set randomly, using TensorFlow's random initialisation function:

```python
[code snippet]
```

Once the weights have been created, the hidden layers are simply implemented as a sum according to the formula above, using TensorFlow's vector sum operations:

```python
[code snippet]
```

All neural network layers need a non-linear activation function at their output. In this case I have used a rectified linear unit (ReLU), which is a remarkably simple unit that nonetheless has been shown to work very effectively (Kittisak and Hossein 2018). It is simply the input but cut off below zero:

$$y = \max(0, x)$$

where $x$ is the input of the ReLU and $y$ the output.

This code applies the ReLU activation:

```python
[code snippet]
```

The implementation of the output layer is very similar to the hidden layer with weights and biases. This time there are only two units, one for pass and one for fail:

```python
[code snippet]
```

For the activation function of the output layer we use a softmax, which converts outputs into probabilities by dividing through all outputs by the sum of outputs using the following formula:

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

The application of softmax is shown in the code below:

```python
[code snippet]
```

> **Note:** I have not included a screenshot of the programme as it is simply command line at this stage and would not be informative.
