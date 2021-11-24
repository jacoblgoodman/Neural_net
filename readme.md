# nearual networkClassifier 

## introduction 
The Goal of this notebook is to demonstraite the ability to create train and predict with a nerual network model. 

Asumptions: for this notebook we will assume an archiecture with 3 layers (input,hidden,output) but should be able to have N number of nodes for each layer.

## **features**

**weight matrix** 

The most import feature of our object is the weight matrix's. based on our architecutere our network will have 2 weight matrix's 1 for input to hidden nodes one for hidden nodes to output.

Matrix1:

* input to hidden node: 

* size K+1 x H : +1 for the bias where k is the number of features and H is the number of hidden nodes 

  \* Bias row to all have same value

Matrix2

* hidden to output 

* size H+1 x O: +1 for the bias where H is the hidden nodes and O is the number of output nodes

  \* Bias row to all have same value



## Functions:

Our model will need the following capibliities

* predict <- given an input vector sample output values for each output node based on the current weight matrixs 
  * extra work input vecs at once reutrn array of predcations 
* error <- given 2 inputs  an input vector and an target vector output difference for each output node based on the current weights 
* train <- For a given 2 inputs an input vector and an target vector. utlize the error to calculate the gradients for every serreis of weights from output to hidden node and hidden node to input and  2 biases and utlizes to update our weight matrixs and biases
  * next work with set of inputs
  * instead of updating for each instance work store and average for mini batch updates

## under the hood

to acomplish the functions above the following functions need to exist under the hood.

**to predict**

* Activation function: we will need a function that activates the nodes base on the sumwe will start with the sigmoid function 
* Predict function: using our activation function and weight matrixs calculate all our outputs using the forward propigation method.
  * should store the following values 
    * inputs
    * sum hidden nodes
    * activated hidden nodes
    * 

**find errors**

* Error function: given an output vector from out predict funciton and target vector find the error 

**to train** 

* Derivitive of the activation function: we will utlize this to find the gradients 
  * intial will be sigmoid
* gradiaent functions: for a given input and error find the gradients of the following:
  * hidden to ouput weights
  * hidden to ouput bias
  * hidden to input weights
  * hideen to input bias 
* each of these gradients or scalars (in the case of the biase) should be stored. the number of samples should also be stored(default = 1 )

