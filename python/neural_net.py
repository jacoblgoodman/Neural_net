import numpy as np


class NeuralNetwork:
    def __init__(self, num_Inputs,  num_HiddenNodes, num_Outputs,
                 alpha=.2, errorfunc='square_error', activation='sigmoid'):
        """
        initialization of our object
        """
        self._num_Inputs = num_Inputs
        self._num_Outputs = num_Outputs
        self._num_HiddenNodes = num_HiddenNodes
        self._alpha = alpha

        if errorfunc == 'square_error':
            self.errorfunc = NeuralNetwork.square_error
            self.Der_errorfunc = NeuralNetwork.Der_square_error

        if activation != 'sigmoid':
            self.Actfunc = getattr(NeuralNetwork,f'act_{activation}')
            self.Der_Actfunc = getattr(NeuralNetwork,f'der_{activation}')

        else:
            self.Actfunc = NeuralNetwork.sigmoid
            self.Der_Actfunc = NeuralNetwork.Der_sigmoid

        self.weight_matrix_initialization()

    def weight_matrix_initialization(self):
        rng = np.random.default_rng()

        # matrix set up
        w = rng.uniform(-.5, .51, size=self._num_HiddenNodes * self._num_Inputs)  # .51 duo to inclusive nature of uniform
        w = w.reshape((self._num_Inputs, self._num_HiddenNodes))

        W = rng.uniform(-.5, .51, size=self._num_HiddenNodes * self._num_Outputs)
        W = W.reshape((self._num_HiddenNodes,self._num_Outputs))

        # bias set up
        w = np.append(w, np.full((1, self._num_HiddenNodes), rng.uniform(-.5, .51, 1)), axis=0)
        W = np.append(W, np.full((1, self._num_Outputs), rng.uniform(-.5, .51, 1)), axis=0)

        # replacing any zero values
        w[w == 0] = rng.uniform(-.5, .51, 1)
        W[W == 0] = rng.uniform(-.5, .51, 1)

        # assigning to object

        self.w = w
        self.W = W

    @staticmethod
    def square_error(pred, actual):
        """default error function of our nueralnetwork given predicated value and actual returns error"""
        return  (sum(actual - pred)**2)*.5

    @staticmethod
    def Der_square_error(pred, target):
        """ derivative or our default error function"""
        return pred-target


    def predict(self, inp,y=None, w=None, W=None, func=None, ):
        """this function takes in inputs, weight matrix, and an activation function and makes a predication """
        if func is None:
            func = self.Actfunc
        if w is None:
            w = self.w
        if W is None:
            W = self.W

        x = np.append(inp, 1)
        pred = func(np.append(func(x.dot(w)),1).dot(W))

        if y is None:
            return pred
        else:
            return self.errorfunc(pred,y)

    def hiddennodes(self,inp):
        x = np.append(inp, 1)
        return self.Actfunc(x.dot(self.w))



    def train(self,x, y):
        # initalize
        G = np.zeros((self._num_HiddenNodes, self._num_Outputs))
        g = np.zeros((self._num_Inputs, self._num_HiddenNodes))
        Bo = np.zeros((1,self._num_Outputs))
        Bh = np.zeros((1,self._num_HiddenNodes))

        # create predictions
        cur_pred = self.predict(x)
        Thid = self.hiddennodes(x)  # trans formed hidden nodes

        # first step loop
        for k in range(self._num_Outputs):
            # calculate for an output node
            Bias = self.Der_errorfunc(cur_pred[k],y[k])*self.Der_Actfunc(cur_pred[k])
            for j in range(self._num_HiddenNodes):
                # add for hidden node
                G[j, k] = Bias*Thid[j]

            # update bias vector
            Bo[0, k] = Bias     # note this will be summed before utlized in updates but information stored
                                # here can be utlized in next step

        # second step loop
        for k in range(self._num_HiddenNodes):
            bias = Bo.dot(self.W[k,:])*self.Der_Actfunc(Thid[k])
            for j in range(self._num_Inputs):
                g[j, k] = bias*x[j]
            Bh += bias  # updating all bias because we aren't using these values again

        # clean up gradient arrays and append bias's
        Bo[:] = np.sum(Bo)
        G = np.append(G,Bo,axis=0)
        g = np.append(g,Bh,axis=0)

        self.W = self.W-self._alpha*G
        self.w = self.w-self._alpha*g

        return print('one pass')


    # activation functions

    @staticmethod
    def der_relu(x:float):
        if x >= 0:
            return 1
        else:
            return 0

    @staticmethod
    def act_relu(x:float):
        return max(0, x)

    @staticmethod
    def Der_sigmoid(x):
        return x *(1-x)

    @staticmethod
    def sigmoid(x):
        """our default activation function"""
        return 1/(1+np.e**(-x))

    @staticmethod
    def der_hyperbolic(x):
        return 4/(np.e**x+np.e**-x)**2

    @staticmethod
    def act_hyperbolic(x):
        return (np.e**x-np.e**-x)/(np.e**x+np.e**-x)
