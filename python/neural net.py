import numpy as np


class NeuralNetwork:
    def __init__(self, num_Inputs,  num_HiddenNodes, num_Outputs, alpha=.2, errorfunc='square_error', activation='sigmoid'):
        """
        initalization of our obeject
        """
        self._num_Inputs = num_Inputs
        self._num_Outputs = num_Outputs
        self._num_HiddenNodes = num_HiddenNodes
        self.alpha = alpha

        if errorfunc == 'square_error':
            self.errorfunc = NeuralNetwork.square_error
            self.Der_errorfunc = NeuralNetwork.Der_square_error

        if activation == 'sigmoid':
            self.Actfunc = NeuralNetwork.sigmoid
            self.Der_Actfunc = NeuralNetwork.Der_sigmoid

        self.weight_matrix_initialization()

    def weight_matrix_initialization(self):
        rng = np.random.default_rng()

        # matrix set up
        w = rng.integers(-5, 5, size=self._num_HiddenNodes * self._num_Inputs)
        w = w.reshape((self._num_Inputs, self._num_HiddenNodes))

        W = rng.integers(-5, 5, size=self._num_HiddenNodes * self._num_Outputs)
        W = W.reshape((self._num_HiddenNodes,self._num_Outputs))

        # bias set up
        self.w = np.append(w, np.full((1, self._num_HiddenNodes), rng.integers(-5, 5, 1)), axis=0)
        self.W = np.append(W, np.full((1, self._num_Outputs), rng.integers(-5, 5, 1)), axis=0)

    @staticmethod
    def square_error(pred, actual):
        """default error function of our nueralnetwork given predicated value and actual returns error"""
        return  (sum(actual - pred)**2)*.5


    @staticmethod
    def sigmoid(x):
        """our default activation function"""
        return 1/(1+np.e**(-x))

    def predict(self, inp, w=None, W=None, func=None, y=None):
        """this function takes in inputs, weight matrix, and an activation function and makes a predication """
        if func is None:
            func = self.Actfunc
        if w is None:
            w = self.w
        if W is none:
            W - self.W
        x = np.append(x,1)
        pred = func(np.append(func(x.dot(w)),1).dot(W))

        if y is None:
            return pred
        else:
            return self.errorfunc(pred,y)

    def hiddennodes(self,inp):
        x = np.append(inp, 1)
        return self.Actfunc(x.dot(self.w))

    @staticmethod
    def Der_square_error(pred, target):
        """ derivative or our default error function"""
        return pred-target
    @staticmethod
    def Der_sigmoid(val):
        return val *(1-val)

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
            bias = Bo.dot(self.W[k,:])*Der_sigmoid(Thid[k])
            for j in range(num_inputs):
                g[j, k] = bias*x[j]
            Bh += bias  # updating all bias because we aren't using these values again

        # clean up gradient arrays and append bias's
        Bo[:] = np.sum(Bo)
        G = np.append(G,Bo,axis=0)
        g = np.append(g,Bh,axis=0)

        self.W = self.W-self.alpha*G
        self.w = self.w-self.alpha*g

        return print('one epoch')