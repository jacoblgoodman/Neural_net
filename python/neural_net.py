import numpy as np
from sympy import log


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

        # setting dtype
        w = w.astype(np.longdouble)
        W = W.astype(np.longdouble)
        # assigning to object

        self.w = w
        self.W = W

    @staticmethod
    def square_error(pred, actual):
        """default error function of our nueralnetwork given predicated value and actual returns error"""
        return (sum(actual - pred)**2)*.5

    @staticmethod
    def Der_square_error(pred, target):
        """ derivative or our default error function"""
        return pred-target


    def predict(self, inp, y=None, w=None, W=None, func=None):
        """this function takes in inputs, weight matrix, and an activation function and makes a predication """
        if func is None:
            func = self.Actfunc
        if w is None:
            w = self.w
        if W is None:
            W = self.W

        x = self.make_ones(inp)

        # forward propagating
        pred = func(self.make_ones(func(x.dot(w))).dot(W))

        if y is None:
            return pred
        else:
            return self.errorfunc(pred,y)

    def make_ones(self, inp):
        # appending 1's for bias
        if inp.ndim == 1:
            x = np.append(inp, 1)
        elif inp.ndim > 1:
            x = np.c_[inp, np.ones(inp.shape[0])]
        return x

    def hiddennodes(self,inp):
        x = np.append(inp, 1).astype(np.longdouble)
        return self.Actfunc(x.dot(self.w))



    def get_gradients(self, x, y, batch=False):
        # initalize
        G = np.zeros((self._num_HiddenNodes, self._num_Outputs), dtype=np.longdouble)
        g = np.zeros((self._num_Inputs, self._num_HiddenNodes), dtype=np.longdouble)
        Bo = np.zeros((1, self._num_Outputs), dtype=np.longdouble)
        Bh = np.zeros((1, self._num_HiddenNodes), dtype=np.longdouble)

        # create predictions
        cur_pred = self.predict(x)
        Thid = self.hiddennodes(x)  # trans formed hidden nodes

        # first step loop
        for k in range(self._num_Outputs):
            # calculate for an output node
            Bias = np.multiply(self.Der_errorfunc(cur_pred[k],y[k]),self.Der_Actfunc(cur_pred[k]))


            for j in range(self._num_HiddenNodes):

                # add for hidden node
                 G[j, k] = Bias * Thid[j]

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

        if batch is True:
            return g, G

        elif batch is False:
            self.W = np.nan_to_num(self.W-(self._alpha*G))
            self.w = np.nan_to_num(self.w-(self._alpha*g))

            print("one pass")
            return
        else:
            print('bad options for batch should be True/False')

    def train(self, X, Y, batchsize=1, numbatches= None):
        """Primary training method of our neural network"""
        # if input is length of number of inputs assume single input
        if X.size == self._num_Inputs:
            status = self.get_gradients(X, Y)
            return
        # if batch size is 1 loop over array
        elif batchsize == 1:
            for x, y in zip(X, Y):
                self.get_gradients(x, y)
            print("one epoch")
            return
        else:
            import math
            if numbatches is not None:
                for x, y  in zip(np.array_split(X,numbatches),  np.array_split(Y,numbatches)):
                    numrecords = 0
                    weight1 = []
                    weight2 = []
                    for record, target in zip(x, y):
                        g, G = self.get_gradients(record, target,batch=True)
                        numrecords += 1
                        weight1.append(g)
                        weight2.append(G)

                    g = sum(weight1)/numrecords
                    G = sum(weight2)/numrecords
                    self.W = self.W-self._alpha*G
                    self.w = self.w-self._alpha*g
                    print("one batch")
            else:
                batches = math.ceil(len(X)/batchsize)
                for x, y  in zip(np.array_split(X,batches),  np.array_split(Y,batches)):
                    #initialize
                    numrecords = 0
                    weight1 = []
                    weight2 = []
                    for record, target in zip(x, y):
                        g, G = self.get_gradients(record, target,batch=True)
                        numrecords += 1
                        weight1.append(g)
                        weight2.append(G)

                    g = sum(weight1)/numrecords
                    G = sum(weight2)/numrecords
                    self.W = self.W-self._alpha*G
                    self.w = self.w-self._alpha*g
                    print("one batch")

    # activation functions

    @staticmethod
    def der_relu(x):
        return np.where(x < 0, 0, 1).astype(np.longdouble)

    @staticmethod
    def act_relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def Der_sigmoid(x):
        return x *(1-x)

    @staticmethod
    def sigmoid(x):
        """our default activation function"""
        return 1/(1+np.exp(-x))

    @staticmethod
    def der_hyperbolic(x):
        return 4/(np.e**x+np.e**-x)**2

    @staticmethod
    def act_hyperbolic(x):
        return (np.e**x-np.e**-x)/(np.e**x+np.e**-x)
