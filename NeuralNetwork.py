import numpy as np
import matplotlib.pyplot as plt
from Activations import Activation
np.random.seed(1)


class NeuralNetwork:

    def __init__(self, X, Y, layers_dims, learning_rate=0.001, num_iterations=3000, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)

        # save network parameters and hyper parameters
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.activation = Activation()

        # Parameters initialization.
        self.parameters = self.initialize_parameters_deep(layers_dims)

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation='relu'):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.activation.sigmoid(Z)

        else:
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.activation.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def deep_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(1, L):
            A_prev = A
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A, cache = self.linear_activation_forward(A_prev, W, b, activation="relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        W = parameters["W" + str(L)]
        b = parameters["b" + str (L)]
        AL, cache = self.linear_activation_forward(A, W, b, activation="sigmoid")
        caches.append (cache)

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, cache[0].T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[1].T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation='relu'):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = self.activation.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        else:
            dZ = self.activation.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def deep_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (deep_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

        # Lth layer (SIGMOID -> LINEAR) gradients.
        # Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(
            dAL,
            current_cache,
            activation="sigmoid"
        )

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches".
            #  Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 2)],
                current_cache,
                activation="relu"
            )

            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of deep_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    def train(self, X=None, Y=None):
        """
        train the model

        Arguments:
        X -- numpy array for X training data
        Y -- numpy array for Y targets
        """
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        costs = []
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.deep_model_forward(self.X, self.parameters)

            # Compute cost.
            cost = self.compute_cost(AL, self.Y)

            # Backward propagation.
            grads = self.deep_model_backward(AL, self.Y, caches)

            # Update parameters.
            self.parameters = self.update_parameters(self.parameters, grads, self.learning_rate)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze (costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def test_accuracy(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        predictions -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(parameters) // 2  # number of layers in the neural network
        predictions = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.deep_model_forward(X, parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((predictions == y) / m)))

        return predictions

    def get_trained_parameters(self):
        return self.parameters
