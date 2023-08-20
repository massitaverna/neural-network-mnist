import numpy as np

class NN:
	def __init__(self, neurons, eta, epochs, mb_size):
		self.neurons = neurons
		self.eta = eta
		self.epochs = epochs
		self.mb_size = mb_size

		num_layers = len(neurons)
		self.weights = [np.zeros(shape=(neurons[l+1], neurons[l])) for l in range(num_layers-1)]
		self.biases  = [np.zeros(shape=(neurons[l+1])) for l in range(num_layers-1)]
