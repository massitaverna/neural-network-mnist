import numpy as np
import random
import math

class Example:
	def __init__(self, ex_input, ex_output):
		self.input = ex_input
		self.output = ex_output

	def __str__(self):
		return f"{self.input} -> {self.output}"

	def __repr__(self):
		return self.__str__()

class NN:
	def __init__(self, neurons, eta, epochs, mb_size):
		self.neurons = neurons
		self.eta = eta
		self.epochs = epochs
		self.mb_size = mb_size

		num_layers = len(neurons)
		self.weights = [np.random.rand(neurons[l+1], neurons[l]) for l in range(num_layers-1)]
		self.biases  = [np.random.rand(neurons[l+1]) for l in range(num_layers-1)]

	def train(self, training_data, test_data=None, conv2result=None):
		if len(training_data) % self.mb_size != 0:
			raise ValueError(f"Training data size ({len(training_data)}) must be a multiple of mini-batch size {self.mb_size}") 
		
		for epoch in range(self.epochs):
			print(f"Running epoch {epoch}")
			self.run_epoch(training_data)
			if test_data and conv2result:
				accuracy = 0
				for test in test_data:
					output = self.process(test.input)
					if conv2result(output) == test.output:
						accuracy += 1
				accuracy /= len(test_data)

				print(f"Epoch {epoch} completed: {round(accuracy*100, 2)}%")

	def run_epoch(self, training_data):
		random.shuffle(training_data)
		mini_batches = [training_data[i:i+self.mb_size] for i in range(0, len(training_data), self.mb_size)]
		self.SGD(mini_batches)

	# Stochastic Gradient Descent
	def SGD(self, batches):
		n = len(batches[0]) # All batches have same size
		L = len(self.weights)
		for ctr, batch in enumerate(batches):
			nabla_w = [np.zeros_like(self.weights[l]) for l in range(L)]
			nabla_b = [np.zeros_like(self.biases[l])  for l in range(L)]
			avg_cost = 0
			for example in batch:
				activations, zs, cost = self.forwardpropagation(example.input, example.output)
				delta_w, delta_b = self.backpropagation(activations, zs, example.output)
				for l in range(L):
					nabla_w[l] += delta_w[l]
					nabla_b[l] += delta_b[l]
				avg_cost += cost
			for l in range(L):
				nabla_w[l] /= n
				nabla_b[l] /= n

				self.weights[l] -= self.eta * nabla_w[l]
				self.biases[l]  -= self.eta * nabla_b[l]

			avg_cost /= n
			#print(f"Cost in training at batch {ctr}: {avg_cost}")

	@staticmethod
	def sigmoid(x):
		return 1/(1+math.e**(-x))

	@staticmethod
	def sigmoid_derivative(x):
		return math.e**(-x) / (1+math.e**(-x))**2

	def forwardpropagation(self, example_input, y=None):
		activations = list()
		zs          = list()

		act = example_input
		activations.append(act)
		for layer in range(len(self.weights)):

			next_act = np.ndarray(shape=(self.neurons[layer+1]))
			z        = np.ndarray(shape=(self.neurons[layer+1]))
			for neuron in range(self.neurons[layer+1]):
				z[neuron] = np.dot(act, self.weights[layer][neuron, :]) + self.biases[layer][neuron]
				next_act[neuron] = NN.sigmoid(z[neuron])
			
			act = next_act
			zs.append(z)
			activations.append(act)

		cost = None
		if y is not None:
			dist = y - act
			cost = np.dot(dist, dist)
		return activations, zs, cost

	def backpropagation(self, activations, zs, y):
		a = activations[-1]
		der_a = 2*(a-y)

		L = len(self.weights)
		nabla_w = list()
		nabla_b = list()
		for l in range(L-1, -1, -1):
			# print("dC/dA =")
			# print(der_a)

			z = zs[l]
			a = activations[l]
			sigmoid_der = NN.sigmoid_derivative(z)
			temp = sigmoid_der * der_a
			nabla_w = [temp.reshape(-1,1) @ a.reshape(-1,1).transpose()] + nabla_w
			nabla_b = [temp] + nabla_b
			der_a = np.sum(temp.reshape(-1,1) * self.weights[l], axis=0)

			# print(f"nabla_w[{l}] =")
			# print(nabla_w[0])
			# print(f"nabla_b[{l}] =")
			# print(nabla_b[0])

			# _ = input("Continue?")

		return nabla_w, nabla_b

	def process(self, input_data):
		activations, _, _ = self.forwardpropagation(input_data)
		output = activations[-1]
		return output