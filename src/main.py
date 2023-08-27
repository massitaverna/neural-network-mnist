import numpy as np
import network
import random
from keras.datasets import mnist
import sys

'''
def evaluate(output, expected_output):
	expected_result = None
	for neuron in expected_output:
		if neuron == 1:
			expected_result = neuron
			break

	maxx = output[0]
	result = 0
	for idx, neuron in enumerate(output):
		if neuron > maxx:
			maxx = neuron
			result = idx

	return 1 if expected_result == result else 0

def conv2result(output):
	maxx = output[0]
	result = 0
	for idx, neuron in enumerate(output):
		if neuron > maxx:
			maxx = neuron
			result = idx
	return result



dataset = list()
for _ in range(10):
	data = np.array([random.randint(0,1) for _ in range (4)])
	result = [1]
	result.insert(random.randint(0,1), 0)
	example = network.Example(data, np.array(result))
	dataset.append(example)

print("Dataset:")
print(dataset)

neurons = [4, 3, 2]
netw = network.NN(neurons, 3, 10, 2)

print("Network:")
print(netw)
_, _, cost = netw.forwardpropagation(dataset[0].input, [0,1])
print("Dummy cost:", cost)

netw.train(dataset, dataset, evaluate)
'''


def main():
	mb_size = 10
	if len(sys.argv) > 1:
		mb_size = int(sys.argv[1])

	neurons = [784, 10, 10]
	netw = network.NN(neurons, eta=3, epochs=10, mb_size=mb_size)
	print(f"Neural network created (mini-batch size set to {mb_size})")

	(train_x, train_y), (test_x, test_y) = mnist.load_data()
	print("MNIST dataset loaded")

	training_dataset = list()
	c = 0
	for example_input, example_result in zip(train_x, train_y):
		c += 1
		if c%600 == 0:
			print(c//600, '%')
		example_output = np.zeros(10)
		example_output[example_result] = 1
		example_input = example_input.flatten()
		for i in range(len(example_input)):
			example_input[i] = 1 if example_input[i] > 0 else 0
		example = network.Example(example_input, example_output)
		training_dataset.append(example)
	
	test_dataset = list()
	for example_input, example_result in zip(test_x, test_y):
		example_input = example_input.flatten()
		for i in range(len(example_input)):
			example_input[i] = 1 if example_input[i] > 0 else 0
		example = network.Example(example_input, example_result)
		test_dataset.append(example)

	print("Dataset reformatted")

	def conv2result(output):
		maxx = output[0]
		result = 0
		for idx, neuron in enumerate(output):
			if neuron > maxx:
				maxx = neuron
				result = idx
		return result

	netw.train(training_dataset, test_dataset, conv2result)



if __name__ == '__main__':
 	main()
