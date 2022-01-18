import numpy as np

# class Node:
#
# 	def __init__(self):
# 		self.weights = None
#
# 	def active_function(self, input_parameter):
# 		raise NotImplemented


def convert_val_to_class(val):
	classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	class_centers = np.array([[0.16667, 0.5, 0.83337]])

	expanded_class_centers = np.repeat(class_centers, val.shape[0], axis=0) # expand rows
	expanded_val = np.repeat(val, class_centers.shape[1], axis=1) # expand columns

	dis = np.abs(expanded_class_centers - expanded_val)
	close_centers_index = np.argmin(dis, axis=1)  # find center with min distance to value
	close_centers_index = np.resize(close_centers_index, (len(close_centers_index), 1))

	map_to_class = lambda x: classes[x]
	vfunc = np.vectorize(map_to_class)

	close_classes = vfunc(close_centers_index)

	return close_classes


def confusion_matrix(target, output):
	cm = {}
	for i in range(0, target.shape[0]):
		for j in range(0, target.shape[1]):
			label = str(target[i][j]) + '|||' + str(output[i][j])
			cm[label] = cm.get(label, 0) + 1
	return cm

def merge_con_matrix(cm1 , cm2):
	for k in cm2.keys():
		try:
			cm1[k] += cm2[k]
		except Exception:
			cm1[k] = cm2[k]
	return cm1


def sigmoid(x):
	z = np.exp(-x)
	sig = 1 / (1 + z)
	return sig


def sigmoid_derivation(x):
	return sigmoid(x) * (1 - sigmoid(x))


class Layer:

	def __init__(self, activation_function, activation_function_derivation, weights=None, outputs=None, ):
		self.weights = None
		self.new_weights = None
		self.outputs = None
		self.net_value = None
		self.activation_function: callable = activation_function
		self.activation_function_derivation: callable = activation_function_derivation

	def initialize_weights(self, neuron_num, input_num):
		self.weights = np.random.randn(neuron_num, input_num)
		# self.weights = np.random.random_integers(1, 4, (neuron_num, input_num))


class NeuralNetwork:
	def __init__(self, alpha=0.6):
		# initialize the list of weights matrices, then store the
		# network architecture and learning rate
		self.layers: list = []
		self.alpha = alpha

	def initialize_layers(self, input_num, *layer_neuron_nums):
		if len(layer_neuron_nums) == 0:
			return
		previous_neuron_num = input_num
		for num in layer_neuron_nums:
			l = Layer(sigmoid, sigmoid_derivation)
			l.initialize_weights(num, previous_neuron_num)
			self.layers.append(l)

			previous_neuron_num = num

	def inference(self, inp):
		previous_out = inp
		for layer in self.layers:
			layer.net_value = np.dot(layer.weights, previous_out)
			layer.outputs = layer.activation_function(layer.net_value)
			previous_out = layer.outputs

	def get_output(self):
		return self.layers[-1].outputs

	def train(self, train_data, test_data, epoch_num):
		errors_per_epoch = []
		for i in range(0, epoch_num):
			train_error = 0
			for d in train_data:
				self.train_by_case(d[0], d[1])
				train_error += self.get_square_error(d[1])

			self.update_learning_rate(epoch=i+1)

			test_error, test_conf_matrix = self.test(test_data)
			errors_per_epoch.append((i, train_error, test_error, test_conf_matrix))

		return errors_per_epoch

	def test(self, data):
		error = 0
		conf_matrix = {}
		for d in data:
			self.inference(d[0])
			out = self.get_output()
			error += self.get_square_error(d[1])
			co = convert_val_to_class(out)  # converted output 0.5 to Iris-versicolor
			ct = convert_val_to_class(d[1])
			new_cm = confusion_matrix(ct, co)
			conf_matrix = merge_con_matrix(conf_matrix, new_cm)
		# print(conf_matrix)
		return error, conf_matrix

	def train_by_case(self, inp, target):
		self.inference(inp)
		error = self.get_error(target)

		self.cal_new_weights(error, inp)
		self.substitute_weights()
		return error

	def get_error(self, target):
		return target - self.get_output()

	def get_square_error(self, target):
		e = target - self.get_output()
		e2 = np.square(e)
		return np.sqrt(np.sum(e2))

	def substitute_weights(self):
		for layer in self.layers:
			layer.weights = layer.new_weights

	def cal_new_weights(self, error, inp):
		for i in range(0, len(self.layers)):
			bp = self.get_backpropagation_val(i, error, inp)
			self.layers[i].new_weights = self.layers[i].weights + self.alpha * bp

	def update_learning_rate(self, epoch):
		self.alpha -= 0.3/epoch

	def get_backpropagation_val(self, layer_index, error, inp):
		llb = self.get_backpropagation_last_layer_base(error)
		value = llb

		plo = self.get_previous_layer_output(layer_index, inp)

		if layer_index == len(self.layers) - 1:    # if we are calculating for last layer
			return np.dot(value, plo.T)

		for i in range(len(self.layers) - 1, layer_index, -1):
			value = np.dot(self.layers[i].weights.T, value)
			value = np.multiply(value, self.layers[i-1].activation_function_derivation(self.layers[i-1].net_value))

		return np.dot(value, plo.T)

	def get_previous_layer_output(self, layer_index, inp):
		if layer_index == 0:
			return inp
		else:
			return self.layers[layer_index - 1].outputs

	def get_backpropagation_last_layer_base(self, error):
		ll_afd = self.layers[-1].activation_function_derivation # last layer  activation function derivation
		lln = self.layers[-1].net_value
		deriv_of_net = ll_afd(lln)
		ret = np.multiply(error, deriv_of_net)
		return ret
