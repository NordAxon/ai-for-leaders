# For building models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

# For code readability
from typing import List, Tuple


def build_ANN(nbr_nodes:List[int] = [32], dropout:bool = True, num_classes:int = 10) -> Model:
	"""Builds a model for supervised learning using Keras Functional API.

	Args:
		nbr_nodes: Each integer represents the number of nodes in each hidden layer e.g. [32, 16].
		dropout: Adding dropout layer or not.
		num_classes: Number of classes to classify.
	"""	

	# Input layer
	inputs = Input(shape=(28*28,))

	x = inputs
	# Fully-connected layers
	for nbr_node in nbr_nodes:
		x = Dense(nbr_node, activation="relu")(x)
		if dropout:
			x = Dropout(0.3)(x)

	# Output Layer
	outputs = Dense(num_classes, activation="softmax")(x)


	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss="categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

	return model

def build_CNN(nbr_filters:List[int] = [64, 32], kernel_shape:Tuple[int] = (3,3), nbr_nodes:List[int] = [32], dropout:bool = True, num_classes:int = 10) -> Model:
	"""Builds a model for supervised learning using Keras Functional API.

	Args:
		nbr_filters: Number of filters in each convolutional layer.
		kernel_shape: Shape of the filters/kernels.
		nbr_nodes: Each integer represents the number of nodes in each hidden layer e.g. [32, 16].
		dropout: Adding dropout layer or not.
		num_classes: Number of classes to classify.
	"""	

	# Input layer
	inputs = Input(shape=(28, 28, 1))

	# Convolutional base
	x = inputs
	for nbr_filter in nbr_filters:
		x = Conv2D(nbr_filter, kernel_shape, activation="relu", padding="same")(x)
		x = MaxPooling2D()(x)
		if dropout:
			x = Dropout(0.3)(x)

	# Fully-connected layers
	x = Flatten()(x)
	for nbr_node in nbr_nodes:
		x = Dense(nbr_node, activation="relu")(x)
		if dropout:
			x = Dropout(0.3)(x)

	# Output Layer
	outputs = Dense(num_classes, activation="softmax")(x)


	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss="categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

	return model