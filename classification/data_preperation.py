# Data engineering
import numpy as np
from tensorflow.keras.utils import to_categorical

# For code readability
from typing import List, Tuple

# Function for preparing data
def prepare_binary_dataset(images: np.ndarray, labels: np.ndarray, first_label: int, second_label:int) -> Tuple[np.ndarray, np.ndarray]:
  """Prepares data for binary classification with LogisticRegression model from scikit-learn.
  
  Args:
    images: All the images as a numpy array.
    labels: All the labels as a numpy array.
    first_label: The first label to classify.
    second_label: The second label to classify.
  """

  # Select only the relevant images for training
  first_indices = np.where(labels == first_label)
  second_indices = np.where(labels == second_label)
  indices = np.hstack([first_indices, second_indices])[0]

  # All images are flattened to 1D-arrays
  new_shape = (-1, 28*28) 
  X = np.reshape(images[indices], new_shape)

  y = labels[indices] == first_label

  return X, y

def prepare_ANN_dataset(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Prepares data for multiclass classification using an Artificial Neural Network with Keras API.
  
  Args:
    images: All the images as a numpy array.
    labels: All the labels as a numpy array.
  """
      
  # Images are reshapen from (-1, 28, 28) to (-1, 28, 28, 1) in acc. with Keras API
  new_shape = (-1, 28*28)

  X = np.reshape(images, new_shape)
  y = to_categorical(labels)

  return X, y

def prepare_CNN_dataset(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Prepares data for multiclass classification using an Convolutional Neural Network with Keras API.
  
  Args:
    images: All the images as a numpy array.
    labels: All the labels as a numpy array.
  """      
      
  # Images are reshapen from (-1, 28, 28) to (-1, 28, 28, 1) in acc. with Keras API
  new_shape = (-1, 28, 28, 1)

  X = np.reshape(images, new_shape)
  y = to_categorical(labels)

  return X, y