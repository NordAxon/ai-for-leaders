from skimage import io, color
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import rescale_intensity
import cv2
import pylab
import skimage.measure
import numpy as np

import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Dimensionality reduction
from sklearn.decomposition import PCA

from typing import Dict, List, Any, Tuple

# Displaying images
def imshow(img: np.ndarray) -> None:
  """Displays an image and it's pixel values
  
  """
  fig, ax = plt.subplots()
  fig.set_size_inches(10.5, 10.5, forward=True)
  min_val, max_val = 0, 15
  ax.matshow(img, cmap=plt.cm.Blues)

  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          c = img[j,i]
          ax.text(i, j, str(c), va='center', ha='center')

def visualize_dataset(images: np.ndarray, labels: np.ndarray, label_to_article: Dict[int, str]) -> None:
  fig, axs = plt.subplots(2, 5, figsize = (12, 7))
  for i in range(10):
    grid_index = (i//5, i%5)

    index = np.where(labels==i)[0][0]
    image = images[index]
    axs[grid_index].imshow(image/255., cmap=plt.cm.gray)
    title = f"Article:  {label_to_article[labels[index]]}\n" + \
            f"Label:  {i}"
    axs[grid_index].set_title(title)
    axs[grid_index].axis('off')
  plt.show()

# Loading image
def load_example_image(url: str) -> np.ndarray:
  image = io.imread(url)    # Load the image
  image = color.rgb2gray(image)       # Convert the image to grayscale (1 channel)
  image *= 255
  return image

def visualize_convolution(original_image: np.ndarray, kernel: List[Any]) -> None:
  fig, axs = plt.subplots(1, 2, figsize = (14, 8))

  # Original image
  axs[0].imshow(original_image, cmap=plt.cm.gray)
  axs[0].set_title("Original image")
  axs[0].axis('off')

  # Convoluted image
  convolved_image = convolve(np.array(original_image), np.array(kernel))
  #convolved_image = convolve2d(original_image, kernel)
  #convolved_image = skimage.measure.block_reduce(convolved_image, (2,2), np.max) # Maxpooling
  #convolved_image = exposure.equalize_adapthist(convolved_image/np.max(np.abs(convolved_image)), clip_limit=0.03)
  axs[1].imshow(convolved_image, cmap=plt.cm.gray)
  axs[1].set_title("Convolved image")
  axs[1].axis('off')

def visualize_convolution_grid(img_list: np.ndarray, kernel: List[Any], figsize: Tuple[int,int] = (14,8)) -> None:
  fig, axs = plt.subplots(len(img_list), 2, figsize = figsize)

  for i in range(len(img_list)):
    # Original image
    axs[i][0].imshow(img_list[i], cmap=plt.cm.gray)
    axs[i][0].set_title("Original image")
    axs[i][0].axis('off')

    # Convoluted image
    convolved_image = convolve(np.array(img_list[i]), np.array(kernel))
    axs[i][1].imshow(convolved_image, cmap=plt.cm.gray)
    axs[i][1].set_title("Convolved image")
    axs[i][1].axis('off')

def convolve(image: np.ndarray, kernel: List[Any]) -> np.ndarray:
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
 	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output

def scatter_plot(images: np.ndarray, labels: np.ndarray, label_to_article: Dict[int,str], title: str = 'PCA of Fasion-MNIST Dataset', nbr_samples: int = 400) -> None:
  fig = plt.figure(figsize=(14, 10))
  fig.suptitle(title, fontsize=40)

  for i in range(10):
    # Select a subset of the images
    indices = np.where(labels == i)[0][:nbr_samples]

    # Display images in a 2D grid
    plt.scatter(images[indices][:,0], images[indices][:,1])
  plt.legend([label_to_article[i] for i in range(10)], prop={'size': 16});

def print_confusion_matrices(x, y, class_names):
    #fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(20,8))
    data = confusion_matrix(y, x)
    data_norm = confusion_matrix(y, x, normalize="true")

    print_confusion_matrix(data, class_names=class_names, title="Confusion matrix (without normalization)");#, ax=ax1);
    print_confusion_matrix(data_norm, class_names=class_names, title="Confusion matrix (with normalization)", float=True);#, ax=ax2);
  
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, title='Confusion matrix', float=False, ax=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )

    if ax is None:
      fig = plt.figure(figsize=figsize)
      fig.suptitle(title, fontsize=16)
    else:
      ax.set_title(title)

    if not float:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=ax)
    else:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2%", ax=ax)
        #raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    if ax is None:
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
    else:
      ax.set_ylabel('True label')
      ax.set_xlabel('Predicted label')