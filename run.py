from os.path import join
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from download import data_exists, download
from parse import parse_labels, parse_images
from render import browse
from preprocess import process_images, process_labels

DATA_FOLDER_NAME = "data"
URL_BASE = "http://yann.lecun.com/exdb/mnist/"
FILES = {
	"train_labels": "train-labels-idx1-ubyte.gz", 
	"test_labels": "t10k-labels-idx1-ubyte.gz", 
	"train_images": "train-images-idx3-ubyte.gz", 
	"test_images": "t10k-images-idx3-ubyte.gz"
}

def plot_coefficients(model):
	fig, axes = plt.subplots(4, 4)
	vmin, vmax = model.coefs_[0].min(), model.coefs_[0].max()
	for coef, ax in zip(model.coefs_[0].T, axes.ravel()):
		ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5*vmin, vmax=.5*vmax)
		ax.set_xticks(())
		ax.set_yticks(())
	plt.show()

def main():

	if not data_exists(DATA_FOLDER_NAME):
		download(URL_BASE, FILES.values(), DATA_FOLDER_NAME)

	train_labels_path = join(DATA_FOLDER_NAME, FILES["train_labels"])
	test_labels_path = join(DATA_FOLDER_NAME, FILES["test_labels"])
	train_images_path = join(DATA_FOLDER_NAME, FILES["train_images"])
	test_images_path = join(DATA_FOLDER_NAME, FILES["test_images"])
	
	train_labels = parse_labels(train_labels_path)
	test_labels = parse_labels(test_labels_path)

	train_images = parse_images(train_images_path)
	test_images = parse_images(test_images_path)

	train_labels = process_labels(train_labels)
	test_labels = process_labels(test_labels)

	train_images = process_images(train_images)
	test_images = process_images(test_images)

	model = MLPClassifier(hidden_layer_sizes=(200,), random_state=1, verbose=True)
	model.fit(train_images, train_labels)

	print("Training set score: {:0.2f}".format(model.score(train_images, train_labels)))
	print("Test set score: {:0.2f}".format(model.score(test_images, test_labels)))

	plot_coefficients(model)

	return train_images, train_labels, test_images, test_labels, model

if __name__ == "__main__":
	main()
