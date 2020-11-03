from matplotlib import pyplot as plt
import numpy as np
from parse import parse_labels, parse_images

def display_image(array, label):
	img = plt.imshow(255-array, cmap='gray', vmin=0, vmax=255)
	txt = plt.text(0, 1, 'label: ' + str(label))
	return img, txt

def browse(images, labels):
	while True:
		index = input('enter index number of image/label combination (blank to quit): ')
		if index == '':
			break
		else:
			index = int(index)
			plt.figure(1)
			display_image(images[index], labels[index])
			plt.show()
			plt.close()


def main():
	train_labels_path = 'data/train-labels-idx1-ubyte.gz'
	test_labels_path = 'data/t10k-labels-idx1-ubyte.gz'
	train_labels = parse_labels(train_labels_path)
	test_labels = parse_labels(test_labels_path)

	train_images_path = 'data/train-images-idx3-ubyte.gz'
	test_images_path = 'data/t10k-images-idx3-ubyte.gz'
	train_images = parse_images(train_images_path)
	test_images = parse_images(test_images_path)

	browse(train_images, train_labels)

	# with np.printoptions(threshold=np.inf, linewidth=np.inf):
	# 	print("shape: {}".format(train_images.shape))
	# 	print("first img")
	# 	print(train_images[0])
	# 	print("last img")
	# 	print(train_images[-1])
	# 	print("individual img shape: {}".format(train_images[0].shape))
	# 	print("shape: {}".format(test_images.shape))
	# 	print("first img")
	# 	print(test_images[0])
	# 	print("last img")
	# 	print(test_images[-1])
	# 	print("individual img shape: {}".format(test_images[0].shape))

if __name__ == "__main__":
	main()