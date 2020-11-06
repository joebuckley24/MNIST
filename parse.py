import gzip
import numpy as np

def parse_labels(labels_path):
	"""Read in labels for training or testing data.

	description from file source (http://yann.lecun.com/exdb/mnist/):

	TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	0004     32 bit integer  60000            number of items
	0008     unsigned byte   ??               label
	0009     unsigned byte   ??               label
	........
	xxxx     unsigned byte   ??               label

	The labels values are 0 to 9.
	"""
	with gzip.open(labels_path, 'rb') as f:
		magic_num = int.from_bytes(f.read(4), "big")
		num_labs = int.from_bytes(f.read(4), "big")
		labs = np.frombuffer(f.read(num_labs), 'u1')
		eof = f.read(1)
	assertion = "Expected exactly {} labels based on bytes 0004 to 0008".format(num_labs)
	assert labs.shape == (num_labs,) and eof == b'', assertion
	return labs

def parse_images(images_path):
	"""Read in images for training or testing data.

	description from file source (http://yann.lecun.com/exdb/mnist/):

	TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000803(2051) magic number
	0004     32 bit integer  60000            number of images
	0008     32 bit integer  28               number of rows
	0012     32 bit integer  28               number of columns
	0016     unsigned byte   ??               pixel
	0017     unsigned byte   ??               pixel
	........
	xxxx     unsigned byte   ??               pixel

	Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)."""
	with gzip.open(images_path, 'rb') as f:
		magic_num = int.from_bytes(f.read(4), "big")
		num_images = int.from_bytes(f.read(4), "big")
		num_rows = int.from_bytes(f.read(4), "big")
		num_cols = int.from_bytes(f.read(4), "big")
		num_bytes = num_images*num_rows*num_cols
		images = np.frombuffer(f.read(num_bytes), 'u1')
		eof = f.read(1)
	assertion = "Expected exactly {} {}-{} images based on bytes 0004 to 0016".format(num_images, num_rows, num_cols)
	assert images.shape == (num_bytes,) and eof == b'', assertion
	images.shape = (num_images, num_rows, num_cols)
	return images

def main():

	train_labels_path = 'data/train-labels-idx1-ubyte.gz'
	test_labels_path = 'data/t10k-labels-idx1-ubyte.gz'
	train_labels = parse_labels(train_labels_path)
	test_labels = parse_labels(test_labels_path)
	print("shape: {}".format(train_labels.shape))
	print("head: {}".format(train_labels[:10]))
	print("tail: {}".format(train_labels[-10:]))
	print("shape: {}".format(test_labels.shape))
	print("head: {}".format(test_labels[:10]))
	print("tail: {}".format(test_labels[-10:]))

	train_images_path = 'data/train-images-idx3-ubyte.gz'
	test_images_path = 'data/t10k-images-idx3-ubyte.gz'
	train_images = parse_images(train_images_path)
	test_images = parse_images(test_images_path)
	with np.printoptions(threshold=np.inf, linewidth=np.inf):
		print("shape: {}".format(train_images.shape))
		print("first img")
		print(train_images[0])
		print("last img")
		print(train_images[-1])
		print("individual img shape: {}".format(train_images[0].shape))
		print("shape: {}".format(test_images.shape))
		print("first img")
		print(test_images[0])
		print("last img")
		print(test_images[-1])
		print("individual img shape: {}".format(test_images[0].shape))
		

if __name__ == "__main__":
	main()