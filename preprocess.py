
def process_images(data):
	"""images: flatten all but 0th dim, rescale to [0,1]
	"""
	data.shape = (data.shape[0], -1)
	return 1 - data/255

def process_labels(labels):
	"""labels: cast int to string of len 1
	"""
	return labels.astype("<U1", copy=False)
	