from os import mkdir
from os.path import isdir, join

def data_exists(dirname):
	if isdir(dirname):
		return True
	else:
		mkdir(dirname)
		return False

def download(base, fnames, dirname):
	import requests
	for fname in fnames:
		url = base + fname
		data = requests.get(url).content
		local_path = join(dirname, fname)
		with open(local_path, "wb") as f:
			f.write(data)
	del requests

def main():

	data_folder_name = "data"
	url_base = "http://yann.lecun.com/exdb/mnist/"
	files = [
		"train-labels-idx1-ubyte.gz", 
		"t10k-labels-idx1-ubyte.gz", 
		"train-images-idx3-ubyte.gz", 
		"t10k-images-idx3-ubyte.gz"
	]
	
	if not data_exists(data_folder_name):
		download(url_base, files, data_folder_name)

	import filecmp
	for fname in files:
		print(filecmp.cmp("data/" + fname, "written_data/" + fname))
	del filecmp

if __name__ == "__main__":
	main()
