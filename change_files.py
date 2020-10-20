def change_file(filepath, new_file, new_string):

	with open(filepath, 'r') as f:     # load file
	    lines = f.read().splitlines()  # read lines

	with open(new_file, 'w') as f: 
	    f.write('\n'.join([new_string + line for line in lines]))  # write lines with '#' appended

filepath = "C:\\Users\\Johanna\\Documents\\GitHub\\PyTorch-YOLOv3\\dataset\\images\\train.txt"
new_file = "train.txt"
new_string = "dataset\\images\\train\\"

change_file(filepath, new_file, new_string)

filepath = "C:\\Users\\Johanna\\Documents\\GitHub\\PyTorch-YOLOv3\\dataset\\images\\valid.txt"
new_file = "valid.txt"
new_string = "dataset\\images\\valid\\"

change_file(filepath, new_file, new_string)