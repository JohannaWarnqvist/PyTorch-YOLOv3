iport matplotlib.pyplot as plt

def read_validation_data(file):
    
    val_precision = []
    val_recall = []
    val_mAP = []
    val_f1 = []
    

    f = open(file, "r")
    lines = f.read().split('\n') 
    lines = [x for x in lines if x !='']
    lines  = [x.split(': ') for x in lines] 


    for line in lines:
        if line[0] == "val_precision":
            val_precision.append(line[1])
        elif line[0] == "val_recall":
            val_recall.append(line[1])
        elif line[0] == "val_mAP":
            val_mAP.append(line[1])
        elif line[0] == "val_f1":
            val_f1.append(line[1])

    return val_precision, val_recall, val_mAP, val_f1



def read_loss(filenames):

	batches = 0
	dict_batches = []
	
	for filename in filenames:
		f = open(filename, 'r')

		for line in f:
			if line.strip():
				batches = batches + 1

				line_list = line.split(")(\'")
				line_list = [a for lines in line_list for a in lines.split("', ")]
				line_list[0] = line_list[0].strip("('")
				line_list[-1] = line_list[-1].strip("\n")
				line_list[-1] = line_list[-1].strip(")")

				dict_batch = {}
				for n in range(0, len(line_list), 2):
					dict_batch[line_list[n]] = float(line_list[n+1])

				dict_batches.append(dict_batch)

	return dict_batches, batches

def plot_things(dict_batches, name, nr_batches):

	plt.plot(range(nr_batches), [dict_batches[i][name] for i in range(nr_batches)])
	plt.show()

def plot_average(dict_batches, name, nr_batches):

	plt.plot(range(nr_batches), [sum([dict_batches[i]["" + name + "_" + str(j)] for j in range(1,4)])/3 for i in range(nr_batches)])

dict_batches, nr_batches = read_loss(["log_files/loss_small_without_1.txt", 
									  "log_files/loss_small_without_2.txt",
									  "log_files/loss_small_without_3.txt"
									 ])

plt.figure(0)
plot_things(dict_batches, 'loss', nr_batches)

plt.figure(1)
plot_average(dict_batches, 'precision', nr_batches)

plt.show()
