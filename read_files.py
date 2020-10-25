import matplotlib.pyplot as plt
import numpy as np

def read_map_data(file):
    
    file = open(file, "r")
    lines = file.read().split('\n') 
    classes = ['buffalo', 'elephant', 'rhino', 'zebra']

    buffalo_ap = []
    elephant_ap = []
    rhino_ap = []
    zebra_ap = []
    mAP = []

    for line in lines:
        if 'buffalo' in line:
            line = line.split("|")
            line = [x for x in line if x != '']
            buffalo_ap.append(float(line[2]))

        if 'elephant' in line:
            line = line.split("|")
            line = [x for x in line if x != '']
            elephant_ap.append(float(line[2]))

        if 'rhino' in line:
            line = line.split("|")
            line = [x for x in line if x != '']
            rhino_ap.append(float(line[2]))

        if 'zebra' in line:
            line = line.split("|")
            line = [x for x in line if x != '']
            zebra_ap.append(float(line[2]))
            
        if 'mAP' in line:
            line = line.split("mAP")

            mAP.append(float(line[1]))
            
    return  (buffalo_ap, elephant_ap, rhino_ap, zebra_ap, mAP), classes

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
            val_precision.append(float(line[1]))
        elif line[0] == "val_recall":
            val_recall.append(float(line[1]))
        elif line[0] == "val_mAP":
            val_mAP.append(float(line[1]))
        elif line[0] == "val_f1":
            val_f1.append(float(line[1]))

    return val_precision, val_recall, val_mAP, val_f1



def read_loss(filenames, doublet_epoch=None, doublet_batch=0):
	"""Read loss file. Delete doublets if program breaked after epoch doublet_epoch 
	and batch doublet_batch, and restart at start of same epoch. Epoch and batch 
	counted from 1."""

	batches = 0
	rows = 0
	dict_batches = []

	if doublet_epoch is not None:
		doublet_start = (doublet_epoch-1)*150+1
		doublet_end = (doublet_epoch-1)*150+doublet_batch

	
	for filename in filenames:
		f = open(filename, 'r')

		for line in f:
			if line.strip():
				rows = rows + 1

				# Check only once per batch
				if rows % 14 == 0:
					batches = batches + 1

					if doublet_epoch is not None and batches >= doublet_start and batches <= doublet_end:
						continue

					else:
						line_list = line.split(")(\'")
						line_list = [a for lines in line_list for a in lines.split("', ")]
						line_list[0] = line_list[0].strip("('")
						line_list[-1] = line_list[-1].strip("\n")
						line_list[-1] = line_list[-1].strip(")")

						dict_batch = {}
						for n in range(0, len(line_list), 2):
							dict_batch[line_list[n]] = float(line_list[n+1])

						dict_batches.append(dict_batch)

	batches = batches - doublet_batch
	return dict_batches, batches

def plot_map(data, classes, augmentation):
    """ Plot mAP for every epoch.
    Args: data is the data to be plotted (iterable with iterables)
          classes is a iterable describing the classes
          augmentation is a boolean indicating if the training was done 
                  with or without augmentation """
    
    fig, ax = plt.subplots() 
    epochs = np.arange(1, len(data[0])+1)
    for i in range(len(data)-1):
        print(data[i])
        plt.plot(epochs, data[i], label=classes[i])
    plt.plot(epochs, data[-1], label='mAP', c='black', linestyle='dashed')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average precision')
    if augmentation == True:
        ax.set_title("Average precision for each class \n when training with augmented data")
    elif augmentation == False:
        ax.set_title("Average precision for each class \n when training without augmented data")
    plt.legend()
    plt.savefig(f'plots/mAP_aug_{augmentation}.png')
    
    
def plot_training_metric(dict_batches, metric, nr_batches, augmentation):
	"""Plot a metric for each batch in each epoch.
	   dict_batches is the dict with all data.
	   metric is the name of the metric.
	   nr_batches is the total number of batches over time.
	   augmentation indicates if augmentation was used during training.
	"""

	plt.plot(range(nr_batches), [dict_batches[i][metric] for i in range(nr_batches)])
	plt.xlabel("batches")
	plt.ylabel(metric)
	if augmentation:
		plt.title(f"{metric} when training with augmented data")
	else:
		plt.title(f"{metric} when training without augmented data")

def plot_average(dict_batches, metric, nr_batches, augmentation):
	"""Plot average of a metric over different YOLO-layers for each batch in each epoch.
	   dict_batches is the dict with all data.
	   metric is the name of the metric.
	   nr_batches is the total number of batches over time.
	   augmentation indicates if augmentation was used during training.
	"""

	plt.plot(range(nr_batches), [sum([dict_batches[i]["" + metric + "_" + str(j)] for j in range(1,4)])/3 for i in range(nr_batches)])
	plt.xlabel("batches")
	plt.ylabel(metric)
	if augmentation:
		plt.title(f"Average {metric} when training with augmented data")
	else:
		plt.title(f"Average {metric} when training without augmented data")

def plot_val_metrics(data, metric, augmentation):
    """ Plot validation metrics for every epoch.
    Args: data is the data to be plotted
          metric is a string describing the plotted metric
          augmentation is a boolean indicating if the training was done
                  with or without augmentation """

    fig,ax = plt.subplots()
    epochs = np.arange(1, len(data)+1)
    plt.plot(epochs, data)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    if augmentation == True:
        ax.set_title(f"{metric} when training with augmented data")
    elif augmentation == False:
        ax.set_title(f"{metric} when training without augmented data")

    plt.savefig(f'plots/val_{metric}_aug_{augmentation}.png')


"""
dict_batches, nr_batches = read_loss(["log_files/loss_small_without_1.txt", 
									  "log_files/loss_small_without_2.txt",
									  "log_files/loss_small_without_3.txt"
									 ], 34, 115)

plt.figure(0)
plot_training_metric(dict_batches, 'loss', nr_batches, False)

plt.figure(1)
plot_average(dict_batches, 'precision', nr_batches, False)

plt.show()
"""
