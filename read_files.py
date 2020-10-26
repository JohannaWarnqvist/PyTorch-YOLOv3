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
        plt.plot(epochs, data[i], label=classes[i])
    plt.plot(epochs, data[-1], label='mAP', c='black', linestyle='dashed')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average precision')
    if augmentation == True:
        ax.set_title("Average precision for each class \n when training with augmented data")
    elif augmentation == False:
        ax.set_title("Average precision for each class \n when training without augmented data")
    print(f"Best mAP:{max(data[-1])}, at epoch:{np.argmax(data[-1])}, aug={augmentation}")
    plt.legend()
    plt.savefig(f'plots/mAP_aug_{augmentation}.png')
    
    
def plot_training_metric(dict_batches, dict_batches_aug, metric, nr_batches,  all_batches = True):
    """Plot a metric for each batch in each epoch.
       dict_batches is the dict with all data.
       metric is the name of the metric.
       nr_batches is the total number of batches over time.
       augmentation indicates if augmentation was used during training.
    """

    if all_batches:
        plt.plot(range(nr_batches), [dict_batches[i][metric] for i in range(nr_batches)], label='No augmentation')
        plt.plot(range(nr_batches), [dict_batches_aug[i][metric] for i in range(nr_batches)], label='Using augmentation')
        plt.xlabel("Batch")
        plt.ylabel(metric)
    else:
        plt.plot(range(1,int(nr_batches/150)+1), [np.mean([dict_batches[i][metric] for i in range(j*150, 150*j+150)]) for j in range(int(nr_batches/150))], label='No augmentation')
        plt.plot(range(1,int(nr_batches/150)+1), [np.mean([dict_batches_aug[i][metric] for i in range(j*150, 150*j+150)]) for j in range(int(nr_batches/150))], label='Using augmentation')
        plt.xlabel("Epoch")
        plt.ylabel(f"Average {metric} per batch")
    
    
    plt.title(f"{metric} for training ")
    plt.legend()

    plt.savefig(f'plots/train_{metric}_.png')

def plot_average(dict_batches, dict_batches_aug, metric, nr_batches, all_batches = True):
    """Plot average of a metric over different YOLO-layers for each batch in each epoch.
       dict_batches is the dict with all data.
       metric is the name of the metric.
       nr_batches is the total number of batches over time.
       augmentation indicates if augmentation was used during training.
    """

    if all_batches:
        plt.plot(range(nr_batches), [sum([dict_batches[i]["" + metric + "_" + str(j)] for j in range(1,4)])/3 for i in range(nr_batches)], label='No augmentation')
        plt.plot(range(nr_batches), [sum([dict_batches_aug[i]["" + metric + "_" + str(j)] for j in range(1,4)])/3 for i in range(nr_batches)], label='Using augmentation')
        plt.xlabel("Batch")
        plt.ylabel(metric)
    else:
        plt.plot(range(1,int(nr_batches/150)+1), [np.mean([np.mean([dict_batches[i]["" + metric + "_" + str(j)] for j in range(1,4)]) for i in range(k*150, 150*k+150)]) for k in range(int(nr_batches/150))], label='No augmentation')
        plt.plot(range(1,int(nr_batches/150)+1), [np.mean([np.mean([dict_batches_aug[i]["" + metric + "_" + str(j)] for j in range(1,4)]) for i in range(k*150, 150*k+150)]) for k in range(int(nr_batches/150))], label='Using augmentation')
        plt.xlabel("Epoch")
        plt.ylabel(f"Average {metric}")

        

    plt.legend()
    plt.title(f"Average {metric} when training")
    plt.savefig(f'plots/train_{metric}_.png')


def plot_val_metrics(data, data_aug, metric):
    """ Plot validation metrics for every epoch.
    Args: data is the data to be plotted
          metric is a string describing the plotted metric
          augmentation is a boolean indicating if the training was done
                  with or without augmentation """

    fig,ax = plt.subplots()
    epochs = np.arange(1, len(data)+1)
    plt.plot(epochs, data, label='not using data augmentation')
    plt.plot(epochs, data_aug, label='using data augmentation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric}')
    #if augmentation == True:
     #   ax.set_title(f"{metric} when training with augmented data")
   # elif augmentation == False:
    ax.set_title(f"{metric} for validation")
    plt.legend()

    plt.savefig(f'plots/val_{metric}.png')

    # print mest metric and the metric value at best epoch (based on mAP)
    print(f"Best {metric}:{max(data)}, at epoch:{np.argmax(data)+1}, aug=False")
    print(f"Best {metric}:{max(data_aug)}, at epoch:{np.argmax(data_aug)+1}, aug=True")
    print(f"{metric} at best mAP epoch:{data_aug[56]} (aug)")
    print(f"{metric} at best mAP epoch:{data[57]} (NO aug) ")

