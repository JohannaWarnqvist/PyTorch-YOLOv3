from read_files import *


# Read the files for without augmentation and plot the data

dict_batches, nr_batches = read_loss(["log_files/loss_small_without_1.txt", 
                                      "log_files/loss_small_without_2.txt",
                                      "log_files/loss_small_without_3.txt"
                                     ], 34, 115)

plt.figure()
plot_training_metric(dict_batches, 'loss', nr_batches, False, all_batches=False)

plt.figure()
plot_average(dict_batches, 'precision', nr_batches, False, False)


mAP_data, classes = read_map_data("log_files/mAP_without.txt")
plt.figure()
plot_map(mAP_data, classes, False)

plt.figure()

val_precision, val_recall, val_mAP, val_f1 = read_validation_data("log_files/validation_without.txt")

plt.figure()
plot_val_metrics(val_precision, 'precision', False)
plt.figure()
plot_val_metrics(val_recall, 'recall', False)
plt.figure()
plot_val_metrics(val_mAP, 'mAP', False)
plt.figure()
plot_val_metrics(val_f1, 'f1', False)

plt.show()