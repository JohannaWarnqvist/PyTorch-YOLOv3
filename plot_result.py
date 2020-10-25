from read_files import *

val_data_aug = read_validation_data('lolboi/log_files/validation.txt')
val_data = read_validation_data('lolboi/log_files/validation_without.txt')
metric = ["precision", "recall", "mAP", "f1"]


for d1, d2, m in zip(val_data, val_data_aug, metric):
    plot_val_metrics(d1, d2, m, )
    

AP, classes = read_map_data('lolboi/log_files/mAP.txt')
plot_map(AP, classes, augmentation=True)



dict_batches_aug, nr_batches = read_loss(['lolboi/log_files/loss.txt'], 25,37 )
dict_batches, nr_batches = read_loss(['lolboi/log_files/loss_small_without_1.txt', 
                                        'lolboi/log_files/loss_small_without_2.txt', 
                                        'lolboi/log_files/loss_small_without_3.txt'], 
                                        34, 115 )

plt.figure()
plot_training_metric(dict_batches, dict_batches_aug, 'loss', nr_batches=nr_batches,  all_batches=False)

plt.figure()
plot_average(dict_batches, dict_batches_aug, 'precision', nr_batches,  all_batches=False)


plt.show()
