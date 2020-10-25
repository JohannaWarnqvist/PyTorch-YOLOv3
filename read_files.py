import matplotlib.pyplot as plt

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

def plot_things(dict_batches, name, nr_batches):

	plt.plot(range(nr_batches), [dict_batches[i][name] for i in range(nr_batches)])
	plt.show()

def plot_average(dict_batches, name, nr_batches):

	plt.plot(range(nr_batches), [sum([dict_batches[i]["" + name + "_" + str(j)] for j in range(1,4)])/3 for i in range(nr_batches)])

dict_batches, nr_batches = read_loss(["log_files/loss_small_without_1.txt", 
									  "log_files/loss_small_without_2.txt",
									  "log_files/loss_small_without_3.txt"
									 ], 34, 115)


plt.figure(0)
plot_things(dict_batches, 'loss', nr_batches)

plt.figure(1)
plot_average(dict_batches, 'precision', nr_batches)

plt.show()