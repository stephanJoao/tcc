from calendar import c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sympy import false, rotations

def read_mat(filename):
	with h5py.File(filename, "r") as f:
		data = np.asarray(f["Sub_score"])
	return data

if __name__ == '__main__':
	# load confort dataset
	df = read_mat("data/Sub_score.mat")
	print(df.shape)
	target_window = 8

	data = df[0, :, 0, :]
	data_mean = data.mean(axis=1)

	confort = []
	for i in range(0, 61 - target_window):
		data = df[0, i:i+target_window, 1, :]
		data = data.mean(axis=0)
		# print(data.shape)
		# exit()
		data_mean = data.mean(axis=0)
		confort.append(data_mean)


	# plt.figure()
	# plt.plot(data_mean)
	# plt.show()
	
	subjects = [i for i in range(1, 31)]
	depth = "high"
	time_windows = [f"[0, {i*500}]" for i in range(1, 5)]
	# shades of pastel blue from light to dark
	shades_of_blue = ["#B3C7D6", "#8AA9C1", "#5E8CB3", "#2E6E9E"]
	print(time_windows)

	# plot accuracies mean for each freqeuncy window
	fig, ax1 = plt.subplots()
	for time_window in time_windows:
		accuracies = []
		for subject in subjects:
			# load dataset
			dataset = pd.read_csv(f"subject_{subject}_depth_{depth}_targets_{target_window}.csv")
			# filter all lines with time_window
			dataset = dataset[dataset['time_window'] == time_window]
			accuracies.append(dataset['accuracy'])
		accuracies = np.array(accuracies)
		print(accuracies)
		accuracies_mean = accuracies.mean(axis=0)
		# scale of accuracies from 0 to 1
		color_idx = time_windows.index(time_window)
		ax1.plot(dataset["target"], accuracies_mean, color=shades_of_blue[color_idx], label=f"Time window: {time_window}")
		ax1.tick_params(axis='x', labelrotation=90)

		


		ax1.set_ylim(0, 1)

	ax2 = ax1.twinx()
	ax2.plot(confort, color='red', label='Conforto')
	ax2.set_ylim(0, 5)
	
	ax1.set_xlim(0, 61 - target_window - 1)
	ax1.set_xlabel('Frequency window')
	ax1.set_ylabel('Accuracy')
	ax2.set_ylabel('Conforto')
	fig.legend()

	# ax1.grid(True)
	# ax2.grid(False)
	plt.xticks(rotation=40)
	plt.tight_layout()

	plt.show()

