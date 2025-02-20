import sys
import numpy as np
import sklearn.cross_decomposition as skcd
import pandas as pd
import time

sys.path.append("bciflow/")

from bciflow.datasets.menggu import menggu


def channel_selection(dataset, channels_selected):
    new_dataset = {}
    ch_names = dataset["ch_names"]
    ch_idx = np.where(np.isin(ch_names, channels_selected))[0]
    new_dataset["X"] = dataset["X"][:, :, ch_idx, :]
    new_dataset["y"] = dataset["y"]
    dataset["ch_names"] = channels_selected
    return new_dataset


def filter_dataset(dataset, targets, time_window):
    new_dataset = {}
    targets = np.arange(targets[0], targets[1] + 1)
    idx = np.isin(dataset["y"], targets)
    new_dataset["X"] = dataset["X"][idx, :, :, time_window[0] : time_window[1]]
    new_dataset["y"] = dataset["y"][idx]
    return new_dataset


def build_target(target_freq, sfreq, total_time, num_harmonics=3):
    y = np.zeros((num_harmonics * 2, total_time))
    for i in range(1, num_harmonics + 1):
        y_sin = np.sin(
            2 * np.pi * target_freq * i * np.arange(total_time) / sfreq
        )
        y_cos = np.cos(
            2 * np.pi * target_freq * i * np.arange(total_time) / sfreq
        )
        y[(i - 1) * 2] = y_sin
        y[(i - 1) * 2 + 1] = y_cos
    return y


def cca(X, sfreq, total_time=5000, targets=(2, 5), num_harmonics=3):
    if type(targets) == tuple:
        targets = np.arange(targets[0], targets[1] + 1)
    # print(targets)

    y = np.array(
        [
            build_target(target, sfreq, total_time, num_harmonics)
            for target in targets
        ]
    )

    output = []
    times = []

    for trial in range(X.shape[0]):
        for band in range(X.shape[1]):
            X_ = X[trial, band, :, :]
            # measure time
            start = time.time()
            corr_coefs = np.array([])
            for y_ in y:
                cca_ = skcd.CCA(n_components=1)
                cca_.fit(X_.T, y_.T)
                X_c, y_c = cca_.transform(X_.T, y_.T)
                corr = np.corrcoef(X_c.T, y_c.T)[0, 1]
                corr_coefs = np.append(corr_coefs, corr)
            end = time.time()
            times.append(end - start)
            output.append(targets[np.argmax(corr_coefs)])

    return output, np.mean(times)


subjects = [i for i in range(1, 31)]
print("Subjects: ", subjects)
targets_window = 16
targets = [
    (i, i + targets_window - 1) for i in range(1, 60 + 1 - targets_window + 1)
]
print("Targets: ", targets)
depths = ["high", "low"]
print("Depths: ", depths)
channels_selected = ["PZ", "PO3", "PO4", "PO5", "PO6", "POZ", "O1", "O2", "OZ"]
print("Channels selected: ", channels_selected)
time_windows = [[0, i * 500] for i in range(1, 5)]
print("Time windows: ", time_windows)


def run(subject, depth):
    table = []
    dataset = menggu(subject=subject, path="data/", depth=[depth])
    dataset_ = channel_selection(dataset, channels_selected)
    for target in targets:
        print("    Running for target: ", target)
        for time_window in time_windows:
            print("      Running for time window: ", time_window)
            dataset__ = filter_dataset(dataset_, target, time_window)
            print("        Dataset shape: ", dataset__["X"].shape)
            output, time = cca(
                X=dataset__["X"],
                sfreq=1000,
                total_time=time_window[1] - time_window[0],
                targets=target,
                num_harmonics=10,
            )
            output = np.array(output)
            accuracy = (output == dataset__["y"]).sum() / len(dataset__["y"])
            print("        Accuracy: ", accuracy)
            table.append([subject, depth, target, time_window, accuracy, time])

    return table


# for each save the results in a table.csv
print("Running the experiments - Target window: ", targets_window)
for subject in subjects:
    for depth in depths:
        print(f"Running subject {subject} with depth '{depth}'")
        table = run(int(subject), depth)
        df = pd.DataFrame(
            table,
            columns=["subject", "depth", "target", "time_window", "accuracy", "time"],
        )
        df.to_csv(
            f"subject_{subject}_depth_{depth}_targets_{targets_window}_new.csv"
        )
