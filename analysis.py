import enum
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py


def read_mat(filename):
    with h5py.File(filename, "r") as f:
        data = np.asarray(f["Sub_score"])
    return data


def itr(acc, n_targets, time):
    if acc != 1.0:
        return (
            60
            / time
            * (
                np.log2(n_targets)
                + acc * np.log2(acc)
                + (1 - acc) * np.log2((1 - acc) / (n_targets - 1))
            )
        )
    else:
        return 60 / time * np.log2(n_targets)


if __name__ == "__main__":
    subjects = [i for i in range(1, 31)]
    target_window = 8
    depth = "low"
    time_windows = [f"[0, {i*500}]" for i in range(1, 5)]
    shades_of_blue = ["#B3C7D6", "#8AA9C1", "#5E8CB3", "#2E6E9E"]

    hilo = {
        "low": 0,
        "high": 1,
    }
    comfort_data = read_mat("data/Sub_score.mat")
    comfort_data = comfort_data[0, :, hilo[depth], :]
    comfort_data = comfort_data.mean(axis=1)
    comfort_data_mean = np.convolve(
        comfort_data, np.ones((target_window,)) / target_window, mode="valid"
    )

    # fig, ax1 = plt.subplots(figsize=(11, 5))
    # for time_window in time_windows:
    #     accuracies = []
    #     for subject in subjects:
    #         # load dataset
    #         dataset = pd.read_csv(
    #             f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
    #         )
    #         dataset = dataset[dataset["time_window"] == time_window]
    #         accuracies.append(dataset["accuracy"])
    #     accuracies = np.array(accuracies)
    #     accuracies_mean = accuracies.mean(axis=0)
    #     color_idx = time_windows.index(time_window)
    #     transfer_rate = [itr(acc, target_window, float(time_window.strip("[]").split(",")[1])/1000) for acc in accuracies_mean]
    #     ax1.plot(
    #         dataset["target"],
    #         transfer_rate,
    #         color=shades_of_blue[color_idx],
    #         label=f"Tempo: {time_window.strip("[]").split(",")[1]}ms",
    #     )
    #     ax1.tick_params(axis="x", labelrotation=90)
    #     # ax1.set_ylim(0, 1)

    # ax2 = ax1.twinx()
    # ax2.plot(comfort_data_mean, color="red", label="Conforto")
    # ax2.set_ylim(0, 5)

    # ax1.set_xlim(0, 60 - target_window)
    # ax1.set_xlabel("Janelas de frequência")
    # ax1.set_ylabel("Acurácia")
    # ax2.set_ylabel("Conforto")
    # fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2))

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"plots/{depth}_targets_{target_window}.png")

    # fig, ax1 = plt.subplots(figsize=(11, 5))
    # for time_window in time_windows:
    #     accuracies = []
    #     for subject in subjects:
    #         # load dataset
    #         dataset = pd.read_csv(
    #             f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
    #         )
    #         dataset = dataset[dataset["time_window"] == time_window]
    #         accuracies.append(dataset["accuracy"])
    #     accuracies = np.array(accuracies)
    #     accuracies_mean = accuracies.mean(axis=0)
    #     plt.scatter(
    #         comfort_data_mean,
    #         accuracies_mean
    #     )

    # plt.xlabel("Conforto")
    # plt.ylabel("Acurácia")
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    all_values = []
    for depth in ["low", "high"]:
        comfort_data = read_mat("data/Sub_score.mat")
        comfort_data = comfort_data[0, :, hilo[depth], :]
        comfort_data = comfort_data.mean(axis=1)
        for target_window in [2, 4, 8, 16]:
            targets = [
                (i, i + target_window - 1)
                for i in range(1, 60 + 1 - target_window + 1)
            ]
            comfort_data_mean = np.convolve(
                comfort_data,
                np.ones((target_window,)) / target_window,
                mode="valid",
            )
            for time_window in time_windows:
                accuracies = []
                for subject in subjects:
                    # load dataset
                    dataset = pd.read_csv(
                        f"results/subject_{subject}_depth_{depth}_targets_{target_window}_ITR.csv"
                    )
                    dataset = dataset[dataset["time_window"] == time_window]
                    accuracies.append(dataset["itr"])
                accuracies = np.array(accuracies)

                # print(accuracies.shape)
                # exit()

                values = accuracies.mean(axis=0)
                # values = np.array(
                #     [
                #         itr(
                #             acc,
                #             target_window,
                #             float(time_window.strip("[]").split(",")[1])
                #             / 1000,
                #         )
                #         for acc in values
                #     ]
                # )
                for idx, value in enumerate(values):
                    all_values.append(
                        ((depth, target_window, targets[idx], time_window), (comfort_data_mean[idx], value))
                    )

                # print values only if some value is greater than 140
                # if np.max(values) > 140:
                #     print(
                #         f"Depth: {depth}, Target: {target_window}, Time:"
                #         f" {time_window}"
                #     )
                    # print(values)
                # plt.scatter(comfort_data_mean, values)

    # all_values = np.array(all_values)
    # print(all_values)

    # in really light grey
    plt.scatter(
        [x[1][0] for x in all_values],
        [x[1][1] for x in all_values],
        color="#D3D3D3",
    )

    non_dom = []
    # print all non-dominated points
    for i in range(len(all_values)):
        dominated = False
        for j in range(len(all_values)):
            if i == j:
                continue
            if all_values[i][1][0] <= all_values[j][1][0] and all_values[i][1][1] <= all_values[j][1][1]:
                dominated = True
                break
        if not dominated:
            non_dom.append(all_values[i])

    # in black 
    plt.scatter(
        [x[1][0] for x in non_dom],
        [x[1][1] for x in non_dom],
        color="black",
    )

    # order non-dominated points by comfort and print
    non_dom = sorted(non_dom, key=lambda x: x[1][0])
    for i in range(len(non_dom)):
        print(f"Depth: {non_dom[i][0][0]}, Target: {non_dom[i][0][1]}, Time: {non_dom[i][0][3]}")
        print(f"Comfort: {non_dom[i][1][0]}, ITR: {non_dom[i][1][1]}")
        print()

    # plt.legend()
    plt.xlabel("Conforto")
    plt.ylabel("ITR")
    plt.show()
