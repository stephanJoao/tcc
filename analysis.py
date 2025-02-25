from dis import disco
import dis
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def proportional_comfort(comfort, total_time):
    discomfort = 5 - comfort
    discomfort = discomfort * total_time / 5000
    comfort = 5 - discomfort
    return comfort


def read_mat(filename):
    with h5py.File(filename, "r") as f:
        data = np.asarray(f["Sub_score"])
    return data


if __name__ == "__main__":
    subjects = [i for i in range(1, 31)]
    target_window = 16
    depth = "high"
    metric = "itr"
    time_windows = [f"[0, {i*500}]" for i in range(1, 5)]
    shades_of_blue = ["#B3C7D6", "#8AA9C1", "#5E8CB3", "#2E6E9E"]

    hilo = {
        "low": 0,
        "high": 1,
    }

    # CONFORTO E METRICA

    # fig, ax1 = plt.subplots(figsize=(11, 5))
    # comfort_data = read_mat("data/Sub_score.mat")
    # comfort_data = comfort_data[0, :, hilo[depth], :]
    # comfort_data = comfort_data.mean(axis=1)
    # comfort_data_mean = np.convolve(
    #     comfort_data, np.ones((target_window,)) / target_window, mode="valid"
    # )
    # for time_window in time_windows:
    #     values = []
    #     for subject in subjects:
    #         # load dataset
    #         dataset = pd.read_csv(
    #             f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
    #         )
    #         dataset = dataset[dataset["time_window"] == time_window]
    #         values.append(dataset[metric])
    #     values = np.array(values)
    #     values_mean = values.mean(axis=0)
    #     color_idx = time_windows.index(time_window)
    #     if depth == "low":
    #         ax1.plot(
    #             dataset["target"],
    #             values_mean,
    #             color=shades_of_blue[color_idx],
    #             label=f"Tempo: {time_window.strip('[]').split(',')[1]}ms",
    #         )
    #     else:
    #         ax1.plot(
    #             dataset["target"],
    #             values_mean,
    #             color=shades_of_blue[color_idx],
    #             label=f"Tempo: {time_window.strip("[]").split(",")[1]}ms",
    #         )
    #     ax1.tick_params(axis="x", labelrotation=90)
    #     if metric == "accuracy":
    #         ax1.set_ylim(0, 1)

    # ax2 = ax1.twinx()
    # ax2.plot(comfort_data_mean, color="red", label="Conforto")
    # ax2.set_ylim(0, 5)
    # if metric == "accuracy":
    #     ax1.axhline(y=1 / target_window, color="black", linestyle="--", alpha=0.5, label="Aleatório")
    # ax1.set_xlim(0, 60 - target_window)
    # ax1.set_xlabel("Janelas de frequência")
    # if metric == "itr":
    #     ax1.set_ylabel("ITR")
    # else:
    #     ax1.set_ylabel("Acurácia")
    # ax2.set_ylabel("Conforto")
    # fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2))

    # plt.tight_layout()
    # # plt.savefig(f"plots/{metric}_{depth}_targets_{target_window}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    # plt.show()
   

    # PARETO TUDO

    fig, ax1 = plt.subplots(figsize=(7, 5))
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
                values = []
                for subject in subjects:
                    dataset = pd.read_csv(
                        f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
                    )
                    dataset = dataset[dataset["time_window"] == time_window]
                    values.append(dataset[metric])
                values = np.array(values)

                values = values.mean(axis=0)

                print(time_window.strip("[]").split(",")[1])
                for idx, value in enumerate(values):
                    all_values.append(
                        (
                            (depth, target_window, targets[idx], time_window),
                            (proportional_comfort(comfort_data_mean[idx], float(time_window.strip("[]").split(",")[1])), value),
                            # (comfort_data_mean[idx], value),
                        )
                    )

    for i in range(len(all_values)):
        plt.scatter(
            all_values[i][1][0],
            all_values[i][1][1],
            color="lightgrey",
            marker="o" if all_values[i][0][0] == "low" else "^",
            zorder=1,
        )
    
    non_dom = []    
    for i in range(len(all_values)):
        dominated = False
        for j in range(len(all_values)):
            if i == j:
                continue
            if (
                all_values[i][1][0] <= all_values[j][1][0]
                and all_values[i][1][1] <= all_values[j][1][1]
            ):
                dominated = True
                break
        if not dominated:
            non_dom.append(all_values[i])

    label_high = False
    label_low = False

    for point in non_dom:
        if point[0][0] == "high":
            if not label_high:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    marker="^",
                    label="Amplitude alta",
                    zorder=3,
                )
                label_high = True
            else:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    marker="^",
                    zorder=3,
                )
        else:
            if not label_low:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    label="Amplitude baixa",
                    zorder=3,
                )
                label_low = True
            else:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    zorder=3,
                )

    # order non-dominated points by comfort and print
    non_dom = sorted(non_dom, key=lambda x: x[1][0])
    for i in range(len(non_dom)):
        print(
            f"Depth: {non_dom[i][0][0]}, Target: {non_dom[i][0][1]}, Time:"
            f" {non_dom[i][0][3]}"
        )
        print(f"Comfort: {non_dom[i][1][0]}, ITR: {non_dom[i][1][1]}")
        print()

    # plt.ylim((1/target_window) - 0.1, 1)
    # plt.ylim(0, 1)
    plt.axhline(y=1 / target_window, color="black", linestyle="--", alpha=0.5, label="Aleatório")

    plt.legend()
    plt.grid(color="grey", linestyle="--", linewidth=0.5, zorder=2)
    plt.xlabel("Conforto")
    plt.ylabel("Acurácia")
    plt.savefig(f"plots/pareto_{metric}_targets_all.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
