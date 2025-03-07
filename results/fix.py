import pandas as pd
import numpy as np

# Read all tables
subjects = [i for i in range(1, 31)]
target_windows = [2, 4, 8, 16]
depths = ["low", "high"]


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


# remove all files ended with _itr.csv _new.csv or with a number
# import os
# import re

# for file in os.listdir("results"):
#     if re.search(r"(_itr.csv|_new.csv|\d+)", file):
#         os.remove(f"results/{file}")


for target_window in target_windows:
    for depth in depths:
        for subject in subjects:
            dataset = pd.read_csv(
                f"results/subject_{subject}_depth_{depth}_targets_{target_window}_new.csv"
            )
            dataset["itr"] = dataset.apply(
                lambda row: itr(
                    row["accuracy"],
                    target_window,
                    float(row["time_window"].strip("[]").split(",")[1]) / 1000 + row["time"],
                ),
                axis=1,
            )
            dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]
            dataset.to_csv(
				f"results/subject_{subject}_depth_{depth}_targets_{target_window}_pls.csv",
				index=False,
			)