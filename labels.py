import platform
import pandas as pd
import numpy as np
import constants

import matplotlib
import matplotlib.pyplot as plt

from utils.feature_loader import load_eye_tracking_data

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')

color_plate = ["#D9D0DE", "#BC8DA0", "#A04668", "#AB4967", "#872424"]  # "#A04668", "#AB4967", "#0C1713"
_, y = load_eye_tracking_data(number_of_classes=5, load_preprocessed=False, label_name=["duration_estimate", "ppot", "valence", "arousal", "flow", "task_difficulty"], include_meta_label=True)

print(f"sahpe of y: {y.shape}")  # 336, 6

## label distribution and binary distribution
# labels = ["very slow", "slow", "medium", "fast", "very fast"]
# _, sizes = np.unique(y["ppot"].values, return_counts=True)
#
# _, bin_sizes = np.unique(np.where(y["ppot"].values > 2, 1, 0), return_counts=True)
#
# fig, ax = plt.subplots(1, 2)
# ax[0].pie(sizes, labels=labels, autopct='%1.1f%%')
# ax[0].title.set_text("Distribution of PPOT labels")
#
# ax[1].pie(bin_sizes, labels=["slow", "fast"], autopct='%1.1f%%')
# ax[1].title.set_text("Distribution of binary PPOT labels")

## duration estimation distribution after time
# fig, ax = plt.subplots(1, 2)
# for i in [1, 3, 5]:
#     ax[0].hist(y.loc[y["time"] == i]["duration_estimate"], alpha=0.5, label=f"Time: {i} min")
#
#     tmp = (y.loc[y["time"] == i]["duration_estimate"].values - i) / i * 100
#     ax[1].hist(tmp, alpha=0.5, label=f"Time: {i} min")
#     # ax[1].title.set_text("Relative Time Estimation")
# ax[0].legend()
# ax[1].legend()
# ax[0].title.set_text("Duration Estimate")
# ax[0].set_xlabel("Duration Estimate [min]")
# ax[0].set_ylabel("Occurances")
# ax[1].title.set_text("Relative Time Estimation")
# ax[1].set_xlabel("Relative Time Estimation [%]")
# ax[1].set_ylabel("Occurances")

## weight data by time
# labels = ["very slow", "slow", "medium", "fast", "very fast"]
# # weighted by time
# all_size_1 = np.zeros((5,))
# all_size_3 = np.zeros((5,))
# all_size_5 = np.zeros((5,))
# for i in [5]:
#     label_num, sizes = np.unique(y.loc[y["time"] == i]["ppot"].values, return_counts=True)
#     all_size_1[label_num.astype(int)] += sizes * i
#
#     label_nums, bin_sizes = np.unique(np.where(y.loc[y["time"] == i]["ppot"].values > 2, 3, 1), return_counts=True)
#     all_size_3[label_nums.astype(int)] += bin_sizes  # * i  # include time weighting
#
#     tmp = y.loc[y["time"] == i]["ppot"].values
#     tmp[tmp > 2] = 3
#     tmp[tmp < 2] = 1
#     label_nums_3, bin_sizes_3 = np.unique(tmp, return_counts=True)
#     all_size_5[label_nums_3.astype(int)] += bin_sizes_3 * i
#
# fig, ax = plt.subplots(1, 3)
# ax[0].pie(all_size_1, labels=labels, autopct='%1.1f%%', colors=[color_plate[i] for i in label_num.astype(int)])
# ax[0].title.set_text("PPOT labels")
#
# ax[1].pie(all_size_3[[1, 3]], labels=["slow", "fast"], autopct='%1.1f%%', colors=[color_plate[i] for i in label_nums.astype(int)])  # [labels[i] for i in label_nums.astype(int)]
# ax[1].title.set_text("Binary PPOT labels")
#
# ax[2].pie(all_size_5[[1, 2, 3]], labels=["slow", "medium", "fast"], autopct='%1.1f%%', colors=[color_plate[i] for i in label_nums_3.astype(int)])  # [labels[i] for i in label_nums_3.astype(int)]
# ax[2].title.set_text("3 PPOT labels")
# fig.suptitle("Distribution of PPOT labels\nonly 5 min")

## check task difficulty
print(y["task_difficulty"].describe())
labels = ["very slow", "slow", "medium", "fast", "very fast"]
_, sizes = np.unique(y["task_difficulty"].values, return_counts=True)
fig, ax = plt.subplots(1, 1)
for i in [1, 3, 5]:
    tmp_x = (y.loc[y["time"] == i]["duration_estimate"].values - i) / y.loc[y["time"] == i]["duration_estimate"].values * 100
    tmp_y = y.loc[y["time"] == i]["ppot"].values
    m_x = []
    for j in range(5):
        m_x.append(np.mean(tmp_x[tmp_y == j]))

    ax.scatter(tmp_y, tmp_x, alpha=0.5)
    ax.plot(range(5), m_x, label=f"Time: {i} min", alpha=0.5)
ax.set_xticks(range(5), labels)
ax.set_xlabel("PPOT")
ax.set_ylabel("Relative time estimation [%]")
ax.hlines(1, 0, 4, colors="r", linestyles="dashed", label="Objective time")
plt.legend()
# ax[0].pie(sizes, labels=labels, autopct='%1.1f%%')
# ax[0].title.set_text("Task difficulty distribution")

# fig, ax = plt.subplots(1, 2)
# for i in [0, 1, 2, 3, 4]:
#     tmp = (y.loc[y["ppot"] == i]["duration_estimate"].values - y.loc[y["ppot"] == i]["time"]) / y.loc[y["ppot"] == i]["time"] * 100
#     ax[0].hist(tmp, label=f"ppot {i}", alpha=0.5)
#
# ax[0].set_xlabel("Relative Time Estimation [%]")
# ax[0].set_ylabel("Occurances")
# ax[0].legend()
# TODO correlate duration estimate with ppot
# TODO correlate task difficulty with ppot

# TODO check distribution with setting?
plt.show()


