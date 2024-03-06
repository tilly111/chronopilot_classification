import platform
import pandas as pd
import constants

import matplotlib
import matplotlib.pyplot as plt

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')



block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
study = "2"  # "1" or "2"

null = []
eins = []

for user in constants.SUBJECTS_STUDY_2:
    # print(user)
    all_labels = pd.DataFrame()
    for block in block_names:
        labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
        # print(block)
        # print(labels["block_estimation"].value_counts())
        all_labels = pd.concat([all_labels, labels.loc[labels["subject"] == user]], axis=0)
    tmp = all_labels["block_estimation"].value_counts()
    try:
        null.append(tmp[0])
    except:
        print(f"{user}\t{0}\t{tmp[1]}")
    try:
        eins.append(tmp[1])
    except:
        print(f"{user}\t{tmp[0]}\t{0}")
    try:
        print(f"{user}\t{tmp[0]}\t{tmp[1]}")
    except:
        # print(f"{user} some weird error")
        pass


# for block in block_names:
#     labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
#
#     tmp = labels["block_estimation"].value_counts()
#     try:
#         null.append(tmp[0])
#     except:
#         print(f"{block}\t{0}\t{tmp[1]}")
#     try:
#         eins.append(tmp[1])
#     except:
#         print(f"{block}\t{tmp[0]}\t{0}")
#     try:
#         print(f"{block}\t{tmp[0]}\t{tmp[1]}")
#     except:
#         # print(f"{user} some weird error")
#         pass

plt.figure()
plt.hist(null, bins=5)
plt.title("null")

plt.figure()
plt.hist(eins, bins=5)
plt.title("eins")
plt.show()