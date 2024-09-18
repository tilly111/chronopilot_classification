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


# load labels
labels = pd.read_csv("/Volumes/Data/chronopilot/helicopter/cleaned_data/labels.csv", sep=";")
labels.drop(columns=["Diff1", "Diff2", "Diff3"], inplace=True)

# scale the PPOT column based on the ParticipantID column: for each participant the values are scaled to 1 or 5
labels["PPOT"] = labels.groupby("ParticipantID")["PPOT"].transform(lambda x: (x - x.min()) / (x.max() - x.min()) * 4 + 1)

# rescale the PPOT column: all values that are smaller than 3 become 0, all values that are greater than or equal to 3 become 1
labels["PPOT"] = labels["PPOT"].apply(lambda x: 0 if x < 3 else 1)

labels.to_csv("preprocessed_data/helicopter/y.csv", index=False)

print(labels)
# block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
# study = "2"  # "1" or "2"
#
# null = []
# eins = []
#
# for user in constants.SUBJECTS_STUDY_2:
#     # print(user)
#     all_labels = pd.DataFrame()
#     for block in block_names:
#         labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
#         # print(block)
#         # print(labels["block_estimation"].value_counts())
#         all_labels = pd.concat([all_labels, labels.loc[labels["subject"] == user]], axis=0)
#     tmp = all_labels["block_estimation"].value_counts()
#     try:
#         null.append(tmp[0])
#     except:
#         print(f"{user}\t{0}\t{tmp[1]}")
#     try:
#         eins.append(tmp[1])
#     except:
#         print(f"{user}\t{tmp[0]}\t{0}")
#     try:
#         print(f"{user}\t{tmp[0]}\t{tmp[1]}")
#     except:
#         # print(f"{user} some weird error")
#         pass
#

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

# plt.figure()
# plt.hist(null, bins=5)
# plt.title("null")
#
# plt.figure()
# plt.hist(eins, bins=5)
# plt.title("eins")
# plt.show()