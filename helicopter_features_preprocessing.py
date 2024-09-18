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


all_data = pd.DataFrame(columns=["ParticipantID", "Session"] + constants.ALL_PPG_FEATURES_HEARTPY+constants.ALL_EDA_FEATURES+constants.ALL_TMP_FEATURES)

print(all_data)
# load calculated features
for i in range(1, 13):
    tmp = pd.read_csv(f"/Volumes/Data/chronopilot/helicopter/features/heartpy_helicopter/tmp/start_posttest/subjectID_{i}.csv")
    ppg = pd.read_csv(f"/Volumes/Data/chronopilot/helicopter/features/heartpy_helicopter/ppg/start_posttest/subjectID_{i}.csv")
    eda = pd.read_csv(f"/Volumes/Data/chronopilot/helicopter/features/heartpy_helicopter/eda/start_posttest/subjectID_{i}.csv")
    tmp.drop(columns=["Unnamed: 0"], inplace=True)
    ppg.drop(columns=["Unnamed: 0"], inplace=True)
    eda.drop(columns=["Unnamed: 0"], inplace=True)

    tmp = pd.concat([ppg, eda, tmp], axis=1)
    tmp["ParticipantID"] = i
    tmp["Session"] = [1,2,3,4]  # Assuming all sessions are 1 for this example

    # all_data.loc["ParticipantID"] = i
    all_data = pd.concat([all_data, tmp], ignore_index=True)

    # print(tmp)  # Print the features for each file loaded
    # Further processing can be done here if needed

# normalize all columns except "ParticipantID" and "Session" individually
for col in all_data.columns:
    if col not in ["ParticipantID", "Session"]:
        all_data[col] = (all_data[col] - all_data[col].mean()) / all_data[col].std()

print(all_data)
all_data.to_csv("preprocessed_data/helicopter/X.csv", index=False)  # Save the preprocessed data
