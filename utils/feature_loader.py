import pandas as pd
import numpy as np
import constants


def load_scream_data(study=1, baseline_subtraction=True):
    if baseline_subtraction:
        X = pd.read_csv(f"/Volumes/Data/chronopilot/2024_scream/study{study}/preprocessed_data/baseline_subtraction/X_duration_estimate_2_classes.csv")
        y = pd.read_csv(f"/Volumes/Data/chronopilot/2024_scream/study{study}/preprocessed_data/baseline_subtraction/y_duration_estimate_2_classes.csv")
    else:
        X = pd.read_csv(
            f"/Volumes/Data/chronopilot/2024_scream/study{study}/preprocessed_data/no_baseline_subtraction/X_duration_estimate_2_classes.csv")
        y = pd.read_csv(
            f"/Volumes/Data/chronopilot/2024_scream/study{study}/preprocessed_data/no_baseline_subtraction/y_duration_estimate_2_classes.csv")
        
    return X, y

# def load_eye_tracking_data(number_of_classes=2, load_preprocessed=True, label_name=["ppot"],
#                            include_meta_label=False) -> tuple[pd.DataFrame, pd.DataFrame]:
#     # TODO outdated
#     if load_preprocessed:
#         if number_of_classes == 2:
#             X_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_train.csv")
#             y_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_train.csv")
#             # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_test.csv")
#             # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_test.csv")
#             X_train.drop(columns=["participant", "time", "robot"], inplace=True)  # drop setting information
#         elif number_of_classes == 3:
#             X_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_train.csv")
#             y_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_train.csv")
#             # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_test.csv")
#             # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_test.csv")
#         else:
#             print("Number of classes not preprocessed")
#             X_train = None
#             y_train = None
#
#         return X_train, y_train
#     if include_meta_label:
#         # include meta data to labels if we want to do analysis with them
#         label_name = label_name + ["participant", "time", "robot"]
#     data = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/pupil_features_sub.csv")
#     data_2 = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/fixations_features_sub.csv")
#     data_2.drop(columns=["participant", "robot", "time"], inplace=True)
#     data_2.columns = [f"{col}_fix" for col in data_2.columns]
#     data_2 = pd.concat([data, data_2], axis=1)
#
#     labels = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")
#
#     data_2.dropna(inplace=True)
#
#     y_all = None
#     x_remove = []
#     for i in range(data_2.shape[0]):
#         time = data_2["time"].iloc[i]
#         robot = data_2["robot"].iloc[i]
#         participant = data_2["participant"].iloc[i]
#
#         y = \
#             labels.loc[
#                 ((labels["time"] == time) & (labels["robot"] == robot) & (labels["participant"] == participant))][
#                 label_name]
#
#         try:
#             y = y.squeeze()
#             if not np.isnan(y).any():
#                 y = np.ndarray(shape=(1, len(label_name)), buffer=np.array([y]))
#                 if y_all is None:
#                     y_all = y
#                 else:
#                     y_all = np.concatenate((y_all, y), axis=0)
#             else:
#                 x_remove.append(i)
#         except:
#             x_remove.append(i)
#
#     if number_of_classes == 2:
#         if len(label_name) == 1:
#             y_all = np.where(y_all > 2, 1, 0)
#         else:
#             y_all[:, 0] = np.where(y_all[:, 0] > 2, 1, 0)
#     elif number_of_classes == 3:
#         if len(label_name) == 1:
#             y_all[y_all < 2] = 0
#             y_all[y_all == 2] = 1
#             y_all[y_all > 2] = 2
#         else:
#             y_all_c = y_all[:, 0]
#             y_all_c[y_all_c < 2] = 0
#             y_all_c[y_all_c == 2] = 1
#             y_all_c[y_all_c > 2] = 2
#             y_all[:, 0] = y_all_c
#     y_all = pd.DataFrame(y_all, columns=label_name)
#     data_2.drop(data_2.index[x_remove], inplace=True)
#
#     return data_2, y_all


# def load_eye_tracking_data_slice(number_of_classes=2, load_preprocessed=True, label_name=["ppot"],
#                                  include_meta_label=False) -> tuple[pd.DataFrame, pd.DataFrame]:
#     # TODO outdated
#     if load_preprocessed:
#         if number_of_classes == 2:
#             X_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_train_slice.csv")
#             y_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_train_slice.csv")
#             # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_test_slice.csv")
#             # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_test_slice.csv")
#         elif number_of_classes == 3:
#             X_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_train_slice.csv")
#             y_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_train_slice.csv")
#             # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_test_slice.csv")
#             # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_test_slice.csv")
#         else:
#             print("Number of classes not preprocessed")
#             X_train = None
#             y_train = None
#         return X_train, y_train
#     if include_meta_label:
#         # include meta data to labels if we want to do analysis with them
#         label_name = label_name + ["participant", "time", "robot"]
#     data = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/pupil_features_sub_slice.csv")
#     data_2 = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/fixations_features_sub_slice.csv")
#     data_2.drop(columns=["participant", "robot", "time"], inplace=True)
#     data_2.columns = [f"{col}_fix" for col in data_2.columns]
#     data_2 = pd.concat([data, data_2], axis=1)
#
#     labels = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")
#
#     # todo hack -> run feature calculation properly
#     data_2.drop(columns=["sub_number_clusters"], inplace=True)
#     data_2.dropna(inplace=True)
#
#     y_all = None
#     x_remove = []
#     for i in range(data_2.shape[0]):
#         time = data_2["time"].iloc[i]
#         robot = data_2["robot"].iloc[i]
#         participant = data_2["participant"].iloc[i]
#
#         y = \
#             labels.loc[
#                 ((labels["time"] == time) & (labels["robot"] == robot) & (labels["participant"] == participant))][
#                 label_name]
#
#         try:
#             y = y.squeeze()
#             if not np.isnan(y).any():
#                 y = np.ndarray(shape=(1, len(label_name)), buffer=np.array([y]))
#                 if y_all is None:
#                     y_all = y
#                 else:
#                     y_all = np.concatenate((y_all, y), axis=0)
#             else:
#                 x_remove.append(i)
#         except:
#             x_remove.append(i)
#
#     if number_of_classes == 2:
#         if len(label_name) == 1:
#             y_all = np.where(y_all > 2, 1, 0)
#         else:
#             y_all[:, 3] = np.where(y_all[:, 3] > 2, 1, 0)
#     elif number_of_classes == 3 and len(label_name) == 1:
#         y_all[y_all < 2] = 0
#         y_all[y_all == 2] = 1
#         y_all[y_all > 2] = 2
#     y_all = pd.DataFrame(y_all, columns=label_name)
#     data_2.drop(data_2.index[x_remove], inplace=True)
#
#     return data_2, y_all

def load_eye_tracking_data_tw(number_of_classes=2, load_preprocessed=True, tw=10, label_name=["ppot"],
                              include_meta_label=False, load_test=False, bls=False, dir="", only_pupil=False) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    if load_preprocessed:
        return _load_eye_tracking_data_tw_preprocessed(number_of_classes, tw, label_name, include_meta_label, load_test,
                                                       bls, dir, only_pupil)
    
    if include_meta_label:
        # include meta-data to labels if we want to do analysis with them
        label_name = label_name + ["participant", "time", "robot"]
    if bls:
        tag = "_bls"
    else:
        tag = ""
    if only_pupil:
        tag_pupil = "_new"
    else:
        tag_pupil = ""
    df_pupil = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_pupil_features_tw_{tw}{tag}{tag_pupil}.csv")
    
    if only_pupil:
        all_features = df_pupil
    else:
        df_fixation = pd.read_csv(
            f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_fixations_features_tw_{tw}{tag}.csv")
        
        df_fixation.columns = [f"{col}_fix" for col in df_fixation.columns]
        df_fixation = df_fixation.rename(
            columns={'time_fix': 'time', 'robot_fix': 'robot', 'participant_fix': 'participant', 'slice_fix': 'slice'})
        
        all_features = pd.merge(df_pupil, df_fixation, on=['time', 'robot', 'participant', 'slice'], how='inner')
    
    labels = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")
    
    all_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    y_all = None
    x_remove = []
    for i in range(all_features.shape[0]):
        time = all_features["time"].iloc[i]
        robot = all_features["robot"].iloc[i]
        participant = all_features["participant"].iloc[i]
        
        y = \
        labels.loc[((labels["time"] == time) & (labels["robot"] == robot) & (labels["participant"] == participant))][
            label_name]
        if label_name[0] == "ppot":
            pass  # no further processing required
        elif label_name[0] == "duration_estimate":
            y = y / time
        elif label_name[0] == "arousal":
            # NOTE the distribution is between 0 and 8
            pass  # no further processing required
        elif label_name[0] == "valence":
            # NOTE the distribution is between 0 and 8
            pass  # no further processing required
        else:
            print("Label name not found")
            y = None
        
        try:
            y = y.squeeze()
            # print(x_remove)
            if not np.isnan(y).any():
                # print(len(label_name), np.array([y, participant, time, robot]))
                y = np.ndarray(shape=(1, 1), buffer=np.array([y]))
                if number_of_classes == 2:
                    if label_name[0] == "ppot":
                        y[0, 0] = 1 if y[0, 0] > 2 else 0
                    elif label_name[0] == "duration_estimate":
                        y[0, 0] = 1 if y[
                                           0, 0] >= 0.9 else 0  # TODO adjust for slight underestimation in most settings -> 0.9 almost balanced classes
                    elif label_name[0] == "arousal":
                        y[0, 0] = 1 if y[0, 0] > 4 else 0
                elif number_of_classes == 3:
                    if label_name[0] == "ppot":
                        if y[0, 0] < 2:
                            y[0, 0] = 0
                        elif y[0, 0] == 2:
                            y[0, 0] = 1
                        else:
                            y[0, 0] = 2
                    elif label_name[0] == "duration_estimate":
                        if y[0, 0] < 0.75:
                            y[0, 0] = 0
                        elif y[0, 0] >= 0.75 and y[0, 0] <= 1.05:
                            y[0, 0] = 1
                        else:
                            y[0, 0] = 2
                    elif label_name[0] == "arousal":
                        #  0 if x <= 3 else 1 if x <= 7 else 2
                        if y[0, 0] <= 2:
                            y[0, 0] = 0
                        elif y[0, 0] <= 5:
                            y[0, 0] = 1
                        else:
                            y[0, 0] = 2
                y = np.ndarray(shape=(1, len(label_name)), buffer=np.array([y[0, 0], participant, time, robot]))
                # print(y)
                y_all = np.concatenate((y_all, y), axis=0) if y_all is not None else y
            else:
                x_remove.append(i)
        except:
            x_remove.append(i)
    
    y_all = pd.DataFrame(y_all, columns=label_name)
    all_features.drop(all_features.index[x_remove], inplace=True)
    
    return all_features, y_all


def _load_eye_tracking_data_tw_preprocessed(number_of_classes, tw, label_name, include_meta_label, load_test, bls=False,
                                            dir="", only_pupil=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bls:
        tag = "_bls"
    else:
        tag = ""
    if only_pupil:
        tag_pupil = "_only_pupil"
    else:
        tag_pupil = ""
    if load_test:
        print("Loading test data")
        if number_of_classes == 2:
            X_test = pd.read_csv(
                f"preprocessed_data/eye_tracking_2_classes/X_test_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
            y_test = pd.read_csv(
                f"preprocessed_data/eye_tracking_2_classes/y_test_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
        elif number_of_classes == 3:
            X_test = pd.read_csv(
                f"preprocessed_data/eye_tracking_3_classes/X_test_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
            y_test = pd.read_csv(
                f"preprocessed_data/eye_tracking_3_classes/y_test_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
        else:
            print("Number of classes not preprocessed")
            X_test = None
            y_test = None
        return X_test, y_test
    elif include_meta_label:
        print("Loading data with meta data")
        if number_of_classes == 2:
            print(
                f"{dir}preprocessed_data/eye_tracking_2_classes/X_tw_{tw}_label_{label_name[0]}_withMetaData{tag}{tag_pupil}.csv")
            X = pd.read_csv(
                f"{dir}preprocessed_data/eye_tracking_2_classes/X_tw_{tw}_label_{label_name[0]}_withMetaData{tag}{tag_pupil}.csv")
            y = pd.read_csv(
                f"{dir}preprocessed_data/eye_tracking_2_classes/y_tw_{tw}_label_{label_name[0]}_withMetaData{tag}{tag_pupil}.csv")
        elif number_of_classes == 3:
            X = pd.read_csv(
                f"{dir}preprocessed_data/eye_tracking_3_classes/X_tw_{tw}_label_{label_name[0]}_withMetaData{tag}{tag_pupil}.csv")
            y = pd.read_csv(
                f"{dir}preprocessed_data/eye_tracking_3_classes/y_tw_{tw}_label_{label_name[0]}_withMetaData{tag}{tag_pupil}.csv")
        else:
            print("Number of classes not preprocessed")
            X = None
            y = None
        return X, y
    else:
        print("Loading regular data without meta data")
        if number_of_classes == 2:
            X_train = pd.read_csv(
                f"preprocessed_data/eye_tracking_2_classes/X_train_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
            y_train = pd.read_csv(
                f"preprocessed_data/eye_tracking_2_classes/y_train_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
        elif number_of_classes == 3:
            X_train = pd.read_csv(
                f"preprocessed_data/eye_tracking_3_classes/X_train_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
            y_train = pd.read_csv(
                f"preprocessed_data/eye_tracking_3_classes/y_train_tw_{tw}_label_{label_name[0]}{tag}{tag_pupil}.csv")
        else:
            print("Number of classes not preprocessed")
            X_train = None
            y_train = None
        
        return X_train, y_train


def load_eye_tracking_data_baseline() -> pd.DataFrame:
    df_pupil = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_pupil_features_baseline.csv")
    df_fixation = pd.read_csv(
        f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_fixations_features_baseline.csv")
    df_fixation.columns = [f"{col}_fix" for col in df_fixation.columns]
    df_fixation = df_fixation.rename(
        columns={'time_fix': 'time', 'robot_fix': 'robot', 'participant_fix': 'participant', 'slice_fix': 'slice'})
    
    print(df_pupil.shape, df_fixation.shape)
    
    all_features = pd.merge(df_pupil, df_fixation, on=['time', 'robot', 'participant', 'slice'], how='inner')
    print(all_features.shape)
    
    all_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_features.dropna(inplace=True)
    print(all_features.shape)
    
    return all_features
