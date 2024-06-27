import pandas as pd
import numpy as np
import constants


def load_scream_data(study=1, block_names=["exp_PU", "exp_MA"], background_block_name="exp_S", use_neurokit=False):
    if use_neurokit:
        ppg_features_bg = pd.read_csv(
            constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg_nk.csv")
    else:
        ppg_features_bg = pd.read_csv(
            constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg.csv")
    eda_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/eda.csv")
    tmp_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/tmp.csv")
    eda_features_bg.drop(columns=["subject"], inplace=True)
    tmp_features_bg.drop(columns=["subject"], inplace=True)

    x_bg = pd.concat([ppg_features_bg, eda_features_bg, tmp_features_bg], axis=1).reset_index().drop(columns=["index"])
    # x_bg = eda_features_bg.reset_index().drop(columns=["index"])
    x_bg.fillna(0, inplace=True)  # TODO hack to resolve nans

    labels_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{background_block_name}.csv")

    # load all features and labels
    x_all = None
    y_all = None

    # class_counter = 0
    for block in block_names:
        labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
        # labels["block_estimation"] = labels["block_estimation"] - labels_bg["block_estimation"] + 1
        # labels["block_estimation"] = class_counter
        # class_counter += 1
        # if block == "exp_MA":
        #     labels["block_estimation"] = 0
        # elif block == "exp_T":  # exp_TU exp_MA
        #     labels["block_estimation"] = 1
        # elif block == "exp_TU":
        #     labels["block_estimation"] = 2
        # else:
        #     labels["block_estimation"] = 3
        # print(block)

        if use_neurokit:
            ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg_nk.csv")
        else:
            ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg.csv")
        eda_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/eda.csv")
        tmp_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/tmp.csv")
        eda_features.drop(columns=["subject"], inplace=True)
        tmp_features.drop(columns=["subject"], inplace=True)

        x = pd.concat([ppg_features, eda_features, tmp_features], axis=1).reset_index().drop(columns=["index"])
        # x = eda_features.reset_index().drop(columns=["index"])
        x.fillna(0, inplace=True)  # TODO hack to resolve nans

        # add background subtraction
        # x.loc[:, x.columns != 'subject'] = x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']
        x.loc[:, x.columns != 'subject'] = (x.loc[:, x.columns != 'subject'] - x_bg.loc[:,
                                                                               x_bg.columns != 'subject']) / x_bg.loc[:,
                                                                                                             x_bg.columns != 'subject']

        if x_all is None:
            x_all = x
            y_all = labels
        else:
            x_all = pd.concat([x_all, x], axis=0)
            y_all = pd.concat([y_all, labels], axis=0)

    return x_all, y_all


def load_eye_tracking_data(number_of_classes=2, load_preprocessed=True, label_name=["ppot"],
                           include_meta_label=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if load_preprocessed:
        if number_of_classes == 2:
            X_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_train.csv")
            y_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_train.csv")
            # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_test.csv")
            # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_test.csv")
            X_train.drop(columns=["participant", "time", "robot"], inplace=True)  # drop setting information
        elif number_of_classes == 3:
            X_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_train.csv")
            y_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_train.csv")
            # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_test.csv")
            # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_test.csv")
        else:
            print("Number of classes not preprocessed")
            X_train = None
            y_train = None

        return X_train, y_train
    if include_meta_label:
        # include meta data to labels if we want to do analysis with them
        label_name = label_name + ["participant", "time", "robot"]
    data = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/pupil_features_sub.csv")
    data_2 = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/fixations_features_sub.csv")
    data_2.drop(columns=["participant", "robot", "time"], inplace=True)
    data_2.columns = [f"{col}_fix" for col in data_2.columns]
    data_2 = pd.concat([data, data_2], axis=1)

    labels = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")

    data_2.dropna(inplace=True)

    y_all = None
    x_remove = []
    for i in range(data_2.shape[0]):
        time = data_2["time"].iloc[i]
        robot = data_2["robot"].iloc[i]
        participant = data_2["participant"].iloc[i]

        y = \
            labels.loc[
                ((labels["time"] == time) & (labels["robot"] == robot) & (labels["participant"] == participant))][
                label_name]

        try:
            y = y.squeeze()
            if not np.isnan(y).any():
                y = np.ndarray(shape=(1, len(label_name)), buffer=np.array([y]))
                if y_all is None:
                    y_all = y
                else:
                    y_all = np.concatenate((y_all, y), axis=0)
            else:
                x_remove.append(i)
        except:
            x_remove.append(i)

    if number_of_classes == 2:
        if len(label_name) == 1:
            y_all = np.where(y_all > 2, 1, 0)
        else:
            y_all[:, 0] = np.where(y_all[:, 0] > 2, 1, 0)
    elif number_of_classes == 3:
        if len(label_name) == 1:
            y_all[y_all < 2] = 0
            y_all[y_all == 2] = 1
            y_all[y_all > 2] = 2
        else:
            y_all_c = y_all[:, 0]
            y_all_c[y_all_c < 2] = 0
            y_all_c[y_all_c == 2] = 1
            y_all_c[y_all_c > 2] = 2
            y_all[:, 0] = y_all_c
    y_all = pd.DataFrame(y_all, columns=label_name)
    data_2.drop(data_2.index[x_remove], inplace=True)

    return data_2, y_all


def load_eye_tracking_data_slice(number_of_classes=2, load_preprocessed=True, label_name=["ppot"],
                                 include_meta_label=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if load_preprocessed:
        if number_of_classes == 2:
            X_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_train_slice.csv")
            y_train = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_train_slice.csv")
            # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/X_test_slice.csv")
            # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_2_classes/y_test_slice.csv")
        elif number_of_classes == 3:
            X_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_train_slice.csv")
            y_train = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_train_slice.csv")
            # X_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/X_test_slice.csv")
            # y_test = pd.read_csv(f"preprocessed_data/eye_tracking_3_classes/y_test_slice.csv")
        else:
            print("Number of classes not preprocessed")
            X_train = None
            y_train = None
        return X_train, y_train
    if include_meta_label:
        # include meta data to labels if we want to do analysis with them
        label_name = label_name + ["participant", "time", "robot"]
    data = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/pupil_features_sub_slice.csv")
    data_2 = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/fixations_features_sub_slice.csv")
    data_2.drop(columns=["participant", "robot", "time"], inplace=True)
    data_2.columns = [f"{col}_fix" for col in data_2.columns]
    data_2 = pd.concat([data, data_2], axis=1)

    labels = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")

    # todo hack -> run feature calculation properly
    data_2.drop(columns=["sub_number_clusters"], inplace=True)
    data_2.dropna(inplace=True)

    y_all = None
    x_remove = []
    for i in range(data_2.shape[0]):
        time = data_2["time"].iloc[i]
        robot = data_2["robot"].iloc[i]
        participant = data_2["participant"].iloc[i]

        y = \
            labels.loc[
                ((labels["time"] == time) & (labels["robot"] == robot) & (labels["participant"] == participant))][
                label_name]

        try:
            y = y.squeeze()
            if not np.isnan(y).any():
                y = np.ndarray(shape=(1, len(label_name)), buffer=np.array([y]))
                if y_all is None:
                    y_all = y
                else:
                    y_all = np.concatenate((y_all, y), axis=0)
            else:
                x_remove.append(i)
        except:
            x_remove.append(i)

    if number_of_classes == 2:
        if len(label_name) == 1:
            y_all = np.where(y_all > 2, 1, 0)
        else:
            y_all[:, 3] = np.where(y_all[:, 3] > 2, 1, 0)
    elif number_of_classes == 3 and len(label_name) == 1:
        y_all[y_all < 2] = 0
        y_all[y_all == 2] = 1
        y_all[y_all > 2] = 2
    y_all = pd.DataFrame(y_all, columns=label_name)
    data_2.drop(data_2.index[x_remove], inplace=True)

    return data_2, y_all
