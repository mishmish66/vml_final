from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from einops import rearrange


def _extract_trial(data_dir: Path, trial_label: str, sensors_to_use: List[str]):

    sensor_data_list = []
    trial_common_headers = None

    # Load each sensor data
    for sensor in sensors_to_use:
        sensor_path = data_dir / f"{sensor}_treadmill_{trial_label}_01.csv"
        if sensor_path.exists():
            df_sensor = pd.read_csv(sensor_path)
            sensor_data_list.append(df_sensor)

            if trial_common_headers is None:
                trial_common_headers = set(df_sensor["Header"])
            else:
                trial_common_headers.intersection_update(df_sensor["Header"])

    conditions_path = data_dir / f"conditions_treadmill_{trial_label}_01.csv"
    if not conditions_path.exists():
        raise ValueError(f"Conditions file {conditions_path} does not exist.")

    df_conditions = pd.read_csv(conditions_path)

    if trial_common_headers is None:
        raise ValueError("There are no headers common to all trials")

    trial_common_headers.intersection_update(df_conditions["Header"])

    if not (trial_common_headers and sensor_data_list):
        raise ValueError("Oh no!")

    filtered_sensor_data = [
        df[df["Header"].isin(trial_common_headers)].sort_values(by="Header")
        for df in sensor_data_list
    ]
    filtered_conditions = df_conditions[
        df_conditions["Header"].isin(trial_common_headers)
    ].sort_values(by="Header")

    # Merge all filtered sensor data for this trial
    merged_sensor_data = pd.concat(filtered_sensor_data, axis=1)

    # Drop duplicated 'Header' columns except for the first one
    merged_sensor_data = merged_sensor_data.loc[
        :, ~merged_sensor_data.columns.duplicated()
    ]

    # Merge sensor and conditions data for this trial using an inner join to avoid NaN
    merged_data_trial = pd.merge(
        merged_sensor_data, filtered_conditions, on="Header", how="inner"
    )

    # Drop any remaining NaN values after merging
    merged_data_trial = merged_data_trial.dropna()

    # Extract features (x) and target (y) for the trial
    x_trial = merged_data_trial.drop(columns=["Speed"])
    y_trial = merged_data_trial["Speed"]

    return x_trial, y_trial


class CSVDataset:
    def __init__(
        self,
        data_dir: Path,
        sensors_to_use: Optional[List[str]] = None,
        stack_size: int = 32,
        train_frac: float = 0.1,
    ):
        all_sensors = [
            "emg",
            "fp",
            "gcLeft",
            "gcRight",
            "gon",
            "id",
            "ik",
            "ik_offset",
            "imu",
            "jp",
            "markers",
        ]
        # Define which sensors you want to use
        if sensors_to_use is None:
            sensors_to_use = all_sensors

        bad_sensor_names = set(sensors_to_use).difference(set(all_sensors))
        if len(bad_sensor_names) != 0:
            raise ValueError(f"{bad_sensor_names} are not in {all_sensors}")

        trial_condition_paths = list(data_dir.rglob("*conditions_*"))
        trial_labels = [
            condition_path.name.split("_")[2]
            for condition_path in trial_condition_paths
        ]

        # To store X and y for each trial
        x_list = []
        y_list = []

        # Load each dataset and process independently
        for trial_label in trial_labels:
            print(f"\nProcessing trial: {trial_label}")
            x_trial, y_trial = _extract_trial(data_dir, trial_label, sensors_to_use)
            x_list.append(x_trial)
            y_list.append(y_trial)

        # Ensure consistency across trials by keeping only columns that are common across all trials
        common_headers = {col for X_trial in x_list for col in x_trial.columns}
        for x_trial in x_list:
            common_headers.intersection_update(x_trial.columns)

        # Filter X to keep only common columns across all trials
        x_list = [x_trial[list(common_headers)] for x_trial in x_list]

        # Filter out trials with less than 100 timesteps
        x_list = [x_trial for x_trial in x_list if len(x_trial) >= 100]
        x_list = [x_trial for y_trial in y_list if len(y_trial) >= 100]

        # # Print the shape of X and y
        # print(
        #     f"Shape of Xs after filtering common headers: {[x_trial.shape for x_trial in x_list]}"
        # )
        # print(f"Shape of ys: {[y_trial.shape for y_trial in y_list]}")

        # # Check for NaN values in x and y
        # nan_count_x = sum(x_trial.isna().sum().sum() for x_trial in x_list)
        # nan_count_y = sum(y_trial.isna().sum() for y_trial in y_list)

        # print(f"Number of NaN values in x: {nan_count_x}")
        # print(f"Number of NaN values in y: {nan_count_y}")

        # # Check for infinite values in x and y
        # inf_count_x = sum(np.isinf(x_trial.values).sum() for x_trial in x_list)
        # inf_count_y = sum(np.isinf(y_trial).sum() for y_trial in y_list)

        # print(f"Number of infinite values in x: {inf_count_x}")
        # print(f"Number of infinite values in y: {inf_count_y}")

        strided_xs = np.concatenate(
            [
                np.lib.stride_tricks.sliding_window_view(x_trial, stack_size, -2)
                for x_trial in x_list
            ]
        )

        strided_xs = rearrange(strided_xs, "... s t -> ... t s")

        self.x = strided_xs
        self.y = np.concatenate([y_trial[stack_size - 1 :] for y_trial in y_list])

        # Get the train and validation splits
        shuffled_inds = np.arange(len(self.y))
        np.random.shuffle(shuffled_inds)

        first_validation_ind = int(train_frac * len(shuffled_inds))
        self.train_inds = shuffled_inds[:first_validation_ind]
        self.validation_inds = shuffled_inds[first_validation_ind:]

    def get_batch(self, batch_size: int, train: bool):
        inds = self.train_inds if train else self.validation_inds

        selected_inds = np.random.choice(batch_size, inds)

        return self.x[selected_inds], self.y[selected_inds]

    def make_epoch_inds(self, batch_size: int, train: bool):
        # Shuffle the train inds
        shuffled = np.copy(self.train_inds)
        np.random.shuffle(shuffled)

        # Trim off the end to make it a perfect number of batches
        shuffled_trimmed = shuffled[: (len(shuffled) // batch_size) * batch_size]
        # Reshape into batches
        return shuffled_trimmed.reshape(-1, batch_size)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CSVDatasetEpochIterator:
    def __init__(self, loader: "CSVDatasetEpochLoader"):
        self.loader = loader
        self.counter = 0
        self.epoch_inds = self.loader.csv_dataset.make_epoch_inds(
            self.loader.batch_size, train=self.loader.train
        )

    def __next__(self):
        if self.counter >= len(self.epoch_inds):
            raise StopIteration
        batch_inds = self.epoch_inds[self.counter]
        self.counter = self.counter + 1
        return self.loader[batch_inds]
    
    def __len__(self):
        return len(self.epoch_inds) - self.counter


class CSVDatasetEpochLoader:
    def __init__(self, csv_dataset: CSVDataset, batch_size: int, train: bool = True):
        self.csv_dataset = csv_dataset
        self.batch_size = batch_size
        self.train = train

    def __iter__(self):
        return CSVDatasetEpochIterator(self)

    def __getitem__(self, index):
        return self.csv_dataset[index]

    def __len__(self):
        return len(self.csv_dataset.train_inds) // self.batch_size
