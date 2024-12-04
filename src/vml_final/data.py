from pathlib import Path
from typing import List, Optional, Callable, Sequence

import jax
import jax_dataclasses as jdc
import numpy as np
import pandas as pd
from einops import rearrange
from jax import numpy as jnp
from typing_extensions import Self


def sliding_windowify(x, stack_size):
    slid = np.lib.stride_tricks.sliding_window_view(
        x,
        stack_size,
        -2,
    )
    # Swapaxes so that the final axis remains the feature dimension (... time feature)
    return slid.swapaxes(-1, -2)


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


def filter_common(x, y):
    y_vals, y_counts = np.unique_counts(y)
    uncommon_vals = y_vals[y_counts < 128]

    idxs_of_uncommon = np.isin(y, uncommon_vals)
    return x[idxs_of_uncommon], y[idxs_of_uncommon]


def _extract_from_set(data_dir: Path, sensors_to_use: List[str]):

    trial_condition_paths = list(data_dir.rglob("*conditions_*"))
    trial_labels = [
        condition_path.name.split("_")[2] for condition_path in trial_condition_paths
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

    return x_list, y_list


@jdc.pytree_dataclass
class CSVDataset:
    x: jax.Array = jdc.field(default_factory=lambda: jnp.empty(0, dtype=jnp.float32))
    y: jax.Array = jdc.field(default_factory=lambda: jnp.empty(0, dtype=jnp.float32))
    train_idxs: jax.Array = jdc.field(
        default_factory=lambda: jnp.empty(0, dtype=jnp.int32)
    )
    validation_idxs: jax.Array = jdc.field(
        default_factory=lambda: jnp.empty(0, dtype=jnp.int32)
    )
    stack_size: int = 200

    @classmethod
    def build(
        cls,
        key: jax.Array,
        data_dirs: List[Path],
        sensors_to_use: Optional[List[str]] = None,
        stack_size: int = 200,
        train_frac: float = 0.9,
    ) -> Self:

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

        x_list = []
        y_list = []

        for data_dir in data_dirs:
            dir_x_list, dir_y_list = _extract_from_set(data_dir, sensors_to_use)
            x_list.extend(dir_x_list), y_list.extend(dir_y_list)

        # Ensure consistency across trials by keeping only columns that are common across all trials
        common_headers = {col for x_trial in x_list for col in x_trial.columns}
        for x_trial in x_list:
            common_headers.intersection_update(x_trial.columns)

        # Filter X to keep only common columns across all trials
        x_list = [x_trial[list(common_headers)] for x_trial in x_list]

        # Filter out trials with less than 100 timesteps
        x_list = [x_trial for x_trial in x_list if len(x_trial) >= 100]
        y_list = [y_trial for y_trial in y_list if len(y_trial) >= 100]

        # Sliding windowify the data
        # x_list = [sliding_windowify(x_trial, stack_size) for x_trial in x_list]
        # y_list = [y_trial[stack_size - 1 :] for y_trial in y_list]

        # # Filter out indices that have overly repetitive speeds to avoid bias
        # for i in range(len(x_list)):
        #     x_list[i], y_list[i] = filter_common(x_list[i], y_list[i])

        # Cat together the lists to make training data
        x_arr = jnp.array(np.concatenate(x_list))
        y_arr = jnp.array(np.concatenate(y_list))

        rng, key = jax.random.split(key)
        valid_idxs = jnp.arange(len(y_arr))

        running_ind = 0
        for y_trial in y_list:
            pad_start = running_ind
            pad_end = pad_start + stack_size
            # Remove pad idxs each time
            valid_idxs = np.concatenate([valid_idxs[:pad_start], valid_idxs[pad_end:]])
            # Advance by the size of the trial minus the padding
            running_ind = pad_start + len(y_trial) - stack_size

        shuffled_idxs = jax.random.permutation(rng, valid_idxs)

        first_validation_ind = int(train_frac * len(shuffled_idxs))
        train_idxs = shuffled_idxs[:first_validation_ind]
        validation_idxs = shuffled_idxs[first_validation_ind:]

        return cls(
            x=x_arr,
            y=y_arr,
            train_idxs=train_idxs,
            validation_idxs=validation_idxs,
            stack_size=stack_size,
        )

    def get_batch(self, key: jax.Array, batch_size: int, train: bool = True):
        idxs = self.train_idxs if train else self.validation_idxs

        rng, key = jax.random.split(key)
        selected_inds = jax.random.choice(rng, idxs, (batch_size,), replace=False)

        return self[selected_inds]

    def make_epoch_inds(self, key, batch_size: int, train: bool):
        # Shuffle the train inds
        idxs = self.train_idxs if train else self.validation_idxs

        key, rng = jax.random.split(key)
        shuffled_inds = jax.random.permutation(rng, idxs)

        # Trim off the end to make it a perfect number of batches
        trim_size = (len(idxs) // batch_size) * batch_size
        shuffled_trimmed = shuffled_inds[:trim_size]
        # Reshape into batches
        return rearrange(shuffled_trimmed, "(n b) ... -> n b ...", b=batch_size)

    def __getitem__(self, idxs):
        def slice_x(idx):
            slice_start = idx - self.stack_size + 1
            return jax.lax.dynamic_slice_in_dim(self.x, slice_start, self.stack_size)

        def index_y(idx):
            return jax.lax.dynamic_index_in_dim(self.y, idx)

        x_indexed = jax.vmap(slice_x)(idxs)
        y_indexed = jax.vmap(index_y)(idxs)[..., 0]

        return x_indexed, y_indexed


# class CSVDatasetEpochIterator:
#     def __init__(self, loader: "CSVDatasetEpochLoader"):
#         self.loader = loader
#         self.counter = 0
#         self.epoch_inds = self.loader.csv_dataset.make_epoch_inds(
#             self.loader.batch_size, train=self.loader.train
#         )

#     def __next__(self):
#         if self.counter >= len(self.epoch_inds):
#             raise StopIteration
#         batch_inds = self.epoch_inds[self.counter]
#         self.counter = self.counter + 1
#         return self.loader[batch_inds]


@jdc.pytree_dataclass
class CSVDatasetEpochLoader:
    csv_dataset: CSVDataset
    batch_size: int
    train: bool = True

    def scan_for_epoch(self, key: jax.Array, f, init_carry):

        rng, key = jax.random.split(key)
        epoch_indices = self.csv_dataset.make_epoch_inds(
            rng, self.batch_size, self.train
        )

        def scanf(carry, batch_indices):
            x, y = self.csv_dataset[batch_indices]
            carry, out = f(carry, (x, y))

            return carry, out

        return jax.lax.scan(scanf, init_carry, epoch_indices)
