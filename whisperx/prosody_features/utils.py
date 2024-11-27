import string
import yaml
import numpy as np

def load_yaml_config(file_path: str) -> dict:
    """Loads config from yaml file
    Args:
        file_path (str): path to config file

    Returns:
        config (dict): config data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def generate_char_frame_sequence(
    char_data, null_token: str = "<NULL>", filter_non_char: bool = True
):
    r"""
    Generate a list of characters for each frame

    Args:
        char_data (list of dict): List of dictionaries with 'char', 'start', and 'end'. Produced by alignmnet function's 'char' output.
        null_token (str): Token to assing for non-speech frames. Defaults to '\<NULL>'.
        filter_non_char (bool): Remove any non-letter symbols. Defaults to. Defaults to True.

    Returns:
        list of str: List of characters for each frame.
    """

    # Standardize to lowercase
    char_data = [
        {**entry, "char": entry["char"].lower()}
        for entry in char_data
        if "start" in entry
    ]

    # Remove non-letters
    if filter_non_char:
        char_data = [
            entry for entry in char_data if entry["char"] in string.ascii_lowercase
        ]

    try:
        # Determine the total time range from the min start and max end
        min_time = int(min(entry["start"] for entry in char_data))
        max_time = int(max(entry["end"] for entry in char_data))
    except ValueError:
        return None

    # Initialize the output list with the <NULL> token
    time_char_sequence = [null_token for _ in range(min_time, max_time)]

    # Assign characters to time intervals
    for entry in char_data:
        char = entry["char"]
        start = int(entry["start"]) - min_time
        end = int(entry["end"]) - min_time
        for t in range(start, end):
            time_char_sequence[t] = char if char.strip() else null_token

    return time_char_sequence

def average_2d_by_labels(data, labels, axis=0):
    """
    Compute the average of 2D data grouped by unique labels along a specified axis.

    Args:
        data (np.ndarray): 2D array of data values to be averaged.
        labels (np.ndarray): Array of labels for grouping (length matches the size of `data` along the specified axis).
        axis (int): Axis along which to group and average (0 for rows, 1 for columns).

    Returns:
        grouped_means (np.ndarray): 2D array with grouped averages along the specified axis.
    """
    if axis == 1:
        data = data.T  # Transpose if grouping by columns
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    grouped_means = np.array([
        data[inverse_indices == i].mean(axis=0) for i in range(len(unique_labels))
    ])
    return grouped_means.T if axis == 1 else grouped_means