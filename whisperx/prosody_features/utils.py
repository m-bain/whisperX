import string


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

    # Determine the total time range from the min start and max end
    min_time = int(min(entry["start"] for entry in char_data))
    max_time = int(max(entry["end"] for entry in char_data))

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
