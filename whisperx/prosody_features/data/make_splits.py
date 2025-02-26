import os
import random
import json
import argparse


def create_split_json(data_dir, test_ratio=0.1):
    """
    Splits the dataset into train and test splits with equal speaker representation, and creates a JSON file. 
    JSON is saved to the data directory as splits.json.

    Args:
        data_dir (str): Path to the dataset directory.
        test_ratio (float): Ratio of data for the test split (default is 10%).
    """
    
    # Check if output file already exists
    output_json = os.path.join(data_dir, "splits.json")
    if os.path.exists(output_json):
        print(
            f"Error: The file {output_json} already exists. Aborting to prevent overwriting."
        )
        return

    # Dictionary to hold paths for each speaker
    speaker_paths = {}

    # Traverse the dataset to gather paths by speaker
    for speaker_id in os.listdir(data_dir): # Assuming each speaker has a directory
        speaker_dir = os.path.join(data_dir, speaker_id)
        if os.path.isdir(speaker_dir):
            feature_paths = []
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if os.path.isdir(chapter_dir):
                    for file_name in os.listdir(chapter_dir):
                        if file_name.endswith(".json"):
                            feature_paths.append(os.path.join(chapter_dir, file_name))
            speaker_paths[speaker_id] = feature_paths

    # Create train and test splits
    train_split = []
    test_split = []

    for speaker_id, feature_paths in speaker_paths.items():
        random.shuffle(feature_paths)  # Shuffle to ensure random split
        split_point = int(
            len(feature_paths) * test_ratio
        )  # Calculate test split size
        test_split.extend(
            [
                {"path": path, "speaker": speaker_id}
                for path in feature_paths[:split_point]
            ]
        )
        train_split.extend(
            [
                {"path": path, "speaker": speaker_id}
                for path in feature_paths[split_point:]
            ]
        )

    # Combine splits into a dictionary
    split_data = {"train": train_split, "test": test_split}

    # Save to JSON file
    with open(output_json, "w") as json_file:
        json.dump(split_data, json_file, indent=4)

    print(f"JSON split file saved to {output_json}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Create train and test splits for a dataset."
    )
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument(
        "output_json", type=str, help="Path to save the resulting JSON file."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.25,
        help="Ratio of data for the test split (default: 0.5).",
    )

    args = parser.parse_args()

    create_split_json(args.data_dir, args.output_json, args.test_ratio)
