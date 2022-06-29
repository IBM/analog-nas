import os
import tarfile
import requests
import pandas as pd
from path import Path

AUDIO_SR = 16000
AUDIO_LENGTH = 16000
LIBROSA_AUDIO_LENGTH = 22050

EPOCHS = 25

categories = {
    "stop": 0,
    "nine": 1,
    "off": 2,
    "four": 3,
    "right": 4,
    "eight": 5,
    "one": 6,
    "bird": 7,
    "dog": 8,
    "no": 9,
    "on": 10,
    "seven": 11,
    "cat": 12,
    "left": 13,
    "three": 14,
    "tree": 15,
    "bed": 16,
    "zero": 17,
    "happy": 18,
    "sheila": 19,
    "five": 20,
    "down": 21,
    "marvin": 22,
    "six": 23,
    "up": 24,
    "wow": 25,
    "house": 26,
    "go": 27,
    "yes": 28,
    "two": 29,
    "_background_noise_": 30,
}


inv_categories = {
    0: "stop",
    1: "nine",
    2: "off",
    3: "four",
    4: "right",
    5: "eight",
    6: "one",
    7: "bird",
    8: "dog",
    9: "no",
    10: "on",
    11: "seven",
    12: "cat",
    13: "left",
    14: "three",
    15: "tree",
    16: "bed",
    17: "zero",
    18: "happy",
    19: "sheila",
    20: "five",
    21: "down",
    22: "marvin",
    23: "six",
    24: "up",
    25: "wow",
    26: "house",
    27: "go",
    28: "yes",
    29: "two",
    30: "_background_noise_",
}


def downloadData(data_path="/input/speech_commands/"):
    """
    Downloads Google Speech Commands dataset (version0.01)
    :param data_path: Path to download dataset
    :return: None
    """

    dataset_path = Path(os.path.abspath(__file__)).parent.parent + data_path

    datasets = ["train", "test"]
    base_url = "http://download.tensorflow.org/data/"
    urls = [
        base_url+"speech_commands_v0.01.tar.gz",
        base_url+"speech_commands_test_set_v0.01.tar.gz",
    ]

    for dataset, url in zip(datasets, urls):
        dataset_directory = dataset_path + dataset

        # Check if we need to extract the dataset
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
            file_name = dataset_path + dataset + ".tar.gz"

            if os.path.isfile(file_name):
                print("{} already downloaded. Skipping download."
                      .format(file_name))
            else:
                print("Downloading '{}' into '{}' file"
                      .format(url, file_name))

                data_request = requests.get(url)
                with open(file_name, "wb") as file:
                    file.write(data_request.content)

            # Extract downloaded file
            print("Extracting {} into {}"
                  .format(file_name, dataset_directory))

            if file_name.endswith("tar.gz"):
                tar = tarfile.open(file_name, "r:gz")
                tar.extractall(path=dataset_directory)
                tar.close()
            else:
                print("Unknown format.")
        else:
            print(f"{dataset} data setup complete.")

    print("Input data setup successful.")


def getDataDict(data_path="/input/speech_commands/"):
    """
    :param data_path: Path to the downloaded dataset
    :return: Dictionary
    """

    data_path = Path(os.path.abspath(__file__)).parent.parent + data_path

    # Get the validation files
    validation_files = open(data_path + "train/validation_list.txt")
    validation_files = validation_files.read().splitlines()
    validation_files = [data_path + "train/" + file_name
                        for file_name in validation_files]

    # Get the dev files
    dev_files = open(data_path + "train/testing_list.txt").read().splitlines()
    dev_files = [data_path + "train/" + file_name for file_name in dev_files]

    # Find train_files as allFiles - {validation_files, dev_files}
    all_files = []
    for root, dirs, files in os.walk(data_path + "train/"):
        all_files += [root + "/" + file_name for file_name in files
                      if file_name.endswith(".wav")]

    train_files = list(set(all_files) - set(validation_files) - set(dev_files))

    # Get the test files
    test_files = list()
    for root, dirs, files in os.walk(data_path + "test/"):
        test_files += [root + "/" + file_name for file_name in files
                       if file_name.endswith(".wav")]

    # Get labels
    validation_file_labels = [getLabel(wav) for wav in validation_files]
    dev_file_labels = [getLabel(wav) for wav in dev_files]
    train_file_labels = [getLabel(wav) for wav in train_files]
    test_file_labels = [getLabel(wav) for wav in test_files]

    # Create dictionaries containing (file, labels)
    trainData = {"files": train_files, "labels": train_file_labels}
    valData = {"files": validation_files, "labels": validation_file_labels}
    devData = {"files": dev_files, "labels": dev_file_labels}
    testData = {"files": test_files, "labels": test_file_labels}

    dataDict = {"train": trainData, "val": valData, "dev": devData,
                "test": testData}

    return dataDict


def getLabel(file_name):
    """
    Extract the label from its file path
    :param file_name: File name
    :return: Class label
    """

    category = file_name.split("/")[-2]
    label = categories.get(category, categories["_background_noise_"])

    return label


def getDataframe(data, include_unknown=False):
    """
    Create a dataframe from a Dictionary and remove _background_noise_
    :param data: Data dictionary
    :param include_unknown: Whether to include unknown sounds or not
    :return: Dataframe
    """

    df = pd.DataFrame(data)
    df["category"] = df.apply(lambda row: inv_categories[row["labels"]],
                              axis=1)

    if not include_unknown:
        df = df.loc[df["category"] != "_background_noise_", :]

    return df
