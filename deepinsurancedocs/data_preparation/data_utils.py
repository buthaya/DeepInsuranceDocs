import glob
import json
import os

from datasets import Dataset, DatasetDict


def get_json_paths(directory):
    """
    Returns a list of all the json files in the directory and its subfolders.
    """
    return glob.glob(os.path.join(directory, '**/*.json'), recursive=True)


def get_file_paths(directory):
    """
    Returns a list of all the json files in the directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def parse_json(file_path):
    """
    Parses a JSON file and returns a dictionary with the file's data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'id': os.path.splitext(os.path.basename(file_path))[0],
        'words': data['words'],
        'bboxes': data['bboxes'],
        'labels': data['labels'],
        'image_path': file_path.replace('annotations', 'images').replace('.json', '.png')
    }


def get_data_samples(directory):
    """
    Returns a list of dictionaries, each containing the data from a JSON file in a directory.
    """
    data_samples = {}
    json_files = get_file_paths(os.path.join(directory, 'annotations'))
    for parsed_data in map(parse_json, json_files):
        for key, value in parsed_data.items():
            data_samples.setdefault(key, []).append(value)
    return data_samples


def create_dataset_dict(root_directory, input_format):
    """
    Returns a Hugging Face Dataset object containing the data from JSON files in train and test directories.
    """
    train_samples = get_data_samples(os.path.join(root_directory, 'train'))
    test_samples = get_data_samples(os.path.join(root_directory, 'test'))

    train_samples = {key: train_samples[key] for key in [
        'id', 'words', 'bboxes', 'labels', 'image_path']}
    test_samples = {key: test_samples[key] for key in [
        'id', 'words', 'bboxes', 'labels', 'image_path']}

    return {'train': train_samples, 'test': test_samples}


def label_dict_to_bio(label_dict):
    """
    Converts a dictionary of labels to a dictionary of BIO tags.

    Args:
        label_dict (dict): A dictionary of labels.

    Returns:
        dict: A dictionary of BIO tags.
    """
    new_label_dict = {0: "O"}
    i = 1
    for label in label_dict.values():
        if label != 'O':
            new_label_dict[i] = f"B-{label}"  # beginning of entity
            new_label_dict[i+1] = f"I-{label}"  # inside entity
            i += 2
    return new_label_dict


def label_dict_to_bieso(label_dict):
    """
    Converts a dictionary of labels to a dictionary of BIESO tags.

    Args:
        label_dict (dict): A dictionary of labels.

    Returns:
        dict: A dictionary of BIESO tags.
    """
    new_label_dict = {0: "O"}
    i = 1
    for label in label_dict.values():
        if label != 'O':
            new_label_dict[i] = f"B-{label}"  # beginning of entity
            new_label_dict[i+1] = f"I-{label}"  # inside entity
            new_label_dict[i+2] = f"E-{label}"  # end of entity
            new_label_dict[i+3] = f"S-{label}"  # single entity
            i += 4
    return new_label_dict


def label_dict_transform(label_dict, scheme):
    """
    Transforms a list of labels to a dictionary of BIO or BIESO tags.

    Args:
        label_dict (list): A dict of labels.
        scheme (str): The tagging scheme to use. Either 'BIO' or 'BIESO'.

    Returns:
        dict: A dictionary of BIO or BIESO tags.
    """
    if scheme == 'BIO':
        return label_dict_to_bio(label_dict)
    return label_dict_to_bieso(label_dict)
    
