import json
import os

from datasets import Dataset


def get_file_paths(directory):
    """
    Returns a list of file paths in a directory.
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
        'ner_tags': data['ner_tags'],
        'image_path': file_path.replace('annotations', 'images').replace('.json', '.png')
    }


def get_data_samples(directory):
    """
    Returns a list of dictionaries, each containing the data from a JSON file in a directory.
    """
    json_files = get_file_paths(os.path.join(directory, 'annotations'))
    return [parse_json(f) for f in json_files]


def create_dataset(root_directory):
    """
    Returns a Hugging Face Dataset object containing the data from JSON files in train and test directories.
    """
    train_samples = get_data_samples(os.path.join(root_directory, 'train'))
    test_samples = get_data_samples(os.path.join(root_directory, 'test'))
    return Dataset.from_dict({'train': train_samples, 'test': test_samples})


def load_data_text_and_boxes(data_dir):
    """
    Loads data with text and boxes in a HuggingFace Dataset.
    """
    # Load data
    dataset = create_dataset(data_dir)

    # Preprocess data
    dataset = dataset.map(
        lambda x: {'words': x['words'], 'bboxes': x['bboxes'], 'ner_tags': x['ner_tags']})
    return dataset


def load_data_text_only(data_dir):
    """
    Loads data with text only in a HuggingFace Dataset.
    """
    # Load data
    dataset = create_dataset(data_dir)
    # Preprocess data
    dataset = dataset.map(
        lambda x: {'words': x['words'], 'ner_tags': x['ner_tags']})
    return dataset


def load_data_text_and_boxes_and_images(data_dir):
    """
    Loads data with text, boxes, and images in a HuggingFace Dataset.
    """
    # Load data
    dataset = create_dataset(data_dir)
    # Preprocess data
    dataset = dataset.map(lambda x: {
                          'image': x['image_path'], 'words': x['words'], 'bboxes': x['bboxes'], 'ner_tags': x['ner_tags']})
    return dataset
