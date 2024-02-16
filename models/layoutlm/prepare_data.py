""" 
Prepare data for LayoutLM model. 
Input: the path to the data directory.
Output: A Dataset object containing the data.
"""
import json
import os
import sys
from typing import Dict
from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value, Array2D, Array3D

sys.path[0] = r''  # nopep8

from transformers import AutoProcessor
import json
from typing import Dict
from transformers import AutoProcessor
from deepinsurancedocs.data_preparation.data_utils import create_dataset_dict, label_dict_transform
from deepinsurancedocs.data_preparation.preprocessing import convert_sequence_to_tags


class CustomDataset(Dataset):
    def __getitem__(self, idx):
        subdict = super().__getitem__(idx)
        return list(subdict.values())


class LayoutLMDataPreparation:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as config_file:
            self.config = json.load(config_file)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlm-base-uncased", apply_ocr=False)
        self.features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        self.idx2label = label_dict_transform(
            label_dict=self.config['label_list'], scheme=self.config['preprocessing']['tagging_scheme'])
        self.label2idx = {label: idx for idx, label in self.idx2label.items()}

    def load_dataset_with_config(self) -> DatasetDict:
        """ 
        Loads data according to the specified config file in a HuggingFace Dataset.
        After loading the data, a preprocessing function is applied to the data to prepare it for the model.

        Returns
        -------
        dataset : Dataset
            A HuggingFace Dataset object containing the data.
        """

        data_dir: str = self.config['data_dir']
        input_format: str = self.config['input_format']

        if input_format not in ["text_and_boxes", "text_and_boxes_and_images", "text_only"]:
            raise ValueError(
                "Invalid input format. Allowed values are 'text_and_boxes', 'text_and_boxes_and_images', and 'text_only'.")

        return self.create_dataset_dict(data_dir, input_format)

    def preprocess_dataset(self, data_dict: Dict[str, Dataset]) -> DatasetDict:
        """
        Preprocesses the data in a HuggingFace Dataset object to prepare it for the model.
        From a dict with ['id', 'words', 'bboxes', 'labels']
        to a DatasetDict with ['input_ids', 'attention_mask', 'bbox', 'labels_idx'] 
        Parameters
        ----------
        data_dict : Dict[str, Dataset]
            A dictionary containing the train and test datasets.

        Returns
        -------
        dataset : DatasetDict
            A HuggingFace DatasetDict object containing the preprocessed data.
        """

        # Preprocess data
        preprocessing_config: dict = self.config.get('preprocessing', {})

        tagging_scheme = preprocessing_config.get('tagging_scheme')

        for mode in ['train', 'test']:
            ner_tags = [self.convert_sequence_to_tags(
                sublist, tagging_scheme) for sublist in data_dict[mode]['labels']]
            ner_tags_idx = [[self.label2idx[tag]
                             for tag in sublist] for sublist in ner_tags]
            data_dict[mode]['labels_idx'] = ner_tags_idx

        dataset_dict = DatasetDict({'train': Dataset.from_dict(data_dict['train']),
                                   'test': Dataset.from_dict(data_dict['test'])})

        # define the column names
        column_names = dataset_dict["train"].column_names
        text_column_name = "words"
        boxes_column_name = "bboxes"
        label_column_name = "labels_idx"

        # define a function to prepare the examples
        def prepare_examples(examples):
            words = examples[text_column_name]
            boxes = examples[boxes_column_name]
            word_labels = examples[label_column_name]
            # encode the examples
            encoding = self.processor(
                                    images=[np.zeros((1,1,1)) for image_path in examples['image_path']],
                                    text=words,
                                    boxes=boxes,
                                    word_labels=word_labels,
                                    add_special_tokens=True,
                                    padding="max_length",
                                    truncation=True,
                                    verbose=True,
                                    return_token_type_ids=True)

            encoding = processor_2(
                                    text=words,
                                    boxes=boxes,
                                    word_labels=word_labels,
                                    add_special_tokens=True,
                                    padding="max_length",
                                    truncation=True,
                                    verbose=True,
                                    return_token_type_ids=True)
            del encoding["image"]
            return encoding

        # prepare the train dataset
        train_dataset = dataset_dict["train"].map(
            prepare_examples,
            batched=True,
            remove_columns=column_names,
            features=self.features,
        )

        # prepare the test dataset
        test_dataset = dataset_dict["test"].map(
            prepare_examples,
            batched=True,
            remove_columns=column_names,
            features=self.features,
        )

        # return the preprocessed train and test datasets
        return {"train": train_dataset, "test": test_dataset}

    def load_training_ready_dataset(self):
        """
        Loads the dataset with the given configuration and preprocesses it for training.

        Returns:
            A preprocessed dataset ready for training, with ['input_ids', 'attention_mask', 'bbox', 'labels_idx'].
        """
        dataset_dict = self.load_dataset_with_config()
        dataset_dict = self.preprocess_dataset(dataset_dict)
        return dataset_dict

    def load_exploration_dataset(self):
        """
        Loads the dataset using the configuration specified in the object FOR EXPLORATION PURPOSES ONLY.


        Returns:
            dataset_dict (dict): A dictionary containing the dataset with ['id', 'words', 'bboxes', 'labels']
        """
        dataset_dict = self.load_dataset_with_config()
        return dataset_dict

    @staticmethod
    def create_dataset_dict(data_dir: str, input_format: str) -> Dict[str, Dataset]:
        return create_dataset_dict(data_dir, input_format)

    @staticmethod
    def convert_sequence_to_tags(sequence, tagging_scheme):
        return convert_sequence_to_tags(sequence, tagging_scheme)

