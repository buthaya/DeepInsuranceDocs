""" 
Prepare data for LayoutLM model. 
Input: the path to the data directory.
Output: A Dataset object containing the data.
"""
import json
import os
import sys
from typing import Dict, List, Union

import torch
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value, Array2D, Array3D

sys.path[0] = r''  # nopep8
print(sys.path)
print(os.getcwd())

import json
from typing import Dict
from transformers import AutoTokenizer, AutoProcessor, DataCollatorForLanguageModeling
from deepinsurancedocs.data_preparation.data_utils import create_dataset_dict, label_dict_transform
from deepinsurancedocs.data_preparation.preprocessing import convert_sequence_to_tags
from models.layoutlm.prepare_data import LayoutLMDataPreparation


class CustomDataset(Dataset):
    def __getitem__(self, idx):
        subdict = super().__getitem__(idx)
        return list(subdict.values())


class LayoutLMDataPreparationMVLM(LayoutLMDataPreparation):
    """
    Modified LayoutLMDataPreparation class for the MVLM task. In this case, we only keep 'input_ids', 'attention_mask',
     'token_type_ids', 'bbox' as features. The 'labels' feature is just the input_ids.
    """

    def __init__(self, config_path: str, mask_proba=0.15, mask_label_token_id=-100):
        self.mask_proba = mask_proba
        self.mask_label_token_id = -100
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as config_file:
            self.config = json.load(config_file)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", apply_ocr=False)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
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
                images=[Image.open(image_path).convert("RGB") for image_path in examples['image_path']],
                text=words,
                boxes=boxes,
                word_labels=word_labels,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                verbose=True,
                return_token_type_ids=True)
            del encoding["image"]
            # Add labels which are the tokens' input_ids
            encoding["labels"] = encoding["input_ids"]

            # Apply random masking
            encoding["input_ids"], encoding["labels"] = self.mask_encoding(encoding["input_ids"],
                                                                           encoding["labels"],
                                                                           self.tokenizer,
                                                                           mask_proba=self.mask_proba,
                                                                           mask_label_token_id=self.mask_label_token_id)
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

    @staticmethod
    def mask_encoding(input_ids: Union[List, torch.tensor], labels: Union[List, torch.tensor], tokenizer,
                      mask_proba: float = 0.15, mask_label_token_id: int = -100):
        """
        Randomly replace tokens from the input_ids by self.tokenizer.mask_token_id in the portion before the padding
        self.tokenizer.pad_token_id
        """
        # Ignore padding tokens, classification tokens and separation tokens
        masked_input_ids = []
        masked_labels = []

        for i in range(len(input_ids)):
            masked_input = torch.tensor(input_ids[i])
            masked_label = torch.tensor(labels[i])

            rand = torch.rand(len(masked_input))
            not_special_tokens_mask = torch.tensor(
                [token_id not in [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id] for token_id
                 in masked_input])
            mask_arr = (rand < mask_proba) & not_special_tokens_mask

            masked_input[mask_arr] = tokenizer.mask_token_id
            masked_label[~mask_arr] = mask_label_token_id

            masked_input_ids.append(masked_input.tolist())
            masked_labels.append(masked_label.tolist())
        return masked_input_ids, masked_labels

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
