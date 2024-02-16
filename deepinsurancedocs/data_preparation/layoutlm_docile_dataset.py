import logging
import os

import torch
import json
import glob
from tqdm import tqdm
from torch.utils.data import Dataset

import sys 
sys.path[0] = ''  # nopep8
sys.path.append('/domino/datasets/docile')
sys.path.append('/mnt')

from docile.dataset import CachingConfig, Document
from docile.dataset import Dataset as DocileDataset

from models.layoutlm.prepare_data import convert_sequence_to_tags
from deepinsurancedocs.data_preparation.preprocessing import normalize_bbox
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


class LayoutLMDocileDataset(Dataset):
    def __init__(self,  docile_dir, subset_data_pages_path, tokenizer, label_list, pad_token_label_id, mode, tagging_scheme):
        """
        loaded_docile_dataset : loaded full docile dataset ('unlabeled')
        subset_data_pages_path : path to the json containing the list of pages of the subset of docile you want to build
        tokenizer : LayoutLM tokenizer
        label_list : list of the labels to use
        pad_token_label_ids : id of the padding token
        tagging_scheme : NER tagging scheme ('BIO' / 'BIOES')
        """
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.pad_token_label_id = pad_token_label_id
        self.mode = mode
        self.docile_dir = docile_dir
        self.tagging_scheme = tagging_scheme
        # # Hard coded, to improve
        # self.subset_data_pages_dir = "/domino/datasets/local/DeepInsuranceDocs/docile_1m/list_data.json"
        self.subset_data_pages_path = subset_data_pages_path
        with open(self.subset_data_pages_path, 'r', encoding='utf-8') as file:
            self.docile_list_data = json.load(file)
        
    def __len__(self):
        return len(self.docile_list_data)

    def __getitem__(self, index):
        # If docile, index must match doc+page index
        example_str = self.docile_list_data[index]
        document_index, page_index = self.docile_list_data[0].replace('.json','').split('_')
        
        page_index = int(page_index)
        docile_document = Document(docid=document_index,
                                    dataset_path=self.docile_dir,
                                    load_annotations=False,
                                    load_ocr=False,
                                    cache_images=CachingConfig.OFF,
                                )
        docile_page_ocr_data = docile_document.ocr.get_all_words(page_index, snapped=True)
        page_width, page_height = docile_document.annotation.page_image_size_at_200dpi(page_index)

        example = convert_docile_obj_to_example(docile_page_ocr_data, page_width, page_height, example_str)
        del docile_document, docile_page_ocr_data, page_width, page_height
        # Convert example to features
        feature = convert_example_to_features(
            example,
            self.label_list,
            max_seq_length=512,
            tokenizer=self.tokenizer,
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_label_id=self.pad_token_label_id,
            tagging_scheme=self.tagging_scheme
        )

        # Convert features to tensors
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        label_ids = torch.tensor(feature.label_ids, dtype=torch.long)
        bboxes = torch.tensor(feature.boxes, dtype=torch.long)

        # Mask indicating the position of the first token of each subtokenized word
        first_token_mask = (label_ids != -100).to(torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "labels": label_ids,
            "bbox": bboxes,
            "first_token_mask": first_token_mask,
        }


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size

def convert_docile_obj_to_example(docile_page_ocr_data, page_width, page_height, page_filename):
    """
    Convert docile page obj to InputExample format
    """
    page_boxes = []
    page_words = []
    page_labels = []
    page_normalized_boxes = []
    page_size = [page_width, page_height]
    # Convert docile example into an InputExample format
    for word in docile_page_ocr_data:
        scaled_bbox = word.bbox.to_absolute_coords(page_width, page_height).to_tuple()

        page_boxes.append(scaled_bbox)  # y1, x1, y2, x2 → x1, y1, x2, y2
        page_normalized_boxes.append(normalize_bbox(scaled_bbox, page_width, page_height))
        assert max(normalize_bbox(scaled_bbox, page_width, page_height)) <= 1000
        page_words.append(word.text)
        page_labels.append(None)

    example = InputExample(guid="",  # "%s-%d" % (mode, guid_index),
                        words=page_words,
                        labels=page_labels,
                        boxes=page_normalized_boxes,
                        actual_bboxes=page_boxes,
                        file_name=page_filename,
                        page_size=page_size)
    return example

def convert_example_to_features(
    example,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    tagging_scheme=None
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    file_name = example.file_name
    page_size = example.page_size
    width, height = page_size

    tokens = []
    token_boxes = []
    actual_bboxes = []
    label_ids = []
    
    try:
        example.labels = convert_sequence_to_tags(example.labels, tagging_scheme)
    except:
        # print('Labels or tagging scheme not provided')
        for i in range(len(example.labels)):
            example.labels[i]='O'
        example.labels = convert_sequence_to_tags(example.labels, tagging_scheme)


    for word, label, box, actual_bbox in zip(
        example.words, example.labels, example.boxes, example.actual_bboxes
    ):
        word_tokens = tokenizer.tokenize(word)
        if word == '':
            continue
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        if len([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))!= len(word_tokens):
            saved_word = word
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
        )

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        token_boxes += [cls_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        actual_bboxes = [[0, 0, width, height]] + actual_bboxes
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
        token_boxes = ([pad_token_box] * padding_length) + token_boxes
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        token_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids,
                         boxes=token_boxes,
                         actual_bboxes=actual_bboxes,
                         file_name=file_name,
                         page_size=page_size)

def convert_examples_to_features_multi_threaded(examples, label_list, max_seq_length, tokenizer, cls_token_at_end=False,
                                            cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]",
                                            sep_token_extra=False, pad_on_left=False, pad_token=0,
                                            cls_token_box=[0, 0, 0, 0], sep_token_box=[1000, 1000, 1000, 1000],
                                            pad_token_box=[0, 0, 0, 0], pad_token_segment_id=0,
                                            pad_token_label_id=-1, sequence_a_segment_id=0,
                                            mask_padding_with_zero=True, tagging_scheme=None, num_threads=4):

    label_map = {label: i for i, label in enumerate(label_list)}

    def process_example(example):
        return convert_example_to_features(example, label_list, max_seq_length, tokenizer, cls_token_at_end,
                                        cls_token, cls_token_segment_id, sep_token, sep_token_extra, pad_on_left,
                                        pad_token, cls_token_box, sep_token_box, pad_token_box,
                                        pad_token_segment_id, pad_token_label_id, sequence_a_segment_id,
                                        mask_padding_with_zero, tagging_scheme)

    features = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in tqdm(executor.map(process_example, examples), total=len(examples), desc='Converting examples to features'):
            features.append(result)

    return features
            
