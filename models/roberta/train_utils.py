from typing import List

import numpy as np
import torch
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchcrf import CRF
import torch
from tqdm import tqdm
from transformers import AdamW, RobertaForTokenClassification

from deepinsurancedocs.utils.metrics import doc_exact_match
from models.roberta.model import RobertaForTokenClassificationInternal

pad_token_label_id = CrossEntropyLoss().ignore_index

def train_epoch(model: RobertaForTokenClassificationInternal, train_dataloader: DataLoader, LABEL_LIST: List,
                optimizer: AdamW, device: torch.device, global_step, writers, csv_data):
    """ training for each epoch

    Parameters
    ----------
    model : TokenClassifier
        The model used for training.
    train_dataloader : DataLoader
        The train dataloader.
    optimizer : AdamW
        The optimizer, Adamw
    device : torch.device
        The device.
    """

    model.train()

    for batch in tqdm(train_dataloader, desc='Batch training'):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        first_token_mask = batch["first_token_mask"]

        out_label_list: List[list] = [[] for _ in input_ids]
        preds_list: List[list] = [[] for _ in input_ids]

        # ------------------------------------ Forward pass ------------------------------------ #
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                        first_token_mask=first_token_mask)

        loss = outputs.loss

        # if global_step % 100 == 0:
        #     print(f"Loss after {global_step} steps: {loss.item()}")

        # ------------------------- Backward pass to get the gradients ------------------------- #
        # ----- Here the loss is only computed for the tokens marked 1 in first_token_mask ----- #

        loss.backward()

        # ------------------------------------- Logs update ------------------------------------ #
        label_map = dict(enumerate(LABEL_LIST))
        pad_token_label_id = CrossEntropyLoss().ignore_index
        writer_train = writers['train']

        logits = outputs.logits.detach().cpu().numpy()
        out_label_ids = labels.detach().cpu().numpy()
        out_label_ids = out_label_ids[first_token_mask]

        preds = np.argmax(logits, axis=1)

        start_idx = 0
        for i in range(len(input_ids)):
            len_sequence = len(labels.detach().cpu().numpy()[i][first_token_mask[i]])
            for j, label_idx in enumerate(out_label_ids[start_idx: start_idx+len_sequence]):
                if label_map[label_idx]!='O' or label_map[preds[start_idx+j]]!='O':
                    out_label_list[i].append(label_map[label_idx])
                    preds_list[i].append(label_map[preds[start_idx+j]])
            start_idx = start_idx + len_sequence+1


        f1 = f1_score(out_label_list, preds_list)
        precision = precision_score(out_label_list, preds_list)
        recall = recall_score(out_label_list, preds_list)

        writer_train.add_scalar("Loss/check", loss, global_step=global_step)
        writer_train.add_scalar("total/f1-score/train",
                                f1, global_step=global_step)
        writer_train.add_scalar("total/precision/train",
                                precision, global_step=global_step)
        writer_train.add_scalar("total/recall/train",
                                recall, global_step=global_step)
        csv_data['train'].append((global_step, float(loss)))
        csv_data['f1-score'].append((global_step, float(f1)))
        csv_data['precision'].append((global_step, float(precision)))
        csv_data['recall'].append((global_step, float(recall)))

        # ---------------------------------- Optimizer update ---------------------------------- #
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    return loss, global_step

def eval(model: RobertaForTokenClassificationInternal, eval_dataloader: DataLoader,
         device: torch.device, pad_token_label_id: int, label_map: dict, logging, print_results=True):
    """ evaluation for each epoch

    Parameters
    ----------
    model : RobertaForTokenClassification
        The model used for evaluation.
    eval_dataloader : DataLoader
        The evaluation dataloader.
    device : torch.device
        The device.
    pad_token_label_id : int
        The id of pad token.
    label_map : dict
        The dictionary of label list with label id.

    Returns
    -------
    float
        The average evaluation loss
    """
    eval_loss = 0.0
    nb_eval_steps = 0
    preds: np.ndarray = np.array([])
    out_label_ids: np.ndarray = np.array([])

    iterator = eval_dataloader
    if print_results:
        iterator = tqdm(eval_dataloader, desc='Evaluating')
    preds_list: List = []
    out_label_list: List = []
    start_idx = 0
    for batch in iterator:
        with torch.no_grad():

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            first_token_mask = batch["first_token_mask"]

            # ------------------------------------ Forward pass ------------------------------------ #
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels, first_token_mask=first_token_mask)

            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits
            sequence_labels = labels[first_token_mask]
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if len(preds) == 0:
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                out_label_ids = sequence_labels.detach().cpu().numpy()
            else:
                preds_logits = logits.detach().cpu().numpy()
                preds_argmax=np.argmax(preds_logits, axis=1)
                preds = np.append(preds, preds_argmax, axis=0)
                out_label_ids = np.append(
                    out_label_ids, sequence_labels.detach().cpu().numpy(), axis=0
                )

            for i in range(len(input_ids)):
                non_o_labels = []
                non_o_preds = []
                len_sequence = len(labels.detach().cpu().numpy()[i][first_token_mask[i]])
                for j, label_idx in enumerate(out_label_ids[start_idx: start_idx+len_sequence]):
                    if label_map[label_idx]!='O' or label_map[preds[start_idx+j]]!='O':
                        non_o_labels.append(label_map[label_idx])
                        non_o_preds.append(label_map[preds[start_idx+j]])
                start_idx = start_idx + len_sequence +1
                out_label_list.append(non_o_labels)
                preds_list.append(non_o_preds)
    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps

    doc_ex_match_metric = doc_exact_match(out_label_list,preds_list)

    results = {
        'loss': eval_loss,
        'precision': precision_score(out_label_list, preds_list),
        'recall': recall_score(out_label_list, preds_list),
        'f1': f1_score(out_label_list, preds_list),
    }

    print(results)

    # ------------------------------ With BIO Post Processing ------------------------------ #
    # preds_list_2 = preds_list.copy()

    # for i, sequence in enumerate(preds_list):
    #     prev_tag = None
    #     for j, tag in enumerate(sequence):
    #         if tag.startswith('I-') and (prev_tag is None or prev_tag == 'O' or prev_tag.startswith('B-')):
    #             # If I-tag is not following a B-tag or O-tag, set it to B-tag
    #             preds_list_2[i][j] = 'B-' + tag[2:]
    #         prev_tag = tag
    

    

    # ------------------------------------- Logs update ------------------------------------ #
    logging.info('Average evaluation loss: ' + str(eval_loss))
    logging.info(results)
    logging.info(classification_report(out_label_list, preds_list, digits=4))

    if print_results:
        print('Average evaluation loss: ' + str(eval_loss))
        print(results)
        print(classification_report(out_label_list, preds_list, digits=4))
        print(
            f'Document Exact Match = {doc_ex_match_metric} on {len(out_label_list)} documents')

    return eval_loss


def roberta_collate_fn(batch):
    """
    Collate function for LayoutLM model training.

    Args:
        batch (list): A list of dictionaries containing the input data for each sample in the batch.

    Returns:
        list: A list of tensors containing the collated input data for the batch.
    """
    final_data = [torch.tensor([]) for _ in range(len(batch[0].keys()))]
    for i, key in enumerate(batch[0].keys()):
        final_data[i] = torch.stack(
            [torch.tensor(batch_item[key], dtype=torch.long) for batch_item in batch])
