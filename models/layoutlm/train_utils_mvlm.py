from typing import List

import numpy as np
import torch
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from seqeval.scheme import Token
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import AdamW, LayoutLMForTokenClassification, AutoTokenizer

from deepinsurancedocs.utils.metrics import doc_exact_match


def train_epoch(model: LayoutLMForTokenClassification, train_dataloader: DataLoader,
                optimizer: AdamW, device: torch.device, global_step, writers, csv_data, pad_token_label_id):
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

    for batch in tqdm(train_dataloader, desc='Training'):
        input_ids = batch[0].to(device)
        bbox = batch[3].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[4].to(device)

        # ------------------------------------ Forward pass ------------------------------------ #
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)

        loss = outputs.loss

        if global_step % 100 == 0:
            print(f"Loss after {global_step} steps: {loss.item()}")

        # ------------------------- Backward pass to get the gradients ------------------------- #
        loss.backward()

        # ------------------------------------- Logs update ------------------------------------ #
        pad_token_label_id = CrossEntropyLoss().ignore_index
        writer_train = writers['train']

        logits = outputs.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=2)
        out_label_ids = labels.detach().cpu().numpy()

        out_label_ids = out_label_ids.tolist()
        preds = preds.tolist()

        # Compare predictions with original MLM labels
        masked_positions = [[out_label_ids[i][j] == pad_token_label_id for j in range(len(out_label_ids[i]))] for i in range(len(out_label_ids))]

        accuracy = np.mean([sklearn_accuracy_score(np.take(out_label_ids[i], masked_positions[i]), np.take(preds[i], masked_positions[i])) for i in range(len(preds))])

        writer_train.add_scalar("Loss/check", loss, global_step=global_step)
        writer_train.add_scalar("total/accuracy/train",
                                accuracy, global_step=global_step)

        csv_data['train'].append((global_step, float(loss)))
        csv_data['accuracy'].append((global_step, float(accuracy)))

        # ---------------------------------- Optimizer update ---------------------------------- #
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    return loss, global_step


def eval(model: LayoutLMForTokenClassification, eval_dataloader: DataLoader,
         device: torch.device, pad_token_label_id: int, logging, print_results=True):
    """ evaluation for each epoch

    Parameters
    ----------
    model : LayoutLMForTokenClassification
        The model used for evaluation.
    eval_dataloader : DataLoader
        The evaluation dataloader.
    device : torch.device
        The device.
    pad_token_label_id : int
        The id of pad token.


    Returns
    -------
    float
        The average evaluation loss
    """
    mask_token_id = AutoTokenizer.from_pretrained(model.config.name_or_path).mask_token_id
    eval_loss = 0.0
    nb_eval_steps = 0
    preds: np.ndarray = np.array([])
    out_label_ids: np.ndarray = np.array([])

    iterator = eval_dataloader
    if print_results:
        iterator = tqdm(eval_dataloader, desc='Evaluating')
    for batch in iterator:
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[3].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[4].to(device)

            # ------------------------------------ Forward pass ------------------------------------ #
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if len(preds) == 0:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    # doc_ex_match_metric = doc_exact_match(out_label_ids, preds)

    masked_positions = [[out_label_ids[i][j] == pad_token_label_id for j in range(len(out_label_ids[i]))] for i in
                        range(len(out_label_ids))]
    accuracy = np.mean(
        [sklearn_accuracy_score(np.take(out_label_ids[i], masked_positions[i]), np.take(preds[i], masked_positions[i]))
         for i in range(len(preds))])
    results = {
        'loss': eval_loss,
        'accuracy': accuracy,
    }

    # ------------------------------------- Logs update ------------------------------------ #
    logging.info('Average evaluation loss: ' + str(eval_loss))
    logging.info(results)
    logging.info(classification_report(out_label_ids, preds, digits=4))

    if print_results:
        print('Average evaluation loss: ' + str(eval_loss))
        print(results)
        # print(classification_report(out_label_ids, preds, digits=4))
        # print(
        #     f'Document Exact Match = {doc_ex_match_metric} on {len(out_label_ids)} documents')

    return eval_loss

def layoutlm_collate_fn(batch):
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

    return final_data