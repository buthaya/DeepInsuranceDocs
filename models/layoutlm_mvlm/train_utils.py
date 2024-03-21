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
from transformers import AdamW, AutoTokenizer, LayoutLMForMaskedLM
from models.layoutlm_mvlm.model import LayoutLMForMaskedLMInternal

from deepinsurancedocs.utils.metrics import doc_exact_match


def train_epoch(model: LayoutLMForMaskedLM, train_dataloader: DataLoader,
                optimizer: AdamW, device: torch.device, global_step, writers, csv_data, pad_token_label_id, accumulation_steps, scheduler):
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

    batch_idx = 0
    total_loss = 0
    for batch in tqdm(train_dataloader, desc='Training MVLM Batch', unit='batch'):
        # input_ids = batch[0].to(device)
        # attention_mask = batch[1].to(device)
        # token_type_ids = batch[2].to(device)
        # labels = batch[3].to(device)
        # bbox = batch[4].to(device)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        bbox = batch["bbox"].to(device)
        masked_tokens_bool = (labels!=pad_token_label_id)

        # ------------------------------------ Forward pass ------------------------------------ #
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                        first_token_mask=masked_tokens_bool)

        loss = outputs.loss
        loss = loss 

        # if global_step % 100 == 0:
        #     print(f"Loss after {global_step} steps: {loss.item()}")

        # ------------------------- Backward pass to get the gradients ------------------------- #
        loss.backward()

        # ------------------------------------- Logs update ------------------------------------ #
        pad_token_label_id = CrossEntropyLoss().ignore_index
        writer_train = writers['train']

        logits = outputs.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        masked_pred_ids_torch = torch.tensor(preds, dtype=torch.long)

        # # Compare predictions with original MLM labels
        masked_label_ids_torch = labels[masked_tokens_bool]

        accuracy = sklearn_accuracy_score(masked_label_ids_torch.detach().cpu(), masked_pred_ids_torch.detach().cpu())

        perplexity = torch.exp(loss)

        csv_data['train']['step'].append(global_step)
        csv_data['train']['loss'].append(float(loss))
        csv_data['train']['accuracy'].append(float(accuracy))
        csv_data['train']['perplexity'].append(float(perplexity))
        writer_train.add_scalar("Loss/mvlm", loss, global_step=global_step)
        writer_train.add_scalar("Metrics/accuracy",accuracy, global_step=global_step)
        writer_train.add_scalar("Metrics/perplexity",perplexity, global_step=global_step)

        global_step += 1
        # ---------------------------------- Optimizer update ---------------------------------- #
        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step(loss)
        total_loss += loss.item()

        batch_idx+=1

    # Print or log the average loss for the epoch
    average_loss = total_loss / len(train_dataloader)

    # Don't forget to perform a final update if the total number of batches is not a multiple of accumulation_steps
    if len(train_dataloader) % accumulation_steps != 0:
        optimizer.step()
    
    return loss, global_step, csv_data


def eval(model: LayoutLMForMaskedLM, 
         eval_dataloader: DataLoader,
         device: torch.device, 
         pad_token_label_id: int, 
         logging,
         print_results=True):
    """ evaluation for each epoch

    Parameters
    ----------
    model : LayoutLMForMaskedLM
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

    # If needed, adjust the mask_token_id for other models (i.e: LayoutLMV3)
    mask_token_id = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased").mask_token_id
    loss = 0.0
    nb_eval_steps = 0
    preds: np.ndarray = np.array([])
    out_label_ids: np.ndarray = np.array([])

    iterator = eval_dataloader
    if print_results:
        iterator = tqdm(eval_dataloader, desc='Evaluating MVLM')
    for batch in iterator:
        with torch.no_grad():
            # input_ids = batch[0].to(device)
            # attention_mask = batch[1].to(device)
            # token_type_ids = batch[2].to(device)
            # labels = batch[3].to(device)
            # bbox = batch[4].to(device)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            bbox = batch["bbox"].to(device)
            masked_tokens_bool = labels!=pad_token_label_id

            # ------------------------------------ Forward pass ------------------------------------ #
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels, first_token_mask=masked_tokens_bool)

            # get the loss and logits
            tmp_loss = outputs.loss
            logits = outputs.logits

            loss += tmp_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if len(preds) == 0:
                preds = np.array(np.argmax(logits.detach().cpu(), axis=1))
                out_label_ids = np.array(labels.detach().cpu())
            else:
                preds = np.append(preds, np.argmax(np.array(logits.detach().cpu().numpy()), axis=1), axis=0)
                out_label_ids = np.append(
                    out_label_ids, np.array(labels.detach().cpu()), axis=0
                )

            del input_ids, attention_mask, token_type_ids, labels, bbox, masked_tokens_bool
    # compute average evaluation loss
    loss = loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(loss))

    masked_pred_ids_torch = torch.tensor(preds, dtype=torch.long)

    # # Compare predictions with original MLM labels
    masked_label_ids_torch = torch.tensor(out_label_ids[out_label_ids!=-100], dtype=torch.long)
    accuracy = sklearn_accuracy_score(masked_label_ids_torch.detach().cpu(), masked_pred_ids_torch.detach().cpu())
    # doc_exact_match_metric = doc_exact_match(out_label_ids, preds)

    results = {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }

    # ------------------------------------- Logs update ------------------------------------ #
    logging.info('Average evaluation loss: ' + str(loss))
    logging.info(results)
    # logging.info(classification_report(out_label_ids, preds, digits=4))

    if print_results:
        print('Average evaluation loss: ' + str(loss))
        print(results)
        # print(classification_report(out_label_ids, preds, digits=4))
        # print(
        #     f'Document Exact Match = {doc_exact_match_metric} on {len(out_label_ids)} documents')

    return results

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
