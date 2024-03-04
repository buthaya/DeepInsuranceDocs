from typing import List

import numpy as np
import torch
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import AdamW, LayoutLMForTokenClassification

from deepinsurancedocs.utils.metrics import doc_exact_match
from models.layoutlm.model import LayoutLMForTokenClassificationInternal

pad_token_label_id = CrossEntropyLoss().ignore_index
# LABEL_LIST = 
# class LayoutLMWithCRF(LayoutLMForTokenClassification):
#     def __init__(self, config):
#         super(LayoutLMWithCRF, self).__init__(config)
#         self.crf = CRF(num_tags=self.config.num_labels, batch_first=True)

#     def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None, labels=None):
#         outputs = super(LayoutLMWithCRF, self).forward(
#             input_ids=input_ids,
#             bbox=bbox,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             labels=labels
#         )
#         # logits = outputs.logits.detach().cpu()
#         logits = outputs.logits.detach()

#         # ------------------------------------ 
#         label_map = dict(enumerate(['O', 'B-BEGIN_PAY_PERIOD', 'I-BEGIN_PAY_PERIOD', 'B-END_PAY_PERIOD', 'I-END_PAY_PERIOD', 'B-GROSS_PAY_PER_PERIOD', 'I-GROSS_PAY_PER_PERIOD', 'B-GROSS_TAXABLE_PER_PERIOD', 'I-GROSS_TAXABLE_PER_PERIOD', 'B-NET_PAY_PER_PERIOD', 'I-NET_PAY_PER_PERIOD', 'B-PAYG_TAX_PER_PERIOD', 'I-PAYG_TAX_PER_PERIOD', 'B-PAY_DATE', 'I-PAY_DATE', 'B-POST_TAX_DEDUCTIONS_PER_PERIOD', 'I-POST_TAX_DEDUCTIONS_PER_PERIOD', 'B-PRE_TAX_DEDUCTION_PER_PERIOD', 'I-PRE_TAX_DEDUCTION_PER_PERIOD']))
#         preds = np.argmax(logits.cpu(), axis=2)
#         preds_list = [[label_map[int(item[j])] for j in item] for item in preds]
#         # -------- Inspect argmax result vs viterbi decode of CRF --------
#         # print(preds_list)
#         crf_labels = labels.detach()
    
#         crf_mask = (crf_labels != pad_token_label_id).byte()[:, 1:] & attention_mask.detach().byte()[:, 1:]
#         crf_labels = crf_labels[:,1:]
#         # filter out padding labels
#         non_pad_mask = crf_labels.flatten() != pad_token_label_id

#         # compute the predictions
#         preds_extension = [item for sublist in self.crf.decode(logits[:, 1:], mask=crf_mask) for item in sublist]
        
#         # ------------------------------------
#         if labels is not None:
#             # [:, 1:] because the first token is always [CLS] in LayoutLM and the CRF cannot start by ignoring 2 tokens
#             crf_mask = (labels != pad_token_label_id).byte()[:, 1:] & attention_mask.byte()[:, 1:]  # Create a mask to ignore positions with value -100

#             crf_labels = labels.detach().clone()[:,1:]
#             crf_labels[crf_labels==pad_token_label_id] = 0

#             loss = -self.crf(logits[:, 1:], crf_labels, mask=crf_mask, reduction='mean')
#             outputs.loss = loss
#         return outputs


def train_epoch(model: LayoutLMForTokenClassificationInternal, train_dataloader: DataLoader, LABEL_LIST: List,
                optimizer: AdamW, device: torch.device, global_step, writers, csv_data, print_results, accumulation_steps, scheduler):
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
    iterator = train_dataloader
    batch_idx = 0
    total_loss = 0
    if print_results:
        iterator = tqdm(train_dataloader, desc='Training TC Batch')
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        bbox = batch["bbox"].to(device)
        first_token_mask = batch["first_token_mask"]

        out_label_list: List[list] = [[] for _ in input_ids]
        preds_list: List[list] = [[] for _ in input_ids]

        # ------------------------------------ Forward pass ------------------------------------ #
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
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

        # for i, label in enumerate(out_label_ids):
        #     if label != pad_token_label_id:
        #         if label_map[label]!='O' and label_map[preds[i]]!='O':
        #             out_label_list.append(label_map[label])
        #             preds_list.append(label_map[preds[i]])

        # out_label_list = out_label_list
        # preds_list = preds_list
        f1 = f1_score(out_label_list, preds_list)
        precision = precision_score(out_label_list, preds_list)
        recall = recall_score(out_label_list, preds_list)

        csv_data['train']['step'].append(global_step)
        csv_data['train']['loss'].append(float(loss))
        csv_data['train']['f1'].append(float(f1))
        csv_data['train']['precision'].append(float(precision))
        csv_data['train']['recall'].append(float(recall))
        writer_train.add_scalar("Loss/check", loss, global_step=global_step)
        writer_train.add_scalar("Metrics/f1",f1, global_step=global_step)
        writer_train.add_scalar("Metrics/precision",precision, global_step=global_step)
        writer_train.add_scalar("Metrics/recall",recall, global_step=global_step)

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

    return loss, global_step


def eval(model: LayoutLMForTokenClassificationInternal, eval_dataloader: DataLoader,
         device: torch.device, pad_token_label_id: int, label_map: dict, logging, print_results=True):
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
        iterator = tqdm(eval_dataloader, desc='Evaluating Token Classification')
    preds_list: List = []
    out_label_list: List = []
    start_idx = 0
    # Eval mode
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            bbox = batch["bbox"].to(device)
            first_token_mask = batch["first_token_mask"]

            # ------------------------------------ Forward pass ------------------------------------ #
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels, first_token_mask=first_token_mask)

            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits
            sequence_labels = labels[first_token_mask]
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if len(preds) == 0:
                preds = np.array(logits.detach().cpu())
                preds = np.argmax(preds, axis=1)
                out_label_ids = np.array(sequence_labels.detach().cpu())
            else:
                preds_logits = np.array(logits.detach().cpu())
                preds_argmax=np.argmax(preds_logits, axis=1)
                preds = np.append(preds, preds_argmax, axis=0)
                out_label_ids = np.append(out_label_ids, np.array(sequence_labels.detach().cpu()), axis=0)

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

    doc_exact_match_metric = doc_exact_match(out_label_list,preds_list)

    results = {
        'loss': eval_loss,
        'precision': precision_score(out_label_list, preds_list),
        'recall': recall_score(out_label_list, preds_list),
        'f1': f1_score(out_label_list, preds_list),
        'doc_exact_match': doc_exact_match_metric
    }

    if print_results:
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
    if logging:
        logging.info('Average evaluation loss: ' + str(eval_loss))
        logging.info(results)
        logging.info(classification_report(out_label_list, preds_list, digits=4))

    if print_results:
        print('Average evaluation loss: ' + str(eval_loss))
        print(results)
        print(classification_report(out_label_list, preds_list, digits=4))
        print(
            f'Document Exact Match = {doc_exact_match_metric} on {len(out_label_list)} documents')

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



# def train_epoch_crf(model: LayoutLMWithCRF, train_dataloader: DataLoader, LABEL_LIST: List,
#                 optimizer: AdamW, device: torch.device, global_step, writers, csv_data):
#     """ training for each epoch

#     Parameters
#     ----------
#     model : TokenClassifier
#         The model used for training.
#     train_dataloader : DataLoader
#         The train dataloader.
#     optimizer : AdamW
#         The optimizer, Adamw
#     device : torch.device
#         The device.
#     """

#     model.train()

#     for batch in tqdm(train_dataloader, desc='Batch training'):

#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         token_type_ids = batch["token_type_ids"].to(device)
#         labels = batch["labels"].to(device)
#         bbox = batch["bbox"].to(device)

#         # ------------------------------------ Forward pass ------------------------------------ #
#         outputs = model(input_ids=input_ids,
#                         bbox=bbox,
#                         attention_mask=attention_mask,
#                         token_type_ids=token_type_ids,
#                         labels=labels)

#         loss = outputs.loss

#         # if global_step % 100 == 0:
#         #     print(f"Loss after {global_step} steps: {loss.item()}")

#         # ------------------------- Backward pass to get the gradients ------------------------- #
#         loss.backward()

#         # ------------------------------------- Logs update ------------------------------------ #
#         label_map = dict(enumerate(LABEL_LIST))
#         pad_token_label_id = CrossEntropyLoss().ignore_index
#         writer_train = writers['train']

#         logits = outputs.logits.detach().cpu().numpy()
#         out_label_ids = labels.detach().cpu().numpy()

#         preds = np.argmax(logits, axis=2)
#         out_label_list: List[list] = [[]
#                                       for _ in range(out_label_ids.shape[0])]
#         preds_list: List[list] = [[] for _ in range(out_label_ids.shape[0])]

#         for i in range(out_label_ids.shape[0]):
#             for j in range(out_label_ids.shape[1]):
#                 if out_label_ids[i, j] != pad_token_label_id:
#                     out_label_list[i].append(label_map[out_label_ids[i][j]])
#                     preds_list[i].append(label_map[preds[i][j]])

#         f1 = f1_score(out_label_list, preds_list)
#         precision = precision_score(out_label_list, preds_list)
#         recall = recall_score(out_label_list, preds_list)

#         writer_train.add_scalar("Loss/check", loss, global_step=global_step)
#         writer_train.add_scalar("Metrics/f1",
#                                 f1, global_step=global_step)
#         writer_train.add_scalar("Metrics/precision",
#                                 precision, global_step=global_step)
#         writer_train.add_scalar("Metrics/recall",
#                                 recall, global_step=global_step)
#         csv_data['train'].append((global_step, float(loss)))
#         csv_data['f1'].append((global_step, float(f1)))
#         csv_data['precision'].append((global_step, float(precision)))
#         csv_data['recall'].append((global_step, float(recall)))

#         # ---------------------------------- Optimizer update ---------------------------------- #
#         optimizer.step()
#         optimizer.zero_grad()
#         global_step += 1

#     return loss, global_step

# def eval_crf(model: LayoutLMWithCRF, eval_dataloader: DataLoader,
#          device: torch.device, pad_token_label_id: int, label_map: dict, logging, print_results=True):
#     """ evaluation for each epoch

#     Parameters
#     ----------
#     model : LayoutLMWithCRF
#         The model used for evaluation.
#     eval_dataloader : DataLoader
#         The evaluation dataloader.
#     device : torch.device
#         The device.
#     pad_token_label_id : int
#         The id of the pad token.
#     label_map : dict
#         The dictionary of label list with label id.

#     Returns
#     -------
#     float
#         The average evaluation loss
#     """
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     preds: List = []
#     out_label_ids: List = []

#     iterator = eval_dataloader
#     if print_results:
#         iterator = tqdm(eval_dataloader, desc='Evaluating')
#     for batch in iterator:
#         with torch.no_grad():
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             token_type_ids = batch["token_type_ids"].to(device)
#             labels = batch["labels"].to(device)
#             bbox = batch["bbox"].to(device)

#             # ------------------------------------ Forward pass ------------------------------------ #
#             outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
#                             token_type_ids=token_type_ids, labels=labels)

#             # get the loss and logits
#             tmp_eval_loss = outputs.loss
#             logits = outputs.logits

#             eval_loss += tmp_eval_loss.item()
#             nb_eval_steps += 1

#             crf_mask = (labels != pad_token_label_id).byte()[:, 1:] & attention_mask.byte()[:, 1:]
#             crf_labels = labels.detach().clone()[:,1:]
#             # filter out padding labels
#             non_pad_mask = crf_labels.flatten() != pad_token_label_id

#             # compute the predictions
#             preds_extension = [item for sublist in model.crf.decode(logits[:, 1:], mask=crf_mask) for item in sublist]
#             preds.extend(preds_extension)
#             out_label_ids.extend(crf_labels.flatten()[non_pad_mask].tolist())

#     # compute average evaluation loss
#     eval_loss = eval_loss / nb_eval_steps

#     preds = np.array(preds)
#     out_label_ids = np.array(out_label_ids)

#     out_label_list = [label_map[label_id] for label_id in out_label_ids.flatten() if label_id != pad_token_label_id]
#     preds_list = [label_map[pred_id] for pred_id in preds.flatten() if pred_id != pad_token_label_id]

#     # filter out padding labels
#     out_label_list = [out_label_list]
#     preds_list = [preds_list]

#     doc_exact_match_metric = doc_exact_match(out_label_list, preds_list)

#     results = {
#         'loss': eval_loss,
#         'precision': precision_score(out_label_list, preds_list, average='weighted'),
#         'recall': recall_score(out_label_list, preds_list, average='weighted'),
#         'f1': f1_score(out_label_list, preds_list, average='weighted'),
#     }

#     print(results)

#     # ------------------------------------- Logs update ------------------------------------ #
#     logging.info('Average evaluation loss: ' + str(eval_loss))
#     logging.info(results)
#     logging.info(classification_report(out_label_list, preds_list, digits=4))

#     if print_results:
#         print('Average evaluation loss: ' + str(eval_loss))
#         print(results)
#         print(classification_report(out_label_list, preds_list, digits=4))
#         print(f'Document Exact Match = {doc_exact_match_metric} on {len(out_label_list)} documents')

#     return eval_loss
