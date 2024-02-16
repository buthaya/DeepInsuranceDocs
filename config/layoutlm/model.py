from typing import List

import numpy as np
import torch
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchcrf import CRF

from tqdm import tqdm
from transformers import AdamW, LayoutLMForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Tuple, Union

from deepinsurancedocs.utils.metrics import doc_exact_match

pad_token_label_id = CrossEntropyLoss().ignore_index
# LABEL_LIST = 
class LayoutLMForTokenClassificationInternal(LayoutLMForTokenClassification):
    def __init__(self, config):
        super(LayoutLMForTokenClassificationInternal, self).__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        first_token_mask: Optional[torch.FloatTensor] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Calculate output embedding of LayoutLM for whole sequence 
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # ( =outputs[0] in hugging face code, but having "hidden_states" key is more legible)
        sequence_output = outputs['hidden_states'][-1] 
        sequence_output = self.dropout(sequence_output)

        # Take only the first token for the words that are subtokenized for loss
        # tokens           = [[CLS], 'i', 'lo##', '##ve', 'new', 'york', [SEP]]
        # first_token_mask = [0,      1,   1,       0,      1,     1,    0    ]
        sequence_output = sequence_output[first_token_mask]
        sequence_labels = labels[first_token_mask]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), sequence_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
