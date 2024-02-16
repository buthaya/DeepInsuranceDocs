from typing import List

import numpy as np
import torch
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import AdamW, LayoutLMPreTrainedModel, LayoutLMForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional, Tuple, Union

from deepinsurancedocs.utils.metrics import doc_exact_match

pad_token_label_id = CrossEntropyLoss().ignore_index
# LABEL_LIST = 

class LayoutLMForMaskedLMInternal(LayoutLMForMaskedLM):
    def __init__(self, config):
        super(LayoutLMForMaskedLMInternal, self).__init__(config)


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
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        first_token_mask: Optional[torch.FloatTensor] = None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Calculate output embedding of LayoutLM for whole sequence 
        outputs = self.layoutlm(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ( =outputs[0] in hugging face code, but having "hidden_states" key is more legible)
        sequence_output = outputs['hidden_states'][-1]

        # Take only the first token for the words that are subtokenized for loss
        # tokens           = [[CLS], 'i', 'lo##', '##ve', 'new', 'york', [SEP]]
        # first_token_mask = [0,      1,   1,       0,      1,     1,    0    ]
        sequence_output = sequence_output[first_token_mask]
        sequence_labels = labels[first_token_mask]

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),sequence_labels.view(-1),)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
