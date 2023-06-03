import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import DebertaV2Model, DebertaV2Config, DebertaV2PreTrainedModel, DebertaV2Tokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from utils.utils_full import *


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast),
    'deberta': (DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Tokenizer)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'electra': 'electra', 'deberta': 'deberta'}
CLS_INDEXES = {'bert': 0, 'electra': 0, 'deberta': 0}
model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MultipleChoiceModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        if args.model_type == 'bert':
            self.bert = BertModel(config)
            if args.mlm_weight > 0.0:
                self.cls = BertOnlyMLMHead(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)
            if args.mlm_weight > 0.0:
                self.cls = nn.Linear(config.embedding_size, config.vocab_size)
        elif args.model_type == 'deberta':
            self.deberta = DebertaV2Model(config)
            if args.mlm_weight > 0.0:
                self.cls = DebertaV2OnlyMLMHead(config)

        self.indicator_embs = nn.Embedding(2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sep_poses=None,
        indicator_ids=None,
        e_step=False,
        labels=None,
        link_labels=None,
        link_masks=None,
        mlm_dict=None,
        tau=0.1
    ):
        training = labels is not None
        transformer = getattr(self, self.transformer_name)
        assert input_ids is not None

        inputs_embeds = transformer.get_input_embeddings()(input_ids) + self.indicator_embs(indicator_ids)

        single_output = transformer(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )
        if args.model_type == 'bert':
            pooler_out = single_output[1] # (bsz, hsz)
        else:
            hidden_states = single_output[0] # (bsz, slen, hsz)
            pooler_out = hidden_states[:, self.cls_index, :] # (bsz, hsz)
        logits = self.classifier(pooler_out).squeeze(-1) # (bsz)

        if training:
            # compute context-response matching loss
            choice_loss_fct = BCEWithLogitsLoss()
            choice_loss = choice_loss_fct(logits, labels)
            total_loss = choice_loss
            loss_dict = {"CL": choice_loss.item()}
            # compute MLM loss
            if mlm_dict is not None:
                mlm_input_ids, mlm_token_type_ids, mlm_attention_mask, mlm_indicator_ids, mlm_sep_poses, mlm_labels = mlm_dict['input_ids'],\
                    mlm_dict['token_type_ids'], mlm_dict['attention_mask'], mlm_dict['indicator_ids'], mlm_dict['sep_poses'], mlm_dict['mlm_labels']
                mlm_inputs_embeds = transformer.get_input_embeddings()(mlm_input_ids) + self.indicator_embs(mlm_indicator_ids)
                mlm_output = transformer(
                    inputs_embeds=mlm_inputs_embeds,
                    token_type_ids=mlm_token_type_ids,
                    attention_mask=mlm_attention_mask,
                    output_attentions=False
                )
                mlm_hidden_states = mlm_output[0] # (bsz, slen, hsz)
                bsz = mlm_hidden_states.shape[0]
                response_tensors = []
                for bidx in range(bsz):
                    response_start, _ = mlm_sep_poses[bidx][-1]
                    cur_response_states = mlm_hidden_states[bidx][response_start: response_start+args.response_max_length, :] # (rlen, hsz)
                    response_tensors.append(cur_response_states)
                response_states = torch.stack(response_tensors, dim=0) # (bsz, rlen, hsz)
                mlm_logits = self.cls(response_states) # (bsz, rlen, vocab_size)
                mlm_loss_fct = CrossEntropyLoss()
                mlm_loss = mlm_loss_fct(mlm_logits.view(-1, self.vocab_size), mlm_labels.view(-1)) # (bsz*rlen if reduction is none else 1)
                total_loss += mlm_loss * args.mlm_weight
                loss_dict['ML'] = mlm_loss.item()
            outputs = (total_loss, loss_dict)
        else:
            outputs = (logits,)

        return outputs
