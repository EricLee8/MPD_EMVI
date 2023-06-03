import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from utils.utils import *


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast),
}
TRANSFORMER_CLASS = {'bert': 'bert', 'electra': 'electra'}
CLS_INDEXES = {'bert': 0, 'electra': 0}
model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class ClassificationModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.hidden_size = config.hidden_size

        if args.model_type == 'bert':
            self.bert = BertModel(config, add_pooling_layer=False)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.indicator_embs = nn.Embedding(2, config.hidden_size)
        self.pred_head = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sep_poses=None,
        indicator_ids=None,
        labels=None,
        output_attentions=False
    ):
        training = labels is not None
        transformer = getattr(self, self.transformer_name)
        assert input_ids is not None

        inputs_embeds = transformer.get_input_embeddings()(input_ids) + self.indicator_embs(indicator_ids)

        single_output = transformer(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = single_output.last_hidden_state
        bsz = hidden_states.shape[0]
        pooler_out = []
        for bidx in range(bsz):
            s, e = sep_poses[bidx][-1]
            pooler_out.append(hidden_states[bidx, s: e, :].mean(dim=0))
        pooler_out = torch.stack(pooler_out, dim=0) # (bsz, hsz)
        logits = self.pred_head(pooler_out).squeeze(-1) # (bsz)

        if training:
            choice_loss_fct = BCEWithLogitsLoss()
            choice_loss = choice_loss_fct(logits, labels)
            outputs = (choice_loss, {}) # loss dict contains nothing
        else:
            preds = torch.sigmoid(logits) # (bsz)
            outputs = (preds,)

        return outputs
