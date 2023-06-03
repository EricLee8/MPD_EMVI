import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
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


class ParsingModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.hidden_size = config.hidden_size
        self.num_relations = args.num_relations

        if args.model_type == 'bert':
            self.bert = BertModel(config, add_pooling_layer=False)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.indicator_embs = nn.Embedding(2, config.hidden_size)
        self.link_cls = nn.Sequential(
            nn.Linear(config.hidden_size*4, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        self.relation_cls = nn.Sequential(
            nn.Linear(config.hidden_size*4, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, self.num_relations)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        indicator_ids=None,
        sep_poses=None,
        link_labels=None,
        relation_labels=None,
        output_attentions=False
    ):
        training = link_labels is not None and relation_labels is not None
        transformer = getattr(self, self.transformer_name)
        assert input_ids is not None
        if training:
            link_loss, relation_loss = 0.0, 0.0
        else:
            link_preds, relation_preds = [], []

        # Embedding for link prediction (w/o addressee embedding)
        single_output = transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = single_output.last_hidden_state
        bsz = hidden_states.shape[0]
        loss_fct = CrossEntropyLoss()

        for bidx in range(bsz):
            # get utterance embeddings
            cur_utter_embs = []
            for s, e in sep_poses[bidx]:
               cur_utter_embs.append(hidden_states[bidx, s: e, :].mean(dim=0))
            cur_utter_embs = torch.stack(cur_utter_embs[:-1], dim=0) # (context_len, hsz)
            cur_response_embs = cur_utter_embs[-1]
            cur_response_embs_expanded = cur_utter_embs[-1].unsqueeze(0).expand_as(cur_utter_embs) # (context_len, hsz)

            # for link prediction
            cur_link_logits = self.link_cls(torch.cat([cur_utter_embs, cur_response_embs_expanded,\
                 cur_utter_embs-cur_response_embs_expanded, cur_utter_embs*cur_response_embs_expanded], dim=-1)).squeeze(-1) # (context_len)
            if training:
                link_loss += loss_fct(cur_link_logits.unsqueeze(0), link_labels[bidx].unsqueeze(0)) / bsz
                cur_link_label = link_labels[bidx].item()
            else: # inference
                cur_link_label = torch.argmax(cur_link_logits, dim=-1).item()
                link_preds.append(cur_link_label)
        
        # Embedding for relation classification (w/ addressee embedding)
        if not training: # inference, use predicted link as addressee
            for bidx in range(bsz):
                s, e = sep_poses[bidx][link_preds[bidx]]
                indicator_ids[bidx, s: e] = 1

        inputs_embeds = transformer.get_input_embeddings()(input_ids) + self.indicator_embs(indicator_ids)
        single_output = transformer(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = single_output.last_hidden_state
        
        for bidx in range(bsz):
            # get utterance embeddings
            cur_utter_embs = []
            for s, e in sep_poses[bidx]:
               cur_utter_embs.append(hidden_states[bidx, s: e, :].mean(dim=0))
            cur_utter_embs = torch.stack(cur_utter_embs[:-1], dim=0) # (context_len, hsz)
            cur_response_embs = cur_utter_embs[-1]
            cur_response_embs_expanded = cur_utter_embs[-1].unsqueeze(0).expand_as(cur_utter_embs) # (context_len, hsz)

            # for relation classification
            cur_addr_embs = cur_utter_embs[link_labels[bidx].item() if training else link_preds[bidx]] # (hsz)
            relation_logits = self.relation_cls(torch.cat([cur_addr_embs, cur_response_embs,\
                 cur_addr_embs-cur_response_embs, cur_addr_embs*cur_response_embs], dim=-1)) # (n_relations)
            if training:
                relation_loss += loss_fct(relation_logits.unsqueeze(0), relation_labels[bidx].unsqueeze(0)) / bsz
            else: # inference
                relation_preds.append(torch.argmax(relation_logits, dim=-1).item())

        if training:
            total_loss = link_loss + relation_loss
            loss_dict = {'LL': link_loss.item(), 'RL': relation_loss.item()}
            outputs = (total_loss, loss_dict,)
        else:
            outputs = (link_preds, relation_preds,)

        return outputs
