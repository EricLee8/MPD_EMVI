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
        self.n_hop = args.n_hop

        if args.model_type == 'bert':
            self.bert = BertModel(config, add_pooling_layer=False)
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

        self.link_classifier = nn.Sequential() # for debug use this named modules
        self.link_classifier.add_module("L1", nn.Linear(config.hidden_size*4, config.hidden_size))
        self.link_classifier.add_module("Tanh", nn.Tanh())
        self.link_classifier.add_module("L2", nn.Linear(config.hidden_size, 1))

        self.response_classifier = nn.Sequential() # for debug use this named modules
        self.response_classifier.add_module("L1", nn.Linear(config.hidden_size*(self.n_hop+1), config.hidden_size))
        self.response_classifier.add_module("Tanh", nn.Tanh())
        self.response_classifier.add_module("L2", nn.Linear(config.hidden_size, 1))

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
        
        hidden_states = single_output.last_hidden_state
        bsz = hidden_states.shape[0]
        link_loss_fct = CrossEntropyLoss()
        train_link_labels = link_labels.clone()
        utter_embs, response_embs, batch_utter_nums = [], [], []

        # get utterance_embedding
        for bidx in range(bsz):
            cur_all_utter_embs = []
            for s, e in sep_poses[bidx]:
               cur_all_utter_embs.append(hidden_states[bidx, s: e, :].mean(dim=0)) # (hsz)
            cur_utter_num = len(cur_all_utter_embs) - 1 # the last utterance is the response
            train_link_labels[bidx][cur_utter_num] = -100
            batch_utter_nums.append(cur_utter_num)
            cur_utter_embs = torch.stack(cur_all_utter_embs[:-1], dim=0) # (utter_num, hsz)
            cur_utter_embs = F.pad(cur_utter_embs, (0, 0, 0, args.max_utter_num-cur_utter_num), mode="constant", value=0) # (max_utter, hsz)
            utter_embs.append(cur_utter_embs)
            response_embs.append(cur_all_utter_embs[-1]) # (hsz,)
        
        utter_embs = torch.stack(utter_embs, dim=0) # (bsz, max_utter, hsz)
        response_embs = torch.stack(response_embs, dim=0) # (bsz, hsz)
        utter_num = utter_embs.shape[1]
        expand_utter_embs = utter_embs.unsqueeze(2).expand(-1, -1, utter_num, -1) # (bsz, max_utter, max_utter, hsz)
        transposed_utter_embs = utter_embs.unsqueeze(1).expand(-1, utter_num, -1, -1) # (bsz, max_utter, max_utter, hsz)

        # for link prediction
        link_logits = self.link_classifier(torch.cat([expand_utter_embs, transposed_utter_embs,\
             expand_utter_embs-transposed_utter_embs, expand_utter_embs*transposed_utter_embs], dim=-1)).squeeze(-1) # (bsz, max_utter, max_utter)
        link_logits = link_logits * link_masks - 65500 * (1-link_masks)
        if training:
            link_loss = link_loss_fct(link_logits.view(-1, args.max_utter_num), train_link_labels.view(-1)) # ignore index -100
        
        # for context-response matching
        link_probs = F.gumbel_softmax(link_logits, tau=tau, hard=False, dim=-1) # (bsz, max_utter, max_utter)
        adr_link_probs = []
        for bidx in range(bsz):
            cur_utter_num = batch_utter_nums[bidx]
            assert train_link_labels[bidx, cur_utter_num].item() == -100
            adr_link_idx = link_labels[bidx, cur_utter_num].item()
            cur_initial_link_prob = torch.zeros(args.max_utter_num, dtype=torch.float)
            if adr_link_idx != -100:
                cur_initial_link_prob[adr_link_idx] = 1.
            adr_link_probs.append(cur_initial_link_prob) # (probs over length max_utter)
        adr_link_probs = torch.stack(adr_link_probs, dim=0).to(link_probs.device) # (bsz, max_utter)

        n_hop_utter_indices = []
        for idx in range(self.n_hop):
            cur_ancestor_link_probs = adr_link_probs.unsqueeze(1).clone() # (bsz, 1, max_utter)
            for _ in range(idx):
                cur_ancestor_link_probs = torch.bmm(cur_ancestor_link_probs, link_probs) # (bsz, 1, max_utter)
            n_hop_utter_indices.append(cur_ancestor_link_probs.squeeze(1))

        n_hop_utter_indices = torch.stack(n_hop_utter_indices, dim=1) # (bsz, n_hop, max_utter)
        n_hop_utter_embs = torch.bmm(n_hop_utter_indices, utter_embs).view(bsz, -1) # (bsz, n_hop*hsz)
        logits = self.response_classifier(torch.cat([response_embs, n_hop_utter_embs], dim=-1)).squeeze(-1) # (bsz, (n_hop+1)*hsz) -> (bsz,)

        if training:
            # compute context-response matching loss
            choice_loss_fct = BCEWithLogitsLoss()
            choice_loss = choice_loss_fct(logits, labels)
            total_loss = choice_loss
            loss_dict = {"CL": choice_loss.item()}
            # add the link prediction loss
            total_loss += link_loss * args.link_weight
            loss_dict['KL'] = link_loss.item()
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
