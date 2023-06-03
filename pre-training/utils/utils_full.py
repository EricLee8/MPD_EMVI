import gc
import os
import json
import torch
import random
import typing
import argparse
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy
from string import punctuation


# CONFIGS ===============================================================
USE_CUDA = True
USE_ARRAY = True
METRICS = ['r@1', 'r@2', 'mrr']


parser = argparse.ArgumentParser(description='Parameters for Multi-party Dialogue Response Selection')

# storage path argument
parser.add_argument('--store_path', type=str, default='.', help="path for storage (output, checkpoints, caches, data, etc.)")

# training arguments
parser.add_argument('-lr', '--learning_rate', type=float, default=4e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1926817)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='bert')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=256)
parser.add_argument('-rml', '--response_max_length', type=int, default=46)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--small', type=int, default=0, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='ubuntu', choices=['ubuntu', 'reddit', 'friends', 'molweni'])
parser.add_argument('--logging_times', type=int, default=-1)
parser.add_argument('--cpu_count', type=int, default=16)
parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2]) # 0: training, 1: evaluate, 2: evaluate e-step

# arguments for ReduceLrOnPlateau Training or normal training with early stop
parser.add_argument('--step_patience', type=int, default=1, help='tolerate how many evaluations without improving')
parser.add_argument('--patience', type=int, default=5, help='patience until early stop')

# arguments for models and data
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--fp16', type=int, default=1)
parser.add_argument('--addr', type=int, default=0, choices=[0], help='mode to use the addressee information')
parser.add_argument('--indicator', type=int, default=0, help='whether to use indicator embeddings to indicate addressees')
# 0: no prompt information
parser.add_argument('--negative', type=int, default=1, choices=[0, 1, 2], help='the difficulty-level of negative examples')
# 0: only simple negative examples: sample from the whole corpus
# 1: add moderate negative: sample from the same session
# 2: add hard negative: the response context is the same while the addressee is wrong

# arguments for pre-training
parser.add_argument('--init_path', type=str, default='', help='path of the simulated initializing parser')
parser.add_argument('--addr_window', type=int, default=8, help='maximum utterances distances a speaker can speak to')
parser.add_argument('--force_accuracy', type=float, default=-1.0, help='use the top-k confidence samples to force the accuracy to reach this value')
parser.add_argument('--reinit', type=int, default=1, help='Reinitialize model in each k epoch, (means there are k M-steps)')
parser.add_argument('--init_rounds', type=int, default=1, help='How many epochs of training before the first E-step')
parser.add_argument('--min_mstep_rounds', type=int, default=1, help='The minimal M-step rounds')
parser.add_argument('--trunk_size', type=int, default=600000, help='How many samples in a trunk to perform EM pre-training')
parser.add_argument('--trunk_em_rounds', type=int, default=2, help='how many rounds of EM training for a trunk')
parser.add_argument('--resume', type=int, default=1, help='whether to resume training states from the last training')
parser.add_argument('--keep_layers', type=int, default=0, help='how many layers to keep (not re-init) when preparing the next round of EM')
parser.add_argument('--mlm_weight', type=float, default=0.0, help='weight of mlm loss, 0.0 to turn off this auxiliary task')
parser.add_argument('--attn_weight', type=float, default=0.0, help='weight of the attention regularization loss, 0.0 to turn of this auxiliary task')
parser.add_argument('--attn_heads_ratio', type=float, default=0.5, help='ratio of attention heads that are force to focus on the addressee')
parser.add_argument('--link_weight', type=float, default=0.0, help='weight of the link prediction loss (KL Divergence), 0.0 to turn off this auxiliary task')
parser.add_argument('--n_hop', type=int, default=3, help="how many hops to trace the ancestors")
parser.add_argument('--max_utter_num', type=int, default=20, help='just set it as 20...')
parser.add_argument('--stage_two_trunk', type=int, default=10)

# arguments for fine-tuning
parser.add_argument('--pretrain_path', type=str, default=None, help='whether to use EM pretrained model to finetune, if not None then yes')
parser.add_argument('--del_indicator_embs', type=int, default=0, help='whether to delete indicator embedding weights during finetuning')

args = parser.parse_args()

CPU_COUNT = args.cpu_count
with open(os.path.join(args.store_path, "stop_words.json"), "r", encoding='utf-8') as f:
    STOP_WORDS = set(json.load(f) + list(punctuation))
random.seed(args.seed)
args.data_path = os.path.join(args.store_path, args.data_path)
args.model_file += "_full"

if 'embindicator' in args.model_file:
    args.indicator = 1
if args.init_path != '':
    args.save_path = 'trunk{}_EMrounds{}_reinit{}_initrounds{}_minmstep{}_force{}_pre{}'.format(args.trunk_size,\
         args.trunk_em_rounds, args.reinit, args.init_rounds, args.min_mstep_rounds, args.force_accuracy, args.save_path)
if args.pretrain_path is not None:
    args.save_path = "fine{}".format(args.save_path)
    if args.del_indicator_embs:
        args.save_path = "delfine{}".format(args.save_path)
if "large" in args.model_name:
    args.save_path = args.save_path.replace("save", "lgsave")

args.save_root = '{}_full_saves'.format(args.dataset) if not args.small else '{}_full_saves_small'.format(args.dataset)
if 'ablation' in args.model_file:
    args.save_root = "{}_ablation_saves".format(args.dataset)
if args.mlm_weight > 0.0:
    args.save_root = "mlm_" + args.save_root
    args.save_path = "mlm{}_".format(args.mlm_weight) + args.save_path
if args.attn_weight > 0.0:
    args.save_root = "attn_" + args.save_root
    args.save_path = "attn{}_".format(args.attn_weight) + args.save_path
if args.link_weight > 0.0:
    args.save_root = "link_" + args.save_root
    args.save_path = "link{}_hop{}_".format(args.link_weight, args.n_hop) + args.save_path
args.cache_root = 'caches'
args.save_root = os.path.join(args.store_path, args.save_root)
args.cache_root = os.path.join(args.store_path, args.cache_root)

args.save_path = args.save_root + '/' + args.model_type + '_negamode{}_'.format(args.negative) + '{}_lr{}_'.format(args.model_file, args.learning_rate) +\
     ('small{}_'.format(args.small) if args.small else '') + args.save_path
args.cache_path = args.cache_root + '/' + args.model_type + ('_large_' if 'large' in args.model_name else '_') + args.cache_path

args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
# END ===================================================================


class text(object):
    def __init__(self, lines):
        self.imgToAnns = {}
        for _id, line in enumerate(lines):
            self.imgToAnns[_id] = [{"caption": line}]
        
    def getImgIds(self):
        return self.imgToAnns.keys()


class otherTrainingStates(object):
    def __init__(self):
        self.training_state_dict = {}
    
    def update(self, training_state_dict):
        self.training_state_dict.update(training_state_dict)

    def save_state_dict(self, path):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(self.training_state_dict, f, indent=2)
    
    def load_state(self, path):
        with open(path, "r", encoding='utf-8') as f:
            training_state_dict = json.load(f)
        self.update(training_state_dict)

    def state_dict(self):
        return self.training_state_dict


class StringArray(object):
    def _string_to_sequence(self, s: str, dtype=np.int32) -> np.ndarray:
        return np.array([ord(c) for c in s], dtype=dtype)

    def _sequence_to_string(self, seq: np.ndarray) -> str:
        return ''.join([chr(c) for c in seq])

    def _pack_sequences(self, seqs: typing.Union[np.ndarray, list]):
        values = np.concatenate(seqs, axis=0)
        offsets = np.cumsum([len(s) for s in seqs])
        return values, offsets

    def _unpack_sequence(self, index: int) -> np.ndarray:
        off1 = self.strings_o[index]
        if index > 0:
            off0 = self.strings_o[index - 1]
        elif index == 0:
            off0 = 0
        else:
            raise ValueError(index)
        return self.strings_v[off0:off1]

    def __init__(self, strings):
        self.len = len(strings)
        seqs = [self._string_to_sequence(s) for s in strings]
        self.strings_v, self.strings_o = self._pack_sequences(seqs)

    def __getitem__(self, i):
        seq = self._unpack_sequence(i)
        string = self._sequence_to_string(seq)
        return string
    
    def __len__(self):
        return self.len


class LazyLargeDataset(data.Dataset):
    def __init__(self, context_list, spk_list, adr_list, examples, tokenizer, prefix, record_offset, e_step=False, mlm_examples=None, training=True):
        self.context_list = context_list
        self.spk_list = spk_list
        self.adr_list = adr_list
        self.examples = examples
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.record_offset = record_offset
        self.mlm_examples = mlm_examples
        self.training = training
        self.e_step = e_step
        self.tokenizer_speaker_len = len(tokenizer.tokenize("#speaker#"))

    def __getitem__(self, index):
        data_info = {}
        elements = [int(x) for x in self.examples[index].strip().split('\t')]
        context_idx, offset = elements[:2]
        context_idx -= self.record_offset
        if not self.e_step:
            negative_idx, negative_offset, label = elements[2:]
            negative_idx = negative_idx % len(self.context_list) # for small
            negative_offset = negative_offset % len(self.context_list[negative_idx].split(' @@@ '))
        else:
            addr = elements[-1] # here addr index is from 0 since it is from range(0, offset)
        utterances = self.context_list[context_idx].split(' @@@ ')[:offset]
        response = self.context_list[context_idx].split(' @@@ ')[offset]\
            if self.e_step or label in [-2, 1] else self.context_list[negative_idx].split(' @@@ ')[negative_offset]
            # hard negative (-2), the response is not changed

        response_info = {}
        response_info['response'] = response
        response_info['ans_spk'] = "#speaker{}#".format(self.spk_list[context_idx][offset])
        if not self.e_step:
            if label != -2:
                addr = self.adr_list[context_idx][offset]
            else:
                true_addr = self.adr_list[context_idx][offset]
                if true_addr == negative_idx: # during e-step, the predicted addr happans to be the randomly generated negative_addr
                    negative_idx = random.randint(1, true_addr+1) # TODO
                addr = negative_idx # for hard negative, we store the wrong addr in variable "negative_idx"
            addr = addr - 1 # in training the addr index is from 0... this is really confusing... I don't know why i am doing this...
            if addr < 0 or addr >= offset:
                addr = offset - 1
        response_info['ans_adr'] = addr
        
        addrs = [x-1 for x in self.adr_list[context_idx][:offset]] if args.link_weight > 0.0 or (args.attn_weight > 0.0 and self.training and not self.e_step) else None
        if args.link_weight > 0.0:
            addrs.append(addr)

        input_ids, attention_mask, token_type_ids, sep_poses, indicator_ids, attention_guidance, link_labels = convert_single_example(\
             ["#speaker{}#".format(x) for x in self.spk_list[context_idx][:offset]], utterances, response_info, self.tokenizer, addrs)
        label = None if self.e_step else max(label, 0) # to map labels that is less than 0 to 0
        link_masks = torch.ones(args.max_utter_num, args.max_utter_num, dtype=torch.long).tril_(diagonal=-1)
        link_masks[0, -1] = 1 # act like padding when calculating ancestor embeddings

        if self.mlm_examples is not None and self.training and not self.e_step:
            mlm_dict = self._get_mlm_item(index)
            data_info['mlm_dict'] = mlm_dict

        data_info['qid_tensor'] = torch.tensor(index, dtype=torch.long)
        data_info['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        data_info['sep_poses'] = sep_poses
        data_info['attention_guidance'] = torch.tensor(attention_guidance, dtype=torch.long) if\
            attention_guidance is not None else None
        data_info['link_labels'] = torch.tensor(link_labels, dtype=torch.long) if\
            link_labels is not None else None
        data_info['link_masks'] = link_masks
        data_info['indicator_ids'] = torch.tensor(indicator_ids, dtype=torch.long) if\
            indicator_ids is not None else None
        data_info['labels'] = torch.tensor(label, dtype=torch.float) if label is not None else None
        return data_info

    def _get_mlm_item(self, index):
        data_info = {}
        index = index % len(self.mlm_examples)
        elements = self.mlm_examples[index].strip().split('\t')
        context_idx, offset = [int(x) for x in elements[:2]]
        context_idx -= self.record_offset
        preliminary_masks = set(json.loads(elements[-1]))
        cur_context = self.context_list[context_idx].split(' @@@ ')
        utterances = cur_context[:offset]
        addr = int(elements[2]) if self.e_step else (self.adr_list[context_idx][offset] - 1) # index from 1
        response = cur_context[offset]
        response_with_speaker = "#speaker{}#".format(self.spk_list[context_idx][offset]) + ": " + cur_context[offset]

        masked_response = get_masked_response(preliminary_masks, response, cur_context[addr], self.tokenizer, e_step=self.e_step)
        # masked_response = get_masked_response(preliminary_masks, response, "", self.tokenizer)

        total_speaker_len = len(self.tokenizer.tokenize("#speaker{}#".format(self.spk_list[context_idx][offset])))
        remain_length = args.response_max_length-total_speaker_len-1 - len(self.tokenizer.tokenize(masked_response)) # -1 for the ":"
        if remain_length >= 0: # padding on the right
            masked_response += " ".join([self.tokenizer.pad_token]*remain_length)
        else: # truncation on the right
            masked_response = " ".join(masked_response.split(" "))[:args.response_max_length] # preliminary truncation
            cur_len = len(self.tokenizer.tokenize(masked_response))
            while cur_len > args.response_max_length-total_speaker_len-1:
                masked_response = " ".join(masked_response.split(" ")[:-1])
                cur_len = len(self.tokenizer.tokenize(masked_response))
            remain_length = args.response_max_length-total_speaker_len-1 - cur_len
            if remain_length >= 0: # padding on the right
                masked_response += " ".join([self.tokenizer.pad_token]*remain_length)
        assert len(self.tokenizer.tokenize(masked_response)) == args.response_max_length - total_speaker_len - 1, "{} vs. {}".format(\
             len(self.tokenizer.tokenize(masked_response)), args.response_max_length-total_speaker_len-1)
    
        response_info = {}
        response_info['response'] = masked_response

        masked_speaker_len = total_speaker_len - self.tokenizer_speaker_len
        response_info['ans_spk'] = "#speaker{}#".format(' '.join([self.tokenizer.mask_token]*masked_speaker_len))
        # response_info['ans_spk'] = "#speaker{}#".format(self.spk_list[context_idx][offset])

        response_info['ans_adr'] = addr # here addr index is from 0 since it is from range(0, offset)
        if response_info['ans_adr'] < 0 or response_info['ans_adr'] >= offset:
            response_info['ans_adr'] = offset - 1

        input_ids, attention_mask, token_type_ids, sep_poses, indicator_ids, _, _ = convert_single_example(\
             ["#speaker{}#".format(x) for x in self.spk_list[context_idx][:offset]], utterances, response_info, self.tokenizer)
        response_start, response_end = sep_poses[-1]
        self.tokenizer.truncation_side = 'right'
        label = self.tokenizer.encode(response_with_speaker, truncation=True, max_length=args.response_max_length+2)[1:-1] # truncate the [CLS] and [SEP]
        self.tokenizer.truncation_side = 'left'
        label += [self.tokenizer.pad_token_id] * (args.response_max_length - len(label))
        
        a = self.tokenizer.convert_ids_to_tokens(input_ids[response_start: response_end-1])
        b = self.tokenizer.convert_ids_to_tokens(label)
        assert len(a) == len(b), "\n{}\n vs. \n{}\n".format(a, b)
        assert args.response_max_length == len(label)

        for idx, token_id in enumerate(input_ids[response_start: response_end-1]):
            if token_id != self.tokenizer.mask_token_id:
                label[idx] = -100 # we ignore those positions that are not masked when calculating MLM loss

        data_info['qid_tensor'] = torch.tensor(index, dtype=torch.long)
        data_info['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        data_info['sep_poses'] = sep_poses
        data_info['indicator_ids'] = torch.tensor(indicator_ids, dtype=torch.long) if\
            indicator_ids is not None else None
        data_info['mlm_labels'] = torch.tensor(label, dtype=torch.long) if label is not None else None
        return data_info

    def __len__(self):
        return len(self.examples)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def create_trunk_file(example_lines, record_lines, mlm_lines=None):
    save_path = os.path.join(args.data_path, args.dataset, f"trunks_{args.trunk_size}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Split data trunks to {}...".format(save_path))
    total_trunk_num = len(example_lines) // args.trunk_size
    cur_mlm_start, cur_mlm_end = 0, 0
    for trunk_idx in tqdm(range(total_trunk_num), ncols=100):
        example_start, example_end = trunk_idx*args.trunk_size, min((trunk_idx+1)*args.trunk_size, len(example_lines))
        cur_example_lines = example_lines[example_start: example_end]
        record_start, record_end = int(cur_example_lines[0].split('\t')[0]), int(cur_example_lines[-1].split('\t')[0]) + 1
        cur_record_lines = record_lines[record_start: record_end]
        if mlm_lines is not None:
            cur_mlm_context_id = int(mlm_lines[cur_mlm_end].split('\t')[0])
            while cur_mlm_context_id < record_end:
                cur_mlm_end += 1
                cur_mlm_context_id = int(mlm_lines[cur_mlm_end].split('\t')[0])
            cur_mlm_lines = mlm_lines[cur_mlm_start: cur_mlm_end]
            cur_mlm_start = cur_mlm_end
        cur_save_path = os.path.join(save_path, str(trunk_idx))
        if not os.path.exists(cur_save_path):
            os.mkdir(cur_save_path)
        cur_example_save_path = os.path.join(cur_save_path, "examples.txt")
        if not os.path.exists(cur_example_save_path):
            with open(cur_example_save_path, "w", encoding='utf-8') as f:
                f.writelines(cur_example_lines)
        cur_record_save_path = os.path.join(cur_save_path, "records.json")
        if not os.path.exists(cur_record_save_path):
            with open(cur_record_save_path, "w", encoding='utf-8') as f:
                f.writelines(cur_record_lines)
        cur_mlm_save_path = os.path.join(cur_save_path, "mlm.txt")
        if mlm_lines is not None and not os.path.exists(cur_mlm_save_path):
            with open(cur_mlm_save_path, "w", encoding='utf-8') as f:
                f.writelines(cur_mlm_lines)
    print("Trunk data spliting is done!!!")


def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            tf = os.path.join(dir_path, file_name)
            del_files(tf)
        os.rmdir(dir_path)
    

def collate_fn(data):
    data_info = {}
    float_type_keys = ['labels']
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = torch.stack(data_info[k])
        if isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                if not isinstance(data_info[k][0][id_key], torch.Tensor):
                    new_dict[id_key] = [sub_dict[id_key] for sub_dict in data_info[k]]
                    continue
                id_key_list = [torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float) for sub_dict in data_info[k]] # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = id_key_tensor
            data_info[k] = new_dict
    return data_info


def convert_single_example(speakers, utterances, response_info, tokenizer, addrs=None):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_sep_poses(input_ids):
        sep_poses = []
        last_idx = 0 # except the [CLS]
        for idx, inpidx in enumerate(input_ids):
            if inpidx == tokenizer.sep_token_id:
                sep_poses.append((last_idx+1, idx+1))
                last_idx = idx
        return sep_poses
    
    def _clean_pad_attention_mask(ids_dict):
        for idx in range(len(ids_dict['attention_mask'])):
            if ids_dict['input_ids'][idx] == tokenizer.pad_token_id:
                ids_dict['attention_mask'][idx] = 0
        return ids_dict

    context = ''
    for speaker, utterance in list(zip(speakers, utterances)):
        context += tokenizer.sep_token + ' ' + speaker + ": " + utterance + ' '
    context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
    response = response_info['ans_spk'] + ": " + response_info['response']

    try:
        ids_dict = tokenizer.encode_plus(context, response, padding='max_length',\
                truncation='only_first', max_length=args.max_length)
    except:
        ids_dict = tokenizer.encode_plus(context, response, padding='max_length',\
                truncation=True, max_length=args.max_length)
    ids_dict = _clean_pad_attention_mask(ids_dict)

    input_ids = ids_dict['input_ids']
    attention_mask = ids_dict['attention_mask']
    token_type_ids = ids_dict['token_type_ids']

    sep_poses = _get_sep_poses(input_ids)
    indicator_ids = [0] * len(input_ids)
    offset = len(utterances) - (len(sep_poses) - 1) # deal with truncation
    # # DEBUG ==============================================================================
    # if len(tokenizer.encode(context, response)) <= args.max_length:
    #     assert offset == 0
    # # DEBUG ==============================================================================
    true_addr = response_info['ans_adr'] - offset

    if true_addr >= 0:
        s, e = sep_poses[true_addr]
        for idx in range(s, e):
            indicator_ids[idx] = 1
    
    attention_guidance, link_labels = None, None
    if addrs is not None and args.attn_weight > 0.0: # get the attention guidance matrix
        attn_addrs = addrs[:-1]
        slen = len(input_ids)
        attention_guidance = [[0]*slen for _ in range(slen)]
        for idx in range(1, len(sep_poses)-1):
            s, e = sep_poses[idx]
            addr = attn_addrs[idx]
            addr_s, addr_e = sep_poses[addr]
            for response_token_idx in range(s, e):
                for addr_token_idx in range(addr_s, addr_e):
                    attention_guidance[response_token_idx][addr_token_idx] = 1
    
    if addrs is not None and args.link_weight > 0.0: # get the link prediction labels
        cur_max_addr_num = len(sep_poses)
        cur_offset = len(addrs) - cur_max_addr_num
        assert cur_offset == offset
        link_labels = [x-offset for x in addrs[-cur_max_addr_num:]]
        link_labels[0] = -100 # the first utterance has no reply
        if len(link_labels) < args.max_utter_num:
            link_labels += [-100] * (args.max_utter_num - len(link_labels))
        for idx, link_label in enumerate(link_labels):
            if link_label < 0 and link_label != -100:
                assert link_label + offset >= 0, f"link_label: {link_label}, offset: {offset}"
                link_labels[idx] = -100
            try:
                assert link_labels[idx] == -100 or 0 <= link_labels[idx] < idx, f"invalid link_label: {link_labels[idx]}, idx: {idx}, offset: {offset}, cur_offset: {cur_offset}"
            except AssertionError:
                link_labels[idx] = -100

    return input_ids, attention_mask, token_type_ids, sep_poses, indicator_ids, attention_guidance, link_labels


def get_masked_response(preliminary_masks, response, addressee_utterance, tokenizer, e_step=False):
    masked_response_tokens = []
    # addr_tokens = set() if e_step else set(addressee_utterance.split()) # E-step, avoid different masks in different addressees
    addr_tokens = set(addressee_utterance.split())
    for idx, token in enumerate(response.split()):
        if idx in preliminary_masks or token in addr_tokens and token not in STOP_WORDS:
            masked_response_tokens += [tokenizer.mask_token] * len(tokenizer.tokenize(token))
        else:
            masked_response_tokens.append(token)
    masked_response = ' '.join(masked_response_tokens)
    return masked_response


def get_dataset(tokenizer, postfix, trunk_idx=0, record_lines=None, example_lines=None, confidence_method="rank",\
         training=True, e_step=None, accelerator=None, train_eval=False, epoch=None):
    log_fun = print if accelerator is None else accelerator.print

    if record_lines is None: # training set
        save_path = os.path.join(args.data_path, args.dataset, f"trunks_{args.trunk_size}")
        log_fun("Reading records and examples from {}...".format(save_path))
        with open(os.path.join(save_path, str(trunk_idx), "records.json"), "r", encoding='utf-8') as f:
            cur_record_lines = f.readlines()
        with open(os.path.join(save_path, str(trunk_idx), "examples.txt"), "r", encoding='utf-8') as f:
            cur_example_lines = f.readlines()
        if args.mlm_weight > 0.0:
            with open(os.path.join(save_path, str(trunk_idx), "mlm.txt"), "r", encoding='utf-8') as f:
                mlm_examples = f.readlines()
        else:
            mlm_examples = None
        if train_eval:
            cur_record_lines = cur_record_lines[:10000]
        log_fun("Reading is done!!!")
        record_start = int(cur_example_lines[0].split('\t')[0]) if not train_eval else 0
        
    else:
        if postfix == 'train':
            example_start, example_end = trunk_idx*args.trunk_size, min((trunk_idx+1)*args.trunk_size, len(example_lines))
            cur_example_lines = deepcopy(example_lines[example_start: example_end])
            record_start, record_end = int(cur_example_lines[0].split('\t')[0]), int(cur_example_lines[-1].split('\t')[0]) + 1
            cur_record_lines = deepcopy(record_lines[record_start: record_end])
        else:
            record_start = 0
            cur_example_lines = deepcopy(example_lines)
            cur_record_lines = deepcopy(record_lines)
        mlm_examples = None
        del record_lines, example_lines
        gc.collect()

    log_fun("Building records...")
    if e_step is not None and e_step:
        context_list, spk_list, adr_list = [], [], []
        for line in tqdm(cur_record_lines, ncols=100, disable=accelerator is not None and not accelerator.is_local_main_process):
            record = json.loads(line.strip())
            context_list.append(' @@@ '.join(record['context'] + ([record['answer']] if 'answer' in record else [])))
            spk_list.append(record['ctx_spk'] + ([record['ans_spk']] if 'ans_spk' in record else []))
            del record

        log_fun("Preparing data for expectation step...")
        examples = []
        for idx, line in enumerate(tqdm(cur_record_lines, ncols=100, disable=accelerator is not None and not accelerator.is_local_main_process)):
            record = json.loads(line.strip())
            record['ctx_adr'] = record['ctx_adr'] + ([record['ans_adr']] if 'ans_adr' in record else [])
            context_idx = idx + record_start
            for offset in range(3, len(record['ctx_adr'])):
                for addr in range(max(0, int(offset)-args.addr_window), offset): # here addr index is from 0
                    examples.append(f"{context_idx}\t{offset}\t{addr}")
            adr_list.append(record['ctx_adr'])
            del record
        del cur_record_lines
        gc.collect()

        dataset = LazyLargeDataset(StringArray(context_list), spk_list, adr_list, StringArray(examples),\
             tokenizer, postfix, record_start, e_step=e_step, training=training)

    else: # normal training or m-step
        context_list, spk_list, adr_list = [], [], []

        # m-step =================================================================
        if e_step is not None:
            addr_pred_path = os.path.join(args.save_path, "addr_preds.json")
            log_fun("Reading predicted addressees from {}".format(addr_pred_path))
            with open(addr_pred_path, "r", encoding='utf-8') as f:
                addr_preds = json.load(f)
        # =========================================================================

        for idx, line in enumerate(tqdm(cur_record_lines, ncols=100, disable=accelerator is not None and not accelerator.is_local_main_process)):
            record = json.loads(line.strip())
            record['ctx_adr'] = record['ctx_adr'] + ([record['ans_adr']] if 'ans_adr' in record else [])
            context_list.append(' @@@ '.join(record['context'] + ([record['answer']] if 'answer' in record else [])))
            spk_list.append(record['ctx_spk'] + ([record['ans_spk']] if 'ans_spk' in record else []))
            # m-step =================================================================
            if e_step is not None: # m-step
                qid = postfix + '-' + str(idx+record_start)
                for offset in range(3, len(record['ctx_adr'])): # this "3" is hard-coded, i know it's bad but... other ways are also bad
                    qqid = qid + '-' + str(offset)
                    pred_addr = addr_preds[qqid]['pred'] + 1 # the predicted addr index is from 0, +1 to make sure in record, it is from 1
                    record['ctx_adr'][offset] = pred_addr
            # =========================================================================
            adr_list.append(record['ctx_adr'])
            del record
        
        # m-step ===============================================================================================
        if e_step is not None:
            for qqid in addr_preds.keys():
                if confidence_method == "rank":
                    break
                log_probs = addr_preds[qqid]['log_probs']
                # min-max scaler
                if confidence_method == "minmax_rank":
                    min_log_prob = min(log_probs)
                    max_log_prob = max(log_probs)
                    denominator = max_log_prob - min_log_prob
                    log_probs = [(x-min_log_prob)/denominator for x in log_probs]
                    addr_preds[qqid]['minmax_confidence'] = max_log_prob - sorted(log_probs, reverse=True)[1] # max - second
                # avg scaler
                elif confidence_method == "avg_rank":
                    min_log_prob = min(log_probs)
                    log_probs = [x-min_log_prob for x in log_probs] # make sure all elements > 0
                    avg = sum(log_probs) / len(log_probs)
                    addr_preds[qqid]['avg_confidence'] = (max(log_probs) - sorted(log_probs, reverse=True)[1]) / avg # (max - second) / avg

            if confidence_method == "minmax_rank":
                addr_preds = dict(sorted(addr_preds.items(), key=lambda x: x[1]['minmax_confidence'], reverse=True))
                for idx, (qid, _) in enumerate(addr_preds.items()):
                    addr_preds[qid]['minmax_rank'] = idx + 1

            elif confidence_method == "avg_rank":
                addr_preds = dict(sorted(addr_preds.items(), key=lambda x: x[1]['avg_confidence'], reverse=True))
                for idx, (qid, _) in enumerate(addr_preds.items()):
                    addr_preds[qid]['avg_rank'] = idx + 1

            else:
                assert confidence_method == "rank", "You should specify confidence method in [rank|minmax_rank|avg_rank] !!!"

            if args.force_accuracy > 0.0:
                confidence_rank = int(len(addr_preds) * args.force_accuracy)
        # =============================================================================================================

        del cur_record_lines
        gc.collect()

        examples = cur_example_lines

        # m-step ===============================================================================================================
        # if e_step is not None and args.force_accuracy > 0.0 and epoch >= 0:
        if e_step is not None and args.force_accuracy > 0.0:
            real_examples = []
            for line in examples:
                context_idx, offset = [int(x) for x in line.strip().split('\t')[:2]]
                qqid = "{}-{}-{}".format(postfix, context_idx, offset)
                if addr_preds[qqid][confidence_method] <= confidence_rank:
                    real_examples.append(line)
            ori_len, cur_len = len(examples), len(real_examples)
            log_fun("Used examples: {}/{}, ratio: {}%".format(cur_len, ori_len, round(cur_len/ori_len*100, 3)))
            examples = real_examples
            # mlm =================================================================================================
            if mlm_examples is not None and training and (e_step is None or not e_step):
                mlm_real_examples = []
                for line in mlm_examples:
                    context_idx, offset = [int(x) for x in line.strip().split('\t')[:2]]
                    qqid = "{}-{}-{}".format(postfix, context_idx, offset)
                    if addr_preds[qqid][confidence_method] <= confidence_rank:
                        mlm_real_examples.append(line)
                ori_len, cur_len = len(mlm_examples), len(mlm_real_examples)
                log_fun("Used examples for MLM: {}/{}, ratio: {}%".format(cur_len, ori_len, round(cur_len/ori_len*100, 3)))
                mlm_examples = mlm_real_examples
            # ======================================================================================================
        # ========================================================================================================================

        dataset = LazyLargeDataset(StringArray(context_list), spk_list, adr_list, StringArray(examples),\
             tokenizer, postfix, record_start, mlm_examples=mlm_examples, training=training)

    return dataset


if __name__ == "__main__":
    from transformers import BartTokenizerFast, BertTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    print("Loading tokenizer...")
    # tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=args.cache_path)
    tokenizer.truncation_side = 'left'
    train_dataset = get_dataset(tokenizer, "train",  trunk_idx=0, confidence_method="rank",\
            training=True, e_step=False)
    sampler = SequentialSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=64)
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        # print(batch["qid"][0])
        # print(batch["input_ids"].shape)
        # print(batch["attention_mask"].shape)
        # print(batch["labels"].shape)
        # print(batch["sep_poses"][0])
        pass