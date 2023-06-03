import os
import argparse
from rouge import Rouge

USE_CUDA = True
METRICS = ['Bleu_4', 'ROUGE_L', 'CIDEr', 'METEOR', 'Bleu_1', 'Bleu_2', 'Bleu_3']
FILTER_KEY_WORDS = ["do n't know", "does n't know", "i 'm not sure", "'ll give it a"]
EVALUATOR = Rouge(metrics=['rouge-n', 'rouge-l'],
                    max_n=2,
                    limit_length=True,
                    length_limit=128,
                    length_limit_type='words',
                    apply_avg=True,
                    apply_best=False,
                    alpha=0.5, # Default F1_score
                    weight_factor=1.2,
                    stemming=True)

parser = argparse.ArgumentParser(description='Parameters for Ubuntu IRC Dataset')

parser.add_argument('--store_path', type=str, default='.', help="path for storage (output, checkpoints, caches, data, etc.)")

parser.add_argument('-lr', '--learning_rate', type=float, default=4e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=15)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-emt', '--encoder_model_type', type=str, default='bert')
parser.add_argument('-dmt', '--decoder_model_type', type=str, default='gpt')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=384)
parser.add_argument('-sml', '--response_max_length', type=int, default=50)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--small', type=int, default=0, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='ubuntu', choices=['ubuntu', 'reddit'])
parser.add_argument('--logging_times', type=int, default=-1)
parser.add_argument('--full_data', type=int, default=0)
parser.add_argument('--cpu_count', type=int, default=16)

# parameters for ReduceLrOnPlateau Training
parser.add_argument('--step_patience', type=int, default=1, help='tolerate how many evaluations without improving')
parser.add_argument('--patience', type=int, default=5, help='patience until early stop')

parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--decoder_model_name', type=str, default='gpt2')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--fp16', type=int, default=1)

parser.add_argument('--addr', type=int, default=0, help='mode to use the addressee information')
# 0: no addressee information
# 1: concatenate the addressee information to the end of the dialogue context (like a prompt)
# 2: discard the utterances after the addressee, which means the last utterance is always spoken by the addressee

parser.add_argument('--addr_window', type=int, default=8, help='maximum utterances distances a speaker can speak to')
parser.add_argument('--confidence_ratio', type=float, default=0.5, help='how much proportion to use for pretraining rank on addressee prediction confidence')
parser.add_argument('--indicator', type=int, default=0, help='whether to use indicator embeddings to indicate addressees')
parser.add_argument('--filter_simple', type=float, default=-1.0, help='filter simple responses which can be extract from the context, if rouge-l > this value, filtered')
parser.add_argument('--pretrain_path', type=str, default='None', help='whether to use EM pretrained model to finetune, if not None then yes')
parser.add_argument('--del_indicator_embs', type=int, default=0, help='whether to delete indicator embedding weights during finetuning')

args = parser.parse_args()

CPU_COUNT = args.cpu_count

args.data_path = os.path.join(args.store_path, args.data_path)

if "large" in args.model_name:
    args.save_path = args.save_path.replace("save", "lgsave")
if 'embindicator' in args.model_file:
    args.indicator = 1

save_root = '{}_saves'.format(args.dataset) if not args.small else '{}_saves_small'.format(args.dataset)
cache_root = 'caches'
save_root = os.path.join(args.store_path, save_root)
if args.store_path != ".":
    cache_root = os.path.join(args.store_path[:args.store_path.rfind('storage/')], 'storage', cache_root)

if 'None' not in args.pretrain_path:
    args.pretrain_path = os.path.join(args.store_path[:args.store_path.rfind('/')], args.pretrain_path)
    save_root += "_pretrained"
    if 'mlm' in args.pretrain_path:
        save_root += '_mlm'

args.save_path = save_root + '/' + args.encoder_model_type + '-' + args.decoder_model_type + '_addrmode{}'.format(args.addr) +\
     '_' + '{}lr{}'.format(args.model_file, args.learning_rate) + '_filter{}_'.format(args.filter_simple) + args.save_path
args.cache_path = cache_root

args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
