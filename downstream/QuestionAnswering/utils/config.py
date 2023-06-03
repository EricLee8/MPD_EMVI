import os
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='Parameters for Molweni dataset')

parser.add_argument('--store_path', type=str, default='.', help="path for storage (output, checkpoints, caches, data, etc.)")

parser.add_argument('-lr', '--learning_rate', type=float, default=7e-5)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=5)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='bert')
parser.add_argument('-mn', '--model_name', type=str, default='bert-base-uncased')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=384)
parser.add_argument('-mun', '--max_utter_num', type=int, default=14)
parser.add_argument('-bsz', '--batch_size', type=int, default=16)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-6)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--dataset', type=str, default='molweni', choices=['friends', 'molweni'])
parser.add_argument('--pred_file', type=str, default='pred.json')
parser.add_argument('--na_file', type=str, default='na.json')

parser.add_argument('--model_file', type=str, default='baseline')
parser.add_argument('--colab', type=int, default=1)
parser.add_argument('--num_decouple_layers', type=int, default=1)
parser.add_argument('--num_sup_layers', type=int, default=3)

# arguments for loading pre-trained models
parser.add_argument('--pretrain_path', type=str, default='None')

args = parser.parse_args()

args.data_path = os.path.join(args.store_path, args.data_path)
args.save_root = 'saves/' if not args.small else 'saves_small/'
args.cache_root = 'caches/' if not args.small else 'caches_small/'
args.save_root = os.path.join(args.store_path, args.save_root)
args.cache_root = os.path.join(args.store_path, args.cache_root)
if "None" not in args.pretrain_path:
    args.pretrain_path = os.path.join(args.store_path[:args.store_path.rfind('/')], args.pretrain_path)
    args.save_root += "_pretrained"
    if 'mlm' in args.pretrain_path:
        args.save_root += '_mlm'

if not os.path.exists(args.save_root):
    os.mkdir(args.save_root)
if not os.path.exists(args.cache_root):
    os.mkdir(args.cache_root)

if 'BiDeN' not in args.model_file and 'BIDM' not in args.model_file:
    args.num_decouple_layers = 0
if 'SUP' in args.model_file:
    args.question_max_length = 32

args.save_path = args.save_root + args.model_type +\
     '_' + '{}L{}lr{}'.format(args.model_file, args.num_decouple_layers,\
     args.learning_rate) + '_' + args.save_path
args.cache_path = args.cache_root + args.model_type + '_' + ('sup_' if 'SUP' in args.model_file else '') + args.cache_path

args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)

args.pred_file = args.save_path + '/' + args.pred_file
args.na_file = args.save_path + '/' + args.na_file

