import os
import torch
import random
import argparse
from transformers import AutoTokenizer
from utils import SeqDataLoader, TopNBatchify, now_time, evaluate_ndcg, evaluate_hr
from module import Solomon


parser = argparse.ArgumentParser(description='POD (PrOmpt Distillation) — TopN evaluation')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-7B',
                    help='HuggingFace model name (must match the one used in pretrain.py)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./pod/',
                    help='directory to load the final model')
parser.add_argument('--negative_num', type=int, default=99,
                    help='number of negative items for top-n recommendation')
parser.add_argument('--num_beams', type=int, default=20,
                    help='number of beams')
parser.add_argument('--top_n', type=int, default=10,
                    help='number of items to predict')
parser.add_argument('--max_history_len', type=int, default=10,
                    help='max number of recent interaction items included as context')
parser.add_argument('--max_new_tokens', type=int, default=30,
                    help='max tokens to generate per candidate (item IDs are short)')
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')

###############################################################################
# Load data
###############################################################################

TOPN_TASK_ID = 0

print(now_time() + 'Loading data')
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
topn_iterator = TopNBatchify(
    seq_corpus.user2items_positive, seq_corpus.user2items_negative,
    args.negative_num, nitem, tokenizer, args.batch_size,
    task_id=TOPN_TASK_ID, max_history_len=args.max_history_len,
)

###############################################################################
# Test the model
###############################################################################


def generate():
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, recency, _ = topn_iterator.next_batch_test()
            task = task.to(device)
            source = source.to(device)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            recency = recency.to(device)

            beam_outputs = model.my_beam_search(
                task_id=task,
                input_ids=source,
                whole_word_ids=whole_word,
                attention_mask=source_mask,
                recency_ids=recency,
                max_length=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.top_n,
            )

            # beam_outputs: (batch_size * top_n, seq_len)
            output_tensor = beam_outputs.view(task.size(0), args.top_n, -1)
            for i in range(task.size(0)):
                results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
                idss_predict.append(results)

            if topn_iterator.step == topn_iterator.total_step:
                break
    return idss_predict


# Load the best saved model.
print(now_time() + 'Loading checkpoint from {}'.format(model_path))
with open(model_path, 'rb') as f:
    model = torch.load(f, map_location=device, weights_only=False)
model.to(device)

# Run on test data.
print(now_time() + 'Generating recommendations')
idss_predicted = generate()
print(now_time() + 'Evaluation')
user2item_test = {}
for user, item_list in seq_corpus.user2items_positive.items():
    user2item_test[user] = [int(item_list[-1])]
user2rank_list = {}
for predictions, user in zip(idss_predicted, topn_iterator.user_list):
    prediction_list = []
    for p in predictions:
        try:
            prediction_list.append(int(p.split(' ')[0]))
        except Exception:
            prediction_list.append(random.randint(1, nitem))
    user2rank_list[user] = prediction_list

top_ns = [1]
if args.top_n >= 5:
    for i in range(1, (args.top_n // 5) + 1):
        top_ns.append(i * 5)
for top_n in top_ns:
    hr = evaluate_hr(user2item_test, user2rank_list, top_n)
    print(now_time() + 'HR@{} {:7.4f}'.format(top_n, hr))
    ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)
    print(now_time() + 'NDCG@{} {:7.4f}'.format(top_n, ndcg))
