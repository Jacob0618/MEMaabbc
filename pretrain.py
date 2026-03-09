import os
import torch
import argparse
from transformers import AutoTokenizer
from module import Solomon
from utils import SeqDataLoader, TrainBatchify, TopNBatchify, now_time
from templates import topn_templates, topn_templates_no_history
from peft import get_peft_model, LoraConfig, TaskType


parser = argparse.ArgumentParser(description='POD (PrOmpt Distillation) — TopN with Qwen')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-7B',
                    help='HuggingFace model name or local path. '
                         'Qwen3 dense sizes: 0.6B/1.7B/4B/8B/14B/32B. '
                         'For 7B use Qwen/Qwen2.5-7B if Qwen3-7B is unavailable.')
parser.add_argument('--task_num', type=int, default=1,
                    help='task number (currently only topN)')
parser.add_argument('--prompt_num', type=int, default=100,
                    help='number of continuous prompt vectors per task')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pod/',
                    help='directory to save the final model')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--negative_num', type=int, default=99,
                    help='number of negative items for top-n recommendation')
parser.add_argument('--max_history_len', type=int, default=10,
                    help='max number of recent interaction items included as context')
parser.add_argument('--finetune_lm', action='store_true',
                    help='fine-tune LM backbone via LoRA; otherwise only train prompt')
parser.add_argument('--lora_r', type=int, default=16,
                    help='LoRA rank (only used when --finetune_lm is set)')
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

print(now_time() + 'Loading data')
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
# Qwen 系列的 pad_token 默认等于 eos_token，右 padding 对训练更友好
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)

TOPN_TASK_ID = 0

all_iterator = TrainBatchify(
    seq_corpus.user2items_positive, args.negative_num, nitem,
    tokenizer, args.batch_size,
    task_id=TOPN_TASK_ID, max_history_len=args.max_history_len,
)
topn_iterator = TopNBatchify(
    seq_corpus.user2items_positive, seq_corpus.user2items_negative,
    args.negative_num, nitem, tokenizer, args.batch_size,
    task_id=TOPN_TASK_ID, max_history_len=args.max_history_len,
)

###############################################################################
# Build the model
###############################################################################

print(now_time() + 'Loading model: {}'.format(args.model_name))
model = Solomon.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                trust_remote_code=True)

# 用 topN 模板的 token embedding 初始化 prompt 向量
template_texts = [t.format(0, '0', '0') for t in topn_templates] + \
                 [t.format(0, '0') for t in topn_templates_no_history]
model.init_prompt(args.task_num, args.prompt_num, device,
                  tokenizer=tokenizer, template_texts=template_texts)
model.to(device)

if args.finetune_lm:
    # Qwen3/Qwen2.5 的注意力 + FFN 投影层名称
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        modules_to_save=['prompt_embeddings', 'whole_word_embeddings'],
    )
    model = get_peft_model(model, peft_config)
    # recency_alpha 不在 modules_to_save 中，手动保留梯度
    for name, param in model.named_parameters():
        if 'recency_alpha' in name:
            param.requires_grad = True
    model.print_trainable_parameters()
else:
    # 冻结 LM 主干，只训练三个轻量组件
    for param in model.parameters():
        param.requires_grad = False
    for param in model.prompt_embeddings.parameters():
        param.requires_grad = True
    for param in model.whole_word_embeddings.parameters():
        param.requires_grad = True
    model.recency_alpha.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print('trainable params: {:,} || all params: {:,} || trainable%: {:.4f}'.format(
        trainable, total, 100 * trainable / total))

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)

###############################################################################
# Training code
###############################################################################


def train():
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        task, source, source_mask, whole_word, recency, target = all_iterator.next_batch()
        task = task.to(device)
        source = source.to(device)
        source_mask = source_mask.to(device)
        whole_word = whole_word.to(device)
        recency = recency.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(
            task_id=task,
            input_ids=source,
            whole_word_ids=whole_word,
            attention_mask=source_mask,
            recency_ids=recency,
            labels=target,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = task.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if all_iterator.batch_index % args.log_interval == 0 or \
                all_iterator.batch_index % all_iterator.batch_num == 0:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'topN loss {:4.4f} | {:5d}/{:5d} batches'.format(
                cur_t_loss, all_iterator.batch_index, all_iterator.batch_num))
            text_loss = 0.
            total_sample = 0
        if all_iterator.batch_index % all_iterator.batch_num == 0:
            break


def evaluate(iterator):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, recency, target = iterator.next_batch_valid()
            task = task.to(device)
            source = source.to(device)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            recency = recency.to(device)
            target = target.to(device)
            outputs = model(
                task_id=task,
                input_ids=source,
                whole_word_ids=whole_word,
                attention_mask=source_mask,
                recency_ids=recency,
                labels=target,
            )
            loss = outputs.loss

            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample


with open(model_path, 'wb') as f:
    torch.save(model, f)

print(now_time() + 'Start training')
best_val_loss = float('inf')
endure_count = 0

log_path = os.path.join(args.checkpoint, 'train.log')
log_file = open(log_path, 'a')

for epoch in range(1, args.epochs + 1):
    now = now_time()
    print(now + 'epoch {}'.format(epoch))
    log_file.write(now + 'epoch {}\n'.format(epoch))

    train()

    print(now_time() + 'validation')
    log_file.write(now_time() + 'validation\n')

    val_loss = evaluate(topn_iterator)

    msg = now_time() + 'top-N loss {:4.4f}'.format(val_loss)
    print(msg)
    log_file.write(msg + '\n')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
        log_file.write(now_time() + 'Model saved (best loss)\n')
    else:
        endure_count += 1
        msg = now_time() + 'Endured {} time(s)'.format(endure_count)
        print(msg)
        log_file.write(msg + '\n')

        if endure_count == args.endure_times:
            end_msg = now_time() + 'Cannot endure it anymore | Exiting from early stop'
            print(end_msg)
            log_file.write(end_msg + '\n')
            break

log_file.close()
