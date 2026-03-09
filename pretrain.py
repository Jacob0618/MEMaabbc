import os
import torch
import argparse
import torch.nn as nn
from transformers import T5Tokenizer
from module import Solomon
from utils import SeqDataLoader, TrainBatchify, TopNBatchify, now_time
from templates import topn_templates, topn_templates_no_history
from peft import get_peft_model, LoraConfig, TaskType


parser = argparse.ArgumentParser(description='POD (PrOmpt Distillation) — TopN-focused')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--task_num', type=int, default=1,
                    help='task number (currently only topN)')
parser.add_argument('--prompt_num', type=int, default=100,
                    help='number of continuous prompt vectors per task')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64,
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
                    help='if set, fine-tune the LM backbone via LoRA; otherwise only train prompt')
parser.add_argument('--lora_r', type=int, default=16,
                    help='LoRA rank (only used when --finetune_lm is set)')
args = parser.parse_args()

if args.model_version == 1:
    model_version = 't5-base'
elif args.model_version == 2:
    model_version = 't5-large'
elif args.model_version == 3:
    model_version = 't5-3b'
elif args.model_version == 4:
    model_version = 't5-11b'
else:
    model_version = 't5-small'

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
tokenizer = T5Tokenizer.from_pretrained(model_version)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)

# topN task_id = 0 (唯一任务)
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

model = Solomon.from_pretrained(model_version)

# 用 topN 模板的 token embedding 初始化 prompt 向量
template_texts = [t.format(0, '0', '0') for t in topn_templates] + \
                 [t.format(0, '0') for t in topn_templates_no_history]
model.init_prompt(args.task_num, args.prompt_num, device,
                  tokenizer=tokenizer, template_texts=template_texts)
model.to(device)

if args.finetune_lm:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v", "o", "wi", "wo"],
        modules_to_save=["prompt_embeddings", "whole_word_embeddings"],
    )
    model = get_peft_model(model, peft_config)
    # recency_alpha 不在 modules_to_save 中，需手动开启梯度
    for name, param in model.named_parameters():
        if 'recency_alpha' in name:
            param.requires_grad = True
    model.print_trainable_parameters()
else:
    # 冻结语言模型主干，只训练 prompt 相关参数
    for param in model.parameters():
        param.requires_grad = False
    for param in model.prompt_embeddings.parameters():
        param.requires_grad = True
    for param in model.whole_word_embeddings.parameters():
        param.requires_grad = True
    model.recency_alpha.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}')

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

        if all_iterator.batch_index % args.log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
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
