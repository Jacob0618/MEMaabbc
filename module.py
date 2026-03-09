"""
Solomon: 以 Qwen（decoder-only causal LM）为骨干的 Prompt Distillation 推荐模型。

训练时的序列结构：
  [soft_prompts | input_tokens | target_tokens]
   ← -100 masked → ← -100 masked → ← loss here →

推理时只给模型前两段，由 generate() 续写目标。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# whole_word_embeddings 的最大序列长度（推荐序列远不会超过此值）
_MAX_SEQ_LEN = 2048


class Solomon(nn.Module):

    def __init__(self):
        super().__init__()
        # 占位，由 from_pretrained 填充
        self.lm = None
        self.config = None

    # ------------------------------------------------------------------
    # 构造 / 反序列化
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_name_or_path, torch_dtype=torch.float16, **kwargs):
        """加载 Qwen（或任意 causal LM）并返回 Solomon 实例。"""
        obj = cls()
        obj.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        obj.config = obj.lm.config
        return obj

    # ------------------------------------------------------------------
    # Prompt 初始化
    # ------------------------------------------------------------------

    def init_prompt(self, task_num, prompts_per_task, device,
                    tokenizer=None, template_texts=None):
        emsize = self.lm.get_input_embeddings().weight.size(1)
        self.prompts_per_task = prompts_per_task
        self.model_device = device

        self.prompt_embeddings = nn.Embedding(task_num * prompts_per_task, emsize)
        self.whole_word_embeddings = nn.Embedding(_MAX_SEQ_LEN, emsize)
        self.recency_alpha = nn.Parameter(torch.tensor(1.0))

        # pad token id（用于在 forward 时区分有效 target 和 padding）
        pad_id = None
        if tokenizer is not None:
            pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = getattr(self.config, 'pad_token_id', None)
        if pad_id is None:
            pad_id = getattr(self.config, 'eos_token_id', 0)
        self.pad_token_id = pad_id

        if tokenizer is not None and template_texts is not None:
            self._init_prompt_from_templates(tokenizer, template_texts)
        else:
            self.prompt_embeddings.weight.data.uniform_(-0.1, 0.1)

        self.prompt_offset = torch.arange(prompts_per_task).to(device)

    def _init_prompt_from_templates(self, tokenizer, template_texts):
        """用离散模板的 token embedding 均值初始化 prompt 向量。"""
        all_token_ids = []
        for text in template_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(ids)

        embed_weight = self.lm.get_input_embeddings().weight
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long,
                                     device=embed_weight.device)
        with torch.no_grad():
            template_embs = embed_weight[all_token_ids]          # (T, emsize)
            total = self.prompt_embeddings.weight.size(0)
            T = template_embs.size(0)
            if T >= total:
                init_embs = template_embs[:total]
            else:
                repeats = (total // T) + 1
                init_embs = template_embs.repeat(repeats, 1)[:total]
            self.prompt_embeddings.weight.data.copy_(init_embs)
            # 加微小噪声打破对称性
            self.prompt_embeddings.weight.data += \
                torch.randn_like(self.prompt_embeddings.weight.data) * 0.01

    # ------------------------------------------------------------------
    # 近因加权
    # ------------------------------------------------------------------

    def apply_recency_weight(self, embeddings, recency_ids):
        """对历史物品 token 施加指数近因加权（最新物品 ≈ 1.0，越旧越小）。"""
        if recency_ids is None:
            return embeddings
        max_pos = recency_ids.max(dim=1, keepdim=True)[0].float().clamp(min=1)
        normalized = recency_ids.float() / max_pos          # 0~1，1=最新
        alpha = torch.nn.functional.softplus(self.recency_alpha)
        weight = torch.exp(-alpha * (1.0 - normalized))
        mask = (recency_ids > 0).float()
        final_weight = mask * weight + (1.0 - mask)         # 非历史 token 权重=1
        return embeddings * final_weight.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Embedding 辅助
    # ------------------------------------------------------------------

    def input_plus_whole_word(self, input_ids, whole_word_ids, recency_ids=None):
        text_emb = self.lm.get_input_embeddings()(input_ids)
        whole_word_emb = self.whole_word_embeddings(whole_word_ids)
        text_emb_plus = text_emb + whole_word_emb
        text_emb_plus = self.apply_recency_weight(text_emb_plus, recency_ids)
        return text_emb_plus

    def append_prompt(self, task_id, text_emb, attention_mask):
        batch_size = task_id.size(0)
        task_ids = (task_id * self.prompts_per_task).unsqueeze(1) + \
                   self.prompt_offset.repeat(batch_size, 1)        # (B, K)
        prompt = self.prompt_embeddings(task_ids)                   # (B, K, emsize)
        input_emb = torch.cat([prompt, text_emb], dim=1)
        prompt_mask = torch.ones(batch_size, self.prompts_per_task,
                                 dtype=torch.int64, device=self.model_device)
        input_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        return input_emb, input_mask

    # ------------------------------------------------------------------
    # Forward（训练 / 验证）
    # ------------------------------------------------------------------

    def forward(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        recency_ids=None,
        labels=None,
        **kwargs,
    ):
        """
        labels: (B, tgt_len)  — target token ids（T5/Qwen tokenizer 产生），
                padding 位置为 pad_token_id 或 0。
        """
        # 1. 构建输入 embedding（含 whole_word 与 recency 加权）
        text_emb = self.input_plus_whole_word(input_ids, whole_word_ids, recency_ids)

        # 2. 拼接软 prompt
        if task_id is not None:
            text_emb, attention_mask = self.append_prompt(task_id, text_emb, attention_mask)

        src_len = text_emb.size(1)
        batch_size = text_emb.size(0)
        dev = input_ids.device

        if labels is not None:
            # --- 训练 / 验证 ---
            # 处理 labels padding：pad_token_id 和 0 都视为无效 token
            pad_mask = (labels == self.pad_token_id) | (labels == 0)
            safe_labels = labels.clone()
            safe_labels[pad_mask] = 0

            # target embedding
            tgt_emb = self.lm.get_input_embeddings()(safe_labels)   # (B, tgt_len, emsize)
            tgt_mask = (~pad_mask).long()                             # (B, tgt_len)

            # 完整序列 = [prompt + input | target]
            all_emb = torch.cat([text_emb, tgt_emb], dim=1)
            all_mask = torch.cat([attention_mask, tgt_mask], dim=1)

            # loss 只算 target 位置
            lm_labels = torch.cat([
                torch.full((batch_size, src_len), -100, dtype=torch.long, device=dev),
                labels.masked_fill(pad_mask, -100),
            ], dim=1)

            return self.lm(
                inputs_embeds=all_emb,
                attention_mask=all_mask,
                labels=lm_labels,
            )
        else:
            # --- 推理（不需要 target）---
            return self.lm(inputs_embeds=text_emb, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    # Beam Search（推理）
    # ------------------------------------------------------------------

    def my_beam_search(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        recency_ids=None,
        max_length=30,
        num_beams=20,
        num_beam_groups=1,
        early_stopping=True,
        min_length=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        num_return_sequences=20,
        bad_words_ids=None,
    ):
        text_emb = self.input_plus_whole_word(input_ids, whole_word_ids, recency_ids)
        if task_id is not None:
            text_emb, attention_mask = self.append_prompt(task_id, text_emb, attention_mask)

        gen_kwargs = dict(
            inputs_embeds=text_emb,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping,
        )
        if min_length > 1:
            gen_kwargs['min_new_tokens'] = min_length
        if num_beam_groups > 1 and diversity_penalty > 0.0:
            gen_kwargs['num_beam_groups'] = num_beam_groups
            gen_kwargs['diversity_penalty'] = diversity_penalty
        if repetition_penalty != 1.0:
            gen_kwargs['repetition_penalty'] = repetition_penalty
        if bad_words_ids is not None:
            gen_kwargs['bad_words_ids'] = bad_words_ids

        return self.lm.generate(**gen_kwargs)

    # ------------------------------------------------------------------
    # 保证 model_device / prompt_offset 随 .to() 一起迁移
    # ------------------------------------------------------------------

    def to(self, device_or_dtype=None, *args, **kwargs):
        result = super().to(device_or_dtype, *args, **kwargs)
        if isinstance(device_or_dtype, (str, torch.device)):
            self.model_device = torch.device(device_or_dtype)
            if hasattr(self, 'prompt_offset'):
                self.prompt_offset = self.prompt_offset.to(self.model_device)
        return result
