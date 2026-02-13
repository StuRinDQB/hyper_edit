import os
import re
import json
import math
import difflib
import random
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# NEW: sentence transformer
from sentence_transformers import SentenceTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def split_into_paragraphs(text: str) -> List[str]:
    """按双换行符分割段落"""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs

def split_by_single_newline(paragraph: str) -> List[str]:
    """按单换行符进一步分割"""
    lines = paragraph.split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    return lines

def chunk_by_tokens(paragraphs: List[str], tokenizer, max_tokens=2000) -> List[str]:
    """按 token 数量组合段落为块"""
    chunks = []
    current_chunk_tokens = []
    current_count = 0
    
    for paragraph in paragraphs:
        token_ids = tokenizer(paragraph, add_special_tokens=False)["input_ids"]
        p_len = len(token_ids)
        
        if p_len > max_tokens:
            # 段落过长，退化为按行切分
            fallback_lines = split_by_single_newline(paragraph)
            for line in fallback_lines:
                line_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
                line_len = len(line_ids)
                if line_len > max_tokens:
                    # 单行仍过长：直接作为独立块
                    if current_chunk_tokens:
                        chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True).strip()
                        chunks.append(chunk_text)
                        current_chunk_tokens, current_count = [], 0
                    big_line_text = tokenizer.decode(line_ids, skip_special_tokens=True).strip()
                    chunks.append(big_line_text)
                    continue
                # 能放进当前块
                if current_count + line_len <= max_tokens:
                    current_chunk_tokens.extend(line_ids); current_count += line_len
                else:
                    # 切出当前块
                    if current_chunk_tokens:
                        chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True).strip()
                        chunks.append(chunk_text)
                    current_chunk_tokens = line_ids; current_count = line_len
        else:
            if current_count + p_len <= max_tokens:
                current_chunk_tokens.extend(token_ids); current_count += p_len
            else:
                if current_chunk_tokens:
                    chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True).strip()
                    chunks.append(chunk_text)
                current_chunk_tokens = token_ids; current_count = p_len
    
    if current_chunk_tokens:
        chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True).strip()
        chunks.append(chunk_text)
    return chunks

def make_chunk_pairs(context: str, edit_content: str, edit_request: str, tokenizer, max_chunk_tokens=1024) -> List[Dict[str, str]]:
    """创建分块对（原文与目标各自分块后按索引对齐）"""
    context_pars = split_into_paragraphs(context)
    edited_pars = split_into_paragraphs(edit_content)
    context_chunks = chunk_by_tokens(context_pars, tokenizer, max_tokens=max_chunk_tokens)
    edited_chunks = chunk_by_tokens(edited_pars, tokenizer, max_tokens=max_chunk_tokens)
    n = min(len(context_chunks), len(edited_chunks))
    chunk_data = []
    for i in range(n):
        original_chunk = context_chunks[i]
        edited_chunk = edited_chunks[i]
        prompt = (
            f"### Edit Request:\n{edit_request}\n\n"
            f"### Original Text:\n{original_chunk}\n\n"
            f"### Edited Content:\n"
        )
        chunk_data.append({"prompt": prompt, "target": edited_chunk, "context_chunk": original_chunk})
    return chunk_data

# -------------------------------
# 领域识别
# -------------------------------

DOMAINS = {"code": 0, "latex": 1, "sql": 2, "wiki": 3, "other": 4}

def _infer_domain(path_or_name: str) -> str:
    low = str(path_or_name).lower()
    if "code" in low: return "code"
    if "latex" in low: return "latex"
    if "sql" in low: return "sql"
    if "wiki" in low: return "wiki"
    return "other"

# -------------------------------
# 数据集
# -------------------------------

class EditDataset(Dataset):
    """从多个训练文件读取数据并按 token 分块产出样本"""
    def __init__(self, files: List[str], tokenizer, max_chunk_tokens: int = 2000, use_chunking: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.use_chunking = use_chunking
        
        for f in files:
            domain = _infer_domain(f)
            with open(f, "r", encoding="utf-8") as rf:
                data = json.load(rf)
            for ex in data:
                context = ex.get("context", "")
                edit_request = ex.get("edit_request", "")
                edit_content = ex.get("edit_content", "")
                if context and edit_request and edit_content:
                    if self.use_chunking:
                        chunk_pairs = make_chunk_pairs(
                            context, edit_content, edit_request, 
                            tokenizer, max_chunk_tokens
                        )
                        for pair in chunk_pairs:
                            self.items.append({
                                "prompt": pair["prompt"],
                                "target": pair["target"],
                                "context_chunk": pair["context_chunk"],
                                "domain": domain
                            })
                    else:
                        prompt = (
                            f"### Edit Request:\n{edit_request}\n\n"
                            f"### Original Text:\n{context}\n\n"
                            f"### Edited Content:\n"
                        )
                        self.items.append({
                            "prompt": prompt,
                            "target": edit_content,
                            "context_chunk": context,
                            "domain": domain
                        })
        if not self.items:
            raise RuntimeError("未从提供的训练文件中解析到任何样本。")
        print(f"[数据] 总训练样本数（分块后）：{len(self.items)}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class Collator:
    """
    将 prompt + target 拼接，labels 仅计算 target 部分 loss
    另外返回 context_chunk 的 tokenizer 结果以供 diff 计算
    """
    def __init__(self, tokenizer: AutoTokenizer, max_len: int = 4096):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]):
        prompts, targets, domains, ctx_texts = [], [], [], []
        for ex in batch:
            prompts.append(ex["prompt"])
            targets.append(ex["target"])
            ctx_texts.append(ex["context_chunk"])
            domains.append(DOMAINS.get(ex.get("domain", "other"), DOMAINS["other"]))

        tok_p = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tok_t = self.tok(targets, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        tok_c = self.tok(ctx_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)

        B = tok_p.input_ids.size(0)
        input_ids, attention, labels = [], [], []
        for i in range(B):
            p_ids, p_att = tok_p.input_ids[i], tok_p.attention_mask[i]
            t_ids, t_att = tok_t.input_ids[i], tok_t.attention_mask[i]

            # ---- 去掉 padding 部分 ----
            p_len = int(p_att.sum().item())
            t_len = int(t_att.sum().item())
            p_real = p_ids[:p_len]
            t_real = t_ids[:t_len]

            # ---- 拼接并截断 ----
            ids = torch.cat([p_real, t_real], dim=0)
            ids = ids[: self.max_len]
            att = torch.ones_like(ids)

            # ---- 构造 labels，只计算 target 部分 loss ----
            lab = torch.full_like(ids, -100)
            p_real_len = p_real.size(0)
            tgt_len_after_trunc = ids.size(0) - p_real_len
            if tgt_len_after_trunc > 0:
                lab[p_real_len:] = ids[p_real_len:]

            # ---- 收集到列表 ----
            input_ids.append(ids)
            attention.append(att)
            labels.append(lab)

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tok.pad_token_id)
        attention = nn.utils.rnn.pad_sequence(attention, batch_first=True, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "domains": torch.tensor(domains, dtype=torch.long),
            "prompts_only": tok_p,      # prompt 的 token，用于计算 p_len
            "ctx_tok": tok_c,           # context chunk 的 token，用于 diff 计算
            "targets_tok": tok_t,       # 目标 token（可选）
            "raw_prompts": prompts,     # 供句向量编码
        }

# -------------------------------
# 目标层选择与 Hook
# -------------------------------

TARGET_PATTERNS = [
    re.compile(r".*q_proj$"),
    re.compile(r".*k_proj$"),
    re.compile(r".*v_proj$"),
    re.compile(r".*o_proj$"),
    # （保留或移除）FFN：
    # re.compile(r".*down_proj$"),
    # re.compile(r".*gate_proj$"),
]

def find_target_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            for pat in TARGET_PATTERNS:
                if pat.match(name):
                    targets.append((name, mod))
                    break
    targets.sort(key=lambda x: x[0])
    return targets

def _sanitize(name: str) -> str:
    # 点号和不安全字符替换成下划线
    return re.sub(r'[^0-9a-zA-Z_]', '_', name.replace('.', '__'))


class GRUHyperLoRA(nn.Module):
    """
    基于 GRU 的动态 LoRA-A 生成器；B 为静态参数（可训练，但不随上下文改变）。
    - 将全局 embedding 序列作为输入 token
    - GRU 编码序列，结合上下文 pooled_e
    - 为每个目标层生成动态 A
    """
    def __init__(self, hidden_size: int, targets_meta, rank: int = 16,
                 dropout: float = 0.1, num_codes: int = 16, gru_layers: int = 3, d_h: int = 128  ):
        super().__init__()
        self.rank = rank
        self.hidden = hidden_size
        self.targets_meta = targets_meta
        self.num_codes = num_codes
        self.gru_layers = gru_layers

        self.name2safe, self.safe2name = {}, {}
        self.B = nn.ParameterDict()       # 每个目标层的固定 B 矩阵
        self.target_dims = {}

        # 初始化 B
        for name, in_f, out_f in targets_meta:
            safe = _sanitize(name)
            while safe in self.B:
                safe = safe + "_1"
            self.name2safe[name] = safe
            self.safe2name[safe] = name
            self.target_dims[name] = (in_f, out_f)

            B = nn.Parameter(torch.zeros(out_f, rank))  # 改为全零
            self.B[safe] = B

        self.global_embeddings = nn.Embedding(num_codes, 512)
        nn.init.normal_(self.global_embeddings.weight, mean=0.0, std=0.02)

        # 投影到 GRU 输入
        self.input_proj = nn.Linear(512, hidden_size)

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        # 上下文融合
        self.context_fusion = nn.Linear(hidden_size + hidden_size, hidden_size)

        self.layer_id_embed = nn.Embedding(len(targets_meta), hidden_size)
        nn.init.normal_(self.layer_id_embed.weight, mean=0.0, std=0.02)
        self.layer_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=d_h,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.E_local = nn.ParameterDict()     # E^l: (in_f, r)
        self.a_ln = nn.ModuleDict()           # LN over last dim (r + d_h)
        self.a_proj = nn.ModuleDict()         # Linear((r + d_h) -> r)

        for idx, (name, in_f, _out_f) in enumerate(targets_meta):
            safe = self.name2safe[name]
            # E^l
            E = nn.Parameter(torch.empty(in_f, rank))
            nn.init.xavier_uniform_(E)                 # 也可用 zeros；配合 B=0 初态仍然 Δ=0
            self.E_local[safe] = E
            # LN + 小 MLP（逐维共享）：(r + d_h) -> r
            self.a_ln[safe] = nn.LayerNorm(rank + d_h)
            self.a_proj[safe] = nn.Linear(rank + d_h, rank)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        pooled_e: torch.Tensor,                 # [B, hidden]
        hidden_state: torch.Tensor = None       # [num_layers, B, hidden] (optional)
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            A_map: dict[name] -> [B, r, in_f]
            new_hidden_state: [num_layers, B, hidden]
        """
        Bsz = pooled_e.size(0)

        # ---------- 1) 全局 GRU 得到 gru_features，与 pooled_e 融合 ----------
        embeddings_seq = self.global_embeddings.weight                  # [num_codes, 768]
        gru_input_global = self.input_proj(embeddings_seq)              # [num_codes, hidden]
        gru_input_global = gru_input_global.unsqueeze(0).expand(Bsz, -1, -1)  # [B, num_codes, hidden]

        if hidden_state is None:
            hidden_state = torch.zeros(
                self.gru_layers, Bsz, self.hidden,
                device=pooled_e.device, dtype=pooled_e.dtype
            )
        gru_output, new_hidden_state = self.gru(gru_input_global, hidden_state)  # [B, num_codes, hidden]
        gru_features = gru_output[:, -1, :]                                       # [B, hidden]

        fused = self.context_fusion(torch.cat([pooled_e, gru_features], dim=-1))  # [B, hidden]

        # ---------- 2) 产生每层的 h^l（把 fused 复制为序列，并加上层ID嵌入） ----------
        num_layers = len(self.targets_meta)
        layer_tokens = fused.unsqueeze(1) + self.layer_id_embed.weight.unsqueeze(0)  # [B, L, hidden]
        h_seq, _ = self.layer_gru(layer_tokens)                                      # [B, L, d_h]

        # ---------- 3) 对每个目标层：A^l[i,:] = MLP(LN([E^l[i,:]; h^l])) ----------
        A_map: Dict[str, torch.Tensor] = {}
        for lidx, (name, in_f, _out_f) in enumerate(self.targets_meta):
            safe = self.name2safe[name]
            # h^l: [B, d_h] -> 扩展到 [B, in_f, d_h]
            h_l = h_seq[:, lidx, :]                                # [B, d_h]
            h_l_exp = h_l.unsqueeze(1).expand(-1, in_f, -1)        # [B, in_f, d_h]

            # 取本地 E^l，并扩展 batch 维度
            E = self.E_local[safe]                                 # [in_f, r]
            E_b = E.unsqueeze(0).expand(Bsz, -1, -1)               # [B, in_f, r]

            # 拼接 → LN → GeLU → 线性到 r
            z = torch.cat([E_b, h_l_exp], dim=-1)                  # [B, in_f, r + d_h]
            z = self.a_ln[safe](z)
            z = self.act(z)
            z = self.dropout(z)
            A_li = self.a_proj[safe](z)                            # [B, in_f, r]

            # 转成 [B, r, in_f]（与原 hook 期望一致）
            A_map[name] = A_li.transpose(1, 2).contiguous()        # [B, r, in_f]

        return A_map, new_hidden_state

    def get_B_by_name(self, name: str) -> torch.Tensor:
        return self.B[self.name2safe[name]]

    def reset_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.hidden, device=device)
    
def attach_hyperlora_hooks(targets: List[Tuple[str, nn.Linear]], rank: int):
    hooks = []
    for name, mod in targets:
        mod._hyper_state = None  # (A_any, B, alpha)；A_any: [r,in] 或 [B,r,in]
        mod._hyper_rank = rank

        def _make_hook(m):
            def _hook(m, inp, out):
                state = getattr(m, "_hyper_state", None)
                if state is None:
                    return out
                A_any, Bm, alpha = state
                x = inp[0]  # [B,S,in]
                if A_any.dim() == 2:         # [r,in] 共享
                    z = torch.einsum('bsi,ri->bsr', x, A_any)
                else:                         # [B,r,in] 批次特定
                    z = torch.einsum('bsi,bri->bsr', x, A_any)
                delta = torch.einsum('bsr,or->bso', z, Bm)  # [B,S,out]
                return out + delta * (alpha / float(m._hyper_rank))
            return _hook
        hooks.append(mod.register_forward_hook(_make_hook(mod)))
    return hooks


def clear_hyperlora_state(targets: List[Tuple[str, nn.Linear]]):
    for _, mod in targets:
        mod._hyper_state = None

# -------------------------------
# diff 工具
# -------------------------------

def _seq_mask_changed_by_lcs(ctx_ids: List[int], seq_ids: List[int]) -> torch.Tensor:
    """
    基于 LCS 的 token 级 diff 掩码（只标出 seq 中“非相同”的位置）
    返回 shape [len(seq_ids)] 的 bool mask
    """
    sm = difflib.SequenceMatcher(a=ctx_ids, b=seq_ids, autojunk=False)
    mask = torch.zeros(len(seq_ids), dtype=torch.bool)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            mask[j1:j2] = True
        # "delete" 是 ctx 独有，对 seq 不标记
    return mask

# -------------------------------
# 训练
# -------------------------------

@torch.no_grad()
def encode_prompts_with_st(sent_model: SentenceTransformer,
                           prompts: List[str],
                           device,
                           dtype,
                           sent_max_len: int):
    try:
        backbone_max = int(getattr(getattr(sent_model, "auto_model", None), "config", None).max_position_embeddings)
        if backbone_max is None or backbone_max <= 0:
            backbone_max = 512
    except Exception:
        backbone_max = 512

    window = min(int(sent_max_len), backbone_max)
    # 告诉 ST 截断到 window，避免再次越界
    try:
        sent_model.max_seq_length = window
    except Exception:
        pass

    tok = sent_model.tokenizer
    out_embs = []
    for text in prompts:
        # 先用底座 tokenizer 拆分成 token id，按 window 片段切块
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            # 空文本兜底
            chunk_texts = [""]
        else:
            chunk_texts = []
            for i in range(0, len(ids), window):
                # 直接 decode 回字符串交给 SentenceTransformer 重新编码
                # （这样最省事；一致的 tokenizer 再分词不会有语义问题）
                sub = ids[i:i+window]
                chunk_texts.append(tok.decode(sub, skip_special_tokens=True))

        # 对每个 chunk 单独编码，再做均值池化
        chunk_emb = sent_model.encode(
            chunk_texts,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        if chunk_emb.ndim == 1:
            pooled = chunk_emb
        else:
            pooled = chunk_emb.mean(dim=0)
        out_embs.append(pooled)

    emb = torch.stack(out_embs, dim=0).to(device=device, dtype=dtype)
    return emb


def train(args):
    eval_every = 2500   # 每训练 2500 个样本左右验证一次
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.txt")
    log_f = open(log_path, "a")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[加载LLM]", args.model)
    lm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    ).to(device)

    # 冻结基座
    for p in lm.parameters():
        p.requires_grad = False
    lm.eval()

    # NEW: 加载 Sentence-Transformer（冻结）
    print("[加载SentenceTransformer]", args.sent_model)
    st = SentenceTransformer(args.sent_model)
    for p in st.parameters():
        p.requires_grad = False
    st_dim = st.get_sentence_embedding_dimension()

    # 目标层与 Hook
    targets = find_target_linear_modules(lm)
    meta = [(name, m.in_features, m.out_features) for name, m in targets]
    print(f"[目标层] 命中线性层数量：{len(targets)}")
    for name, m in targets[:6]:
        print(f"  {name}: {m.in_features} -> {m.out_features}")
    if len(targets) > 6:
        print("  ...")
    hooks = attach_hyperlora_hooks(targets, rank=args.rank)

    hyper = GRUHyperLoRA(
        hidden_size=st_dim,
        targets_meta=meta,
        rank=args.rank,
        dropout=args.hyper_dropout,
        num_codes=args.num_codes,
        gru_layers=args.gru_layers
    ).to(device=device, dtype=lm.dtype)


    
    ds = EditDataset(args.train_files, tok, max_chunk_tokens=args.max_chunk_tokens, use_chunking=True)
    collate = Collator(tok, max_len=args.max_len)

    # === New: 随机划分 训练/验证 ===
    N = len(ds)
    val_size = min(args.val_size, N)  # 防止数据量不足
    indices = list(range(N))
    random.shuffle(indices)           # 受 set_seed(args.seed) 控制，可复现
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    ds_train = Subset(ds, train_indices)
    ds_val   = Subset(ds, val_indices)

    print(f"[数据] 训练样本数: {len(ds_train)} | 验证样本数: {len(ds_val)}")

    dl      = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=2, collate_fn=collate, pin_memory=True)
    dl_val  = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

    # 优化器
    opt = torch.optim.AdamW(hyper.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # AMP 设置：优先 bf16，否则 fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    running = 0.0
    running_diff = 0.0

    alpha_tensor = torch.tensor(args.alpha, device=device, dtype=lm.dtype)

    def _compute_diff_loss_for_batch(out_logits, labels, prompts_only, ctx_tok) -> torch.Tensor:
        """
        计算 batch 的 diff loss
        对齐方式：
        - gold label 的每个 token（位于 [p_len, p_len + T)）由 logits 的 [p_len-1, p_len-1 + T) 预测
        - 先从 logits 取出目标区间的预测分布（logits_tgt），再 argmax 得到 pred_ids_tgt
        - 用 ctx_ids 和 (pred_ids_tgt / gold_ids_tgt) 分别做 LCS 得到掩码
        - 依据 diff_mask_mode 选掩码，计算交叉熵（仅对掩码为 True 的位置）
        """
        B, S, V = out_logits.shape
        loss_list = []

        for i in range(B):
            # prompt 长度
            p_len = int(prompts_only.attention_mask[i].sum().item())
            # 目标长度（labels != -100 的数量）
            t_len = int((labels[i] != -100).sum().item())
            if t_len == 0 or p_len == 0 or p_len - 1 < 0:
                continue

            # 取目标对齐的 logits（注意 next-token 位移）
            start_logit = p_len - 1
            end_logit = start_logit + t_len
            if end_logit > S:
                end_logit = S
                t_len = end_logit - start_logit
                if t_len <= 0:
                    continue

            logits_tgt = out_logits[i, start_logit:end_logit, :]                       # [T, V]
            gold_ids_tgt = labels[i, p_len:p_len + t_len]                              # [T]
            # 取 context token（去掉 pad）
            ctx_att_len = int(ctx_tok.attention_mask[i].sum().item())
            ctx_ids = ctx_tok.input_ids[i, :ctx_att_len].tolist()                      # List[int]

            # 预测 token（逐位 argmax）
            pred_ids_tgt = torch.argmax(logits_tgt, dim=-1).detach().cpu().tolist()    # List[int]
            gold_ids_list = gold_ids_tgt.detach().cpu().tolist()                       # List[int]

            # 掩码（pred/gold/union/intersect）
            mask_pred = _seq_mask_changed_by_lcs(ctx_ids, pred_ids_tgt)                # [T] bool
            mask_gold = _seq_mask_changed_by_lcs(ctx_ids, gold_ids_list)               # [T] bool

            if args.diff_mask_mode == "pred":
                use_mask = mask_pred
            elif args.diff_mask_mode == "gold":
                use_mask = mask_gold
            elif args.diff_mask_mode == "union":
                use_mask = mask_pred | mask_gold
            elif args.diff_mask_mode == "intersect":
                use_mask = mask_pred & mask_gold
            elif args.diff_mask_mode == "align":
                    # 上下文 token 去重，放到同设备
                    if ctx_att_len == 0:
                        # 若没有上下文，跳过这条样本的 diff 对齐（也可让 diff_gold=1）
                        continue
                    ctx_ids_unique = torch.unique(
                        ctx_tok.input_ids[i, :ctx_att_len]
                    ).to(device=logits_tgt.device)

                    # 1) 预测侧：对每个位置，累加 softmax 概率在“上下文 token 集合”上的质量（可微）
                    prob = logits_tgt.softmax(dim=-1)                              # [T, V]
                    p_in_ctx = prob.index_select(dim=-1, index=ctx_ids_unique).sum(dim=-1)  # [T]
                    diff_pred = (1.0 - p_in_ctx).to(prob.dtype)                    # [T], 可微

                    # 2) gold 侧：gold 是否出现在上下文集合（常量，作监督）
                    gold_ids = gold_ids_tgt.long()                                 # [T]
                    # [T,1] == [1,K] -> [T,K]，对第二维 any 得到成员判定
                    in_ctx_gold = (gold_ids.unsqueeze(-1) == ctx_ids_unique.unsqueeze(0)).any(dim=-1)
                    diff_gold = (~in_ctx_gold).float().to(prob.dtype)              # [T], 常量监督

                    # 3) 对齐损失：让 diff_A(pred) ≈ diff_A(gold)
                    # 可选归一化以匹配你原来的 normalize 语义
                    loss_align = F.mse_loss(diff_pred, diff_gold, reduction="mean")
                    if args.diff_normalize:
                        loss_align = loss_align * (diff_pred.numel() / float(t_len))

                    loss_list.append(loss_align)
                    continue
                # ===========================================================

            else:
                use_mask = mask_pred

            # 至少需要一定数量的差异位置
            if use_mask.sum().item() < args.diff_min_tokens:
                continue

            # 仅对差异位置计算 CE
            # logits_tgt: [T, V], gold_ids_tgt: [T]
            idx = use_mask.nonzero(as_tuple=False).squeeze(-1).to(logits_tgt.device)
            sel_logits = logits_tgt.index_select(dim=0, index=idx)                     # [K, V]
            sel_gold = gold_ids_tgt.index_select(dim=0, index=idx).long()              # [K]

            ce = F.cross_entropy(sel_logits, sel_gold, reduction="mean")
            if args.diff_normalize:
                # 可选：按掩码比例归一，避免 K 大小时 diff loss 放大
                ce = ce * (idx.numel() / float(t_len))

            loss_list.append(ce)

        if not loss_list:
            return torch.zeros([], device=out_logits.device, dtype=out_logits.dtype)
        return torch.stack(loss_list).mean()

    def evaluate(dataloader):
        """返回 (avg_main, avg_diff, avg_total)；total = main + 4*diff"""
        hyper.eval()
        lm.eval()
        tot_main, tot_diff, steps = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                prompts_only = batch["prompts_only"]
                ctx_tok = batch["ctx_tok"]
                raw_prompts = batch["raw_prompts"]

                # 句向量编码（冻结）
                pooled = encode_prompts_with_st(
                    st, raw_prompts, device=device, dtype=lm.dtype, sent_max_len=args.sent_max_len
                )  # [B, st_dim]

                # 逐 batch 生成 A，并注入
                A_map, _ = hyper(pooled, hidden_state=None)
                for name, mod in targets:
                    A_batch = A_map[name]
                    Bm = hyper.get_B_by_name(name)
                    mod._hyper_state = (A_batch, Bm, alpha_tensor)

                # 前向：eval 不做累积与缩放
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    out = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss_main = out.loss
                    loss_diff = _compute_diff_loss_for_batch(out.logits, labels, prompts_only, ctx_tok)
                tot_main += float(loss_main.item())
                tot_diff += float(loss_diff.item()) if loss_diff.numel() > 0 else 0.0
                steps += 1

                clear_hyperlora_state(targets)

        avg_main = tot_main / max(1, steps)
        avg_diff = tot_diff / max(1, steps)
        avg_total = avg_main + 4.0 * avg_diff  # 与训练目标保持一致的权重
        return avg_main, avg_diff, avg_total
    
    best_val = float('inf')
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        hyper.train()
        lm.eval()
        for step, batch in enumerate(dl, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            # domains = batch["domains"].to(device, non_blocking=True)
            prompts_only = batch["prompts_only"]
            ctx_tok = batch["ctx_tok"]
            raw_prompts = batch["raw_prompts"]

            # 1) 句向量编码（冻结，不求梯度）
            with torch.no_grad():
                pooled = encode_prompts_with_st(
                    st, raw_prompts, device=device, dtype=lm.dtype, sent_max_len=args.sent_max_len
                )  # [B, st_dim]

            # 2) GRU 超网按 batch 生成 A；这里不跨 batch 维持 GRU 状态，直接置 None 即可
            A_map, _ = hyper(pooled, hidden_state=None)  # dict[name] -> [B, r, in], new_hidden

            # 3) 注入 (A_batch, B, alpha)
            for name, mod in targets:
                A_batch = A_map[name]
                Bm = hyper.get_B_by_name(name)
                mod._hyper_state = (A_batch, Bm, alpha_tensor)


            # 若全部为 -100，跳过
            if (labels != -100).sum().item() == 0:
                clear_hyperlora_state(targets)
                continue

            # 4) 前向
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # 原始 CE（已按 accum 分摊）
                loss_main = out.loss / max(1, args.accum)
                # 结构化 diff loss（不分摊，后面总损失再相加）
                loss_diff = _compute_diff_loss_for_batch(out.logits, labels, prompts_only, ctx_tok) / max(1, args.accum)

                loss_total = loss_main + 4 * loss_diff

            # 反向：bf16 不用 GradScaler，fp16 才用
            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            running += float(loss_main.item()) * args.accum
            running_diff += float(loss_diff.item()) * args.accum if loss_diff.numel() > 0 else 0.0

            if step % args.accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)
                global_step += 1
                clear_hyperlora_state(targets)

                if global_step % args.log_every == 0:
                    avg_main = running / args.log_every
                    avg_diff = running_diff / args.log_every
                    avg_total = avg_main + 4 * avg_diff
                    logline = f"[E{epoch} G{global_step}] train_loss={avg_main:.4f}  diff_loss={avg_diff:.4f}  loss_total={avg_total:.4f}"
                    print(logline)
                    log_f.write(logline + "\n")
                    log_f.flush()
                    running = 0.0
                    running_diff = 0.0

                    if global_step * args.batch_size * args.accum >= eval_every:
                        val_main, val_diff, val_total = evaluate(dl_val)
                        val_log = f"[E{epoch} G{global_step}] val_main={val_main:.4f}  val_diff={val_diff:.4f}  val_total={val_total:.4f}"
                        print(val_log)
                        log_f.write(val_log + "\n")
                        log_f.flush()

                        if val_total < best_val:
                            best_val = val_total
                            best_epoch = epoch
                            best_path = os.path.join(args.out_dir, args.best_ckpt_name)
                            torch.save({
                                "hyper": hyper.state_dict(),
                                "config": vars(args),
                                "model_name": args.model,
                                "targets": meta,
                                "sent_model": args.sent_model,
                                "val_loss_total": float(best_val),
                                "epoch": int(best_epoch),
                            }, best_path)
                            print(f"[保存-BEST] {best_path}  (epoch={best_epoch}, val_total={best_val:.4f})")

                        eval_every += 2500  # 下次触发点
                        hyper.train() # 切回训练模式


        # 每个 epoch 保存一次
        ckpt_path = os.path.join(args.out_dir, f"epoch{epoch}.pt")
        torch.save({
            "hyper": hyper.state_dict(),
            "config": vars(args),
            "model_name": args.model,
            "targets": meta,
            "sent_model": args.sent_model,
        }, ckpt_path)
        print(f"[保存] {ckpt_path}")

    # 移除 hooks
    for h in hooks: h.remove()

# -------------------------------
# CLI
# -------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_files", type=str, nargs="*", default=[
        "/home/ieb24002/LLM_Code/data/sql_benchmark_train.json",
        "/home/ieb24002/LLM_Code/data/wiki_benchmark_train.json",
        "/home/ieb24002/LLM_Code/data/code_benchmark_train.json",
        "/home/ieb24002/LLM_Code/data/latex_benchmark_train.json",
    ])
    ap.add_argument("--out_dir", type=str, default="./light_qwen_2.5_2e-5_pre")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_chunk_tokens", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--hyper_dropout", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_codes", type=int, default=16)     # GRU 输入序列长度（代码本 token 数）
    ap.add_argument("--gru_layers", type=int, default=2)     # GRU 层数

    # NEW: Sentence-Transformer
    ap.add_argument("--sent_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--sent_max_len", type=int, default=2048)
    # parse_args() 内
    ap.add_argument("--diff_mask_mode", type=str, default="pred",
                choices=["pred", "gold", "union", "intersect", "align"],
                help="diff 掩码/对齐：pred/gold/union/intersect/align(可微对齐)")

    ap.add_argument("--diff_min_tokens", type=int, default=1, help="至少多少个差异 token 才计算 diff loss")
    ap.add_argument("--diff_normalize", action="store_true", help="是否按掩码比例归一化 diff CE")
    ap.add_argument("--val_size", type=int, default=500, help="随机划分的验证集大小（上限为总样本数）")
    ap.add_argument("--best_ckpt_name", type=str, default="best.pt", help="验证集最优模型的保存文件名")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)