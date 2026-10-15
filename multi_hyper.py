import os
import re
import json
import time
import argparse
import torch
import torch.nn as nn
from typing import List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as hf_logging
)
from sentence_transformers import SentenceTransformer

###########################################################
# 0) Logging
###########################################################
def log_info(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

###########################################################
# 1) Chunking Utilities
###########################################################
def split_into_paragraphs(text: str) -> List[str]:
    """按双换行切分段落。"""
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]

def split_by_single_newline(paragraph: str) -> List[str]:
    """段落过长时按单换行进一步切分。"""
    return [ln.strip() for ln in paragraph.split("\n") if ln.strip()]

def chunk_by_tokens(paragraphs: List[str], tokenizer, max_tokens=1024) -> List[str]:
    """
    按 token 数量组合段落为块（超长段落退化为逐行拼接），以适配生成时上下文窗口。
    """
    chunks, current_tokens, current_count = [], [], 0
    for paragraph in paragraphs:
        p_ids = tokenizer(paragraph, add_special_tokens=False)["input_ids"]
        p_len = len(p_ids)
        if p_len > max_tokens:
            # 退化：按单行切
            for line in split_by_single_newline(paragraph):
                l_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
                l_len = len(l_ids)
                if l_len > max_tokens:
                    # 单行仍超长：直接作为一个块
                    if current_tokens:
                        chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
                        current_tokens, current_count = [], 0
                    chunks.append(tokenizer.decode(l_ids, skip_special_tokens=True).strip())
                    continue
                if current_count + l_len <= max_tokens:
                    current_tokens.extend(l_ids); current_count += l_len
                else:
                    chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
                    current_tokens, current_count = l_ids, l_len
        else:
            if current_count + p_len <= max_tokens:
                current_tokens.extend(p_ids); current_count += p_len
            else:
                chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
                current_tokens, current_count = p_ids, p_len
    if current_tokens:
        chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
    return chunks

###########################################################
# 2) HyperLoRA 目标层搜集与定义（与训练同构）
###########################################################
TARGET_PATTERNS = [
    re.compile(r".*q_proj$"),
    re.compile(r".*k_proj$"),
    re.compile(r".*v_proj$"),
    re.compile(r".*o_proj$"),
]

def find_target_linear_modules(model: nn.Module):
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
    return re.sub(r"[^0-9a-zA-Z_]", "_", name.replace(".", "__"))

class GRUHyperLoRA(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        targets_meta,
        rank: int = 16,
        dropout: float = 0.1,
        num_codes: int = 16,
        gru_layers: int = 3,
        d_h: int = 128,
        global_dim: int = 512,
    ):
        super().__init__()
        self.rank = rank
        self.hidden = hidden_size
        self.targets_meta = targets_meta
        self.num_codes = num_codes
        self.gru_layers = gru_layers

        self.name2safe = {}
        self.safe2name = {}
        self.B = nn.ParameterDict()
        self.target_dims = {}

        # -------- 初始化 B（与训练版完全一致） --------
        for name, in_f, out_f in targets_meta:
            safe = _sanitize(name)
            self.name2safe[name] = safe
            self.safe2name[safe] = name
            self.target_dims[name] = (in_f, out_f)
            B = nn.Parameter(torch.zeros(out_f, rank))
            self.B[safe] = B

        # -------- 全局 embedding + GRU --------
        self.global_embeddings = nn.Embedding(num_codes, global_dim)
        nn.init.normal_(self.global_embeddings.weight, mean=0.0, std=0.02)

        self.input_proj = nn.Linear(global_dim, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.context_fusion = nn.Linear(hidden_size + hidden_size, hidden_size)

        # -------- 层级 GRU + E_local + A MLP --------
        self.layer_id_embed = nn.Embedding(len(targets_meta), hidden_size)
        nn.init.normal_(self.layer_id_embed.weight, mean=0.0, std=0.02)

        self.layer_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=d_h,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.E_local = nn.ParameterDict()
        self.a_ln = nn.ModuleDict()
        self.a_proj = nn.ModuleDict()

        for name, in_f, _out_f in targets_meta:
            safe = self.name2safe[name]
            E = nn.Parameter(torch.empty(in_f, rank))
            nn.init.xavier_normal_(E)
            self.E_local[safe] = E

            self.a_ln[safe] = nn.LayerNorm(rank + d_h)
            self.a_proj[safe] = nn.Linear(rank + d_h, rank)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled_e, hidden_state=None):
        B = pooled_e.size(0)

        # 全局 GRU 生成语义条件
        seq = self.global_embeddings.weight
        seq = self.input_proj(seq)
        seq = seq.unsqueeze(0).expand(B, -1, -1)

        if hidden_state is None:
            hidden_state = torch.zeros(
                self.gru_layers, B, self.hidden, device=pooled_e.device, dtype=pooled_e.dtype
            )

        out, new_hidden = self.gru(seq, hidden_state)
        g = out[:, -1, :]  # [B, hidden]

        fused = self.context_fusion(torch.cat([pooled_e, g], dim=-1))

        # 层级 GRU
        L = len(self.targets_meta)
        layer_tokens = fused.unsqueeze(1) + self.layer_id_embed.weight.unsqueeze(0)
        h_seq, _ = self.layer_gru(layer_tokens)  # [B, L, d_h]

        A_map = {}
        for idx, (name, in_f, _out_f) in enumerate(self.targets_meta):
            safe = self.name2safe[name]

            E_b = self.E_local[safe].unsqueeze(0).expand(B, -1, -1)
            h_l = h_seq[:, idx, :].unsqueeze(1).expand(-1, in_f, -1)

            z = torch.cat([E_b, h_l], dim=-1)
            z = self.a_ln[safe](z)
            z = self.act(z)
            z = self.dropout(z)

            A_li = self.a_proj[safe](z)  # [B, in_f, r]
            A_map[name] = A_li.transpose(1, 2).contiguous()

        return A_map, new_hidden

    def get_B_by_name(self, name):
        safe = self.name2safe[name]
        return self.B[safe]

    def reset_hidden_state(self, batch_size, device):
        return torch.zeros(self.gru_layers, batch_size, self.hidden, device=device)


def attach_hyperlora_hooks(targets: List[Tuple[str, nn.Linear]], rank: int):
    """
    将 (A, B, alpha) 注入目标层输出：y = Wx + (A @ x) @ B^T * (alpha/rank)
    """
    hooks = []
    for name, mod in targets:
        mod._hyper_state = None
        mod._hyper_rank = rank

        def _make_hook(m):
            def _hook(m, inp, out):
                state = getattr(m, "_hyper_state", None)
                if state is None:
                    return out
                A_any, Bm, alpha = state
                x = inp[0]  # [B,S,in]
                if A_any.dim() == 2:         # [r,in] 共享（一般不用）
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

###########################################################
# 3) Sentence-Transformer 编码（安全分块 + 均值池化）
###########################################################
@torch.no_grad()
def encode_prompts_with_st(sent_model: SentenceTransformer,
                           prompts: List[str],
                           device,
                           dtype,
                           sent_max_len: int):
    """
    - window = min(sent_max_len, backbone_max_pos)
    - 用 ST 自身 tokenizer 分块，再逐块 encode 并做均值池化
    """
    try:
        backbone_max = int(getattr(getattr(sent_model, "auto_model", None), "config", None).max_position_embeddings)
        if backbone_max is None or backbone_max <= 0:
            backbone_max = 512
    except Exception:
        backbone_max = 512

    window = min(int(sent_max_len), backbone_max)

    try:
        sent_model.max_seq_length = window
    except Exception:
        pass

    tok = sent_model.tokenizer
    out_embs = []
    for text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            chunk_texts = [""]
        else:
            chunk_texts = [tok.decode(ids[i:i+window], skip_special_tokens=True)
                           for i in range(0, len(ids), window)]
        emb_chunks = sent_model.encode(
            chunk_texts, convert_to_tensor=True, device=device, show_progress_bar=False
        )
        pooled = emb_chunks if emb_chunks.ndim == 1 else emb_chunks.mean(dim=0)
        out_embs.append(pooled)

    emb = torch.stack(out_embs, dim=0).to(device=device, dtype=dtype)
    return emb

###########################################################
# 4) 加载 ckpt，准备 base + ST + HyperLoRA + hooks
###########################################################
def load_hyper_from_pt_st(ckpt_path: str, device: torch.device):
    """
    ckpt 结构约定：
      {
        "model_name": str,                       # HF 基座名
        "targets": [(name, in_f, out_f), ...],   # 目标层元信息
        "hyper": state_dict,                     # GRUHyperLoRA 权重
        "config": { "rank": int, "alpha": float, "hyper_dropout": float, ... },
        "sent_model": str,                       # ST 名称
      }
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    meta = ckpt["targets"]
    cfg = ckpt.get("config", {})
    rank = int(cfg.get("rank", 8))
    dropout = float(cfg.get("hyper_dropout", 0.1))
    alpha = float(cfg.get("alpha", 32.0))
    num_codes = int(cfg.get("num_codes", 16))
    gru_layers = int(cfg.get("gru_layers", 2))
    sent_model_name = ckpt.get("sent_model", "sentence-transformers/all-MiniLM-L6-v2")
    sent_max_len = int(cfg.get("sent_max_len", 2048))

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # baseline 风格：若无 pad_token_id，则设为 eos
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # baseline 风格 dtype：CUDA 用 bfloat16，否则 float32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    log_info(f"Loading base model: {model_name}  (dtype={dtype})")
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    lm.to(device)
    lm.eval()
    lm.config.use_cache = True
    if getattr(lm.config, "pad_token_id", None) is None:
        lm.config.pad_token_id = lm.config.eos_token_id

    # 校验目标层一致性
    targets = find_target_linear_modules(lm)
    found_names = {n for n, _ in targets}
    saved_names = {t[0] for t in meta}
    if found_names != saved_names:
        missing = sorted(list(saved_names - found_names))
        extra   = sorted(list(found_names - saved_names))
        raise RuntimeError(
            "Target layer names mismatch between base model and checkpoint.\n"
            f"Missing in current model: {missing}\n"
            f"Extra in current model:   {extra}\n"
            "Please ensure you are loading the SAME base model as training."
        )

    # Sentence-Transformer（冻结）
    log_info(f"Loading Sentence-Transformer: {sent_model_name}")
    st = SentenceTransformer(sent_model_name)
    for p in st.parameters():
        p.requires_grad = False
    st_dim = st.get_sentence_embedding_dimension()

    # 实例化与训练同构的 GRUHyperLoRA，并严格加载
    hyper = GRUHyperLoRA(
        hidden_size=st_dim,
        targets_meta=meta,
        rank=rank,
        dropout=dropout,
        num_codes=num_codes,
        gru_layers=gru_layers,
        global_dim=ckpt["hyper"]["global_embeddings.weight"].shape[1],
    ).to(device=device, dtype=dtype)
    hyper.load_state_dict(ckpt["hyper"], strict=True)
    hyper.eval()

    # 注册 hook
    hooks = attach_hyperlora_hooks(targets, rank=rank)

    return tok, lm, st, hyper, targets, hooks, dtype, alpha, sent_max_len

###########################################################
# 5) 单轮编辑（分块）—— 使用 ST 池化 + HyperLoRA 注入（baseline 采样参数）
###########################################################
@torch.no_grad()
def generate_edited_text_st(
    context: str,
    edit_request: str,
    lm,
    tok,
    st: SentenceTransformer,
    hyper: GRUHyperLoRA,
    targets: List[Tuple[str, nn.Linear]],
    alpha: float,
    sent_max_len: int = 2048,
    max_chunk_tokens: int = 1024,     # ← 默认 1024
    max_new_tokens: int = 1024,       # ← 固定使用传入值
    prompt_encoder=None,
):
    device = next(lm.parameters()).device
    dtype  = next(lm.parameters()).dtype

    paragraphs = split_into_paragraphs(context)
    context_chunks = chunk_by_tokens(paragraphs, tok, max_tokens=max_chunk_tokens)
    outputs = []

    for chunk_text in context_chunks:
        prompt = (
            f"### Edit Request:\n{edit_request}\n\n"
            f"### Original Text:\n{chunk_text}\n\n"
            f"### Edited Content:\n"
        )

        # 1) 用 ST 对 prompt 编码（安全分块 + 均值池化）
        encoder = prompt_encoder or encode_prompts_with_st
        pooled = encoder(
            st, [prompt], device=device, dtype=dtype, sent_max_len=sent_max_len
        )  # [1, st_dim]

        # 2) 超网生成各层 A，并注入 (A_batch, B, alpha)
        A_map, _ = hyper(pooled, hidden_state=None)  # dict[name] -> [B, r, in]
        alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
        for name, mod in targets:
            A_batch = A_map[name]                 # [1, r, in]
            Bm = hyper.get_B_by_name(name)        # [out, r]
            mod._hyper_state = (A_batch, Bm, alpha_t)

        # 3) 编码完整提示并生成 —— baseline 风格：do_sample=False，固定 max_new_tokens
        try:
            enc = tok(prompt, return_tensors="pt").to(device)
            gen_ids = lm.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            # Decode only newly generated tokens; output may itself contain the marker.
            new_ids = gen_ids[0, enc["input_ids"].shape[1]:]
            outputs.append(tok.decode(new_ids, skip_special_tokens=True).strip())
        finally:
            # Clear adapter state even if generation fails.
            clear_hyperlora_state(targets)

    return "\n\n".join(outputs).strip()

@torch.no_grad()
def generate_edited_text_batch_st(
    contexts, edit_requests, lm, tok, st, hyper, targets, alpha,
    sent_max_len=2048, max_chunk_tokens=2000, max_new_tokens=2000,
    prompt_encoder=None,
):
    """Generate one edit per record, with independent adapters in each batch row.

    Chunk round k batches the kth chunk of every record that still has one.
    Chunking and output joining match generate_edited_text_st.
    """
    if len(contexts) != len(edit_requests):
        raise ValueError("contexts and edit_requests must have equal lengths")
    if max_chunk_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("Token budgets must be positive")
    device = next(lm.parameters()).device
    dtype = next(lm.parameters()).dtype
    chunks = [chunk_by_tokens(split_into_paragraphs(text), tok,
                              max_tokens=max_chunk_tokens) for text in contexts]
    outputs = [[] for _ in contexts]
    encoder = prompt_encoder or encode_prompts_with_st
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for chunk_index in range(max(map(len, chunks), default=0)):
            active = [i for i, parts in enumerate(chunks) if chunk_index < len(parts)]
            prompts = [
                f"### Edit Request:\n{edit_requests[i]}\n\n"
                f"### Original Text:\n{chunks[i][chunk_index]}\n\n"
                f"### Edited Content:\n" for i in active
            ]
            pooled = encoder(st, prompts, device=device, dtype=dtype,
                             sent_max_len=sent_max_len)
            A_map, _ = hyper(pooled, hidden_state=None)
            alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
            try:
                for name, mod in targets:
                    mod._hyper_state = (A_map[name], hyper.get_B_by_name(name), alpha_t)
                enc = tok(prompts, padding=True, return_tensors="pt").to(device)
                gen_ids = lm.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    num_beams=1, num_return_sequences=1,
                    pad_token_id=tok.pad_token_id,
                )
                decoded = tok.batch_decode(
                    gen_ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                for i, text in zip(active, decoded):
                    outputs[i].append(text.strip())
            finally:
                clear_hyperlora_state(targets)
    finally:
        tok.padding_side = original_padding_side
        clear_hyperlora_state(targets)
    return ["\n\n".join(parts).strip() for parts in outputs]


###########################################################
# 6) 多轮编辑：按顺序迭代应用请求（核心）
###########################################################
def parse_edit_requests(edit_request_str: str) -> List[str]:
    """
    支持两种格式：
    1) 显式标签：
       <Edit Request 1> ... </Edit Request 1>
       <Edit Request 2> ... </Edit Request 2>
    2) 无标签：按行拆分，每行为一个请求
    返回：每个请求外包一层 <Edit Request>...</Edit Request>，以保证提示格式一致。
    """
    pattern = r"<Edit Request\s*\d*>(.*?)</Edit Request\s*\d*>"
    requests = re.findall(pattern, edit_request_str, flags=re.DOTALL)
    if not requests:
        lines = [ln.strip() for ln in edit_request_str.split("\n") if ln.strip()]
        requests = [f"<Edit Request>\n{line}\n</Edit Request>" for line in lines]
    else:
        new_requests = []
        for r in requests:
            r = r.strip()
            new_requests.append(f"<Edit Request>\n{r}\n</Edit Request>")
        requests = new_requests
    return requests

@torch.no_grad()
def generate_edited_text_multiturn_st(
    initial_context: str,
    edit_requests_str: str,
    lm,
    tok,
    st: SentenceTransformer,
    hyper: GRUHyperLoRA,
    targets: List[Tuple[str, nn.Linear]],
    alpha: float,
    sent_max_len: int = 2048,
    max_chunk_tokens: int = 2000,     # ← 默认 1024
    max_new_tokens: int = 2000,       # ← 固定
) -> str:
    """
    多轮编辑：将 edit_requests 中的每个请求依次作用于文本。
    """
    requests = parse_edit_requests(edit_requests_str)
    updated = initial_context
    for i, req in enumerate(requests, 1):
        log_info(f"Applying edit request #{i}/{len(requests)}")
        updated = generate_edited_text_st(
            context=updated,
            edit_request=req,
            lm=lm,
            tok=tok,
            st=st,
            hyper=hyper,
            targets=targets,
            alpha=alpha,
            sent_max_len=sent_max_len,
            max_chunk_tokens=max_chunk_tokens,
            max_new_tokens=max_new_tokens,
        )
    return updated

###########################################################
# 7) 主流程：读取输入 JSON，执行多轮编辑，写出结果（无 BLEU）
###########################################################
from typing import Optional

def run_infer(ckpt_path: str,
              input_files: List[str],
              output_dir: str,
              sent_max_len: int,
              max_chunk_tokens: int,
              max_new_tokens: int):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok, lm, st, hyper, targets, hooks, dtype, alpha, st_sent_max_len = load_hyper_from_pt_st(ckpt_path, device)
    # 允许命令行覆盖 ckpt 内的 sent_max_len
    if sent_max_len is None:
        sent_max_len = st_sent_max_len

    log_info("Base model + ST + HyperLoRA loaded. Start MULTI-TURN editing ...")

    for input_path in input_files:
        if not os.path.exists(input_path):
            log_info(f"Input file not found, skip: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        # ---------- baseline-style caps (match by filename) ----------
        # caps = {
        #     "wiki_benchmark_multiturn.json": 400,
        #     "code_benchmark_multiturn.json": 400,
        #     "latex_benchmark_multiturn.json": 400,
        #     "sql_benchmark_multiturn.json": 100,
        # }
        # # fname = os.path.basename(input_path)
        # # limit = caps.get(fname)
        # # if limit is None:
        # #     cap = caps.get(fname)
        # # elif limit == 0:
        # #     cap = None
        # # else:
        # #     cap = limit
        # fname = os.path.basename(input_path)
        # default_cap = caps.get(fname)   # baseline 的按文件名 cap
        # # 应用 CLI 传入的 override（None=使用默认cap；0=关闭cap；其他正数=统一cap）
        # if limit is None:
        #     cap = default_cap
        # elif limit == 0:
        #     cap = None
        # else:
        #     cap = limit
        # if cap is not None:
        #     data = data[:cap]
        #     log_info(f"Apply cap={cap} for {fname}")
        # ---------- baseline-style caps (fixed) ----------
        caps = {
            "wiki_benchmark_multiturn.json": 400,
            "code_benchmark_multiturn.json": 400,
            "latex_benchmark_multiturn.json": 400,
            "sql_benchmark_multiturn.json": 100,
        }
        fname = os.path.basename(input_path)
        cap = caps.get(fname)
        if cap is not None:
            data = data[:cap]
            log_info(f"Apply fixed cap={cap} for {fname}")

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_name = base_name.replace("_eval", "_generate").replace("_multiturn", "_multiturn_generate")
        out_path = os.path.join(output_dir, f"{out_name}_model_output.json")

        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write("[\n")
            first = True

            for rec in data:
                sample_id   = rec.get("id")
                context     = rec.get("context", "")
                edit_req    = rec.get("edit_request", "")

                gen_text = generate_edited_text_multiturn_st(
                    initial_context=context,
                    edit_requests_str=edit_req,
                    lm=lm,
                    tok=tok,
                    st=st,
                    hyper=hyper,
                    targets=targets,
                    alpha=alpha,
                    sent_max_len=sent_max_len,
                    max_chunk_tokens=max_chunk_tokens,
                    max_new_tokens=max_new_tokens,
                )

                item = {
                    "id": sample_id,
                    "final_generated_edit": gen_text
                }
                if not first:
                    fout.write(",\n")
                else:
                    first = False
                fout.write(json.dumps(item, ensure_ascii=False, indent=2))

            fout.write("\n]\n")

        log_info(f"Finished: {input_path} -> {out_path}")

    # 移除 hooks（可选）
    for h in hooks:
        h.remove()
    log_info("All files processed. Exiting.")

###########################################################
# 8) CLI
###########################################################
def build_parser():
    p = argparse.ArgumentParser(description="Multi-turn editing inference (no BLEU) - baseline-like settings")
    p.add_argument("--ckpt", required=True, type=str, help="Path to HyperLoRA checkpoint (.pt)")
    p.add_argument("--inputs", required=True, type=str, nargs="+",
                   help="Input JSON files. Each record must contain: id, context, edit_request")
    p.add_argument("--out_dir", default="HyperLoRA_multiturn_outputs", type=str, help="Output directory")
    p.add_argument("--sent_max_len", default=None, type=int, help="Sentence-Transformer max tokens per chunk (override ckpt)")
    p.add_argument("--max_chunk_tokens", default=2000, type=int, help="Context chunk size in tokens for LM")  
    p.add_argument("--max_new_tokens", default=2000, type=int, help="Max new tokens per chunk for generation") 
    return p

import glob

def main():
    args = build_parser().parse_args()

    # 如果传入的是文件夹，就展开其中所有 .json 文件
    expanded_inputs = []
    for path in args.inputs:
        if os.path.isdir(path):
            expanded_inputs.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        else:
            expanded_inputs.append(path)

    if not expanded_inputs:
        raise ValueError("No JSON input files found!")

    run_infer(
        ckpt_path=args.ckpt,
        input_files=expanded_inputs,
        output_dir=args.out_dir,
        sent_max_len=args.sent_max_len,
        max_chunk_tokens=args.max_chunk_tokens,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
