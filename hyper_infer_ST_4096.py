import os
import re
import json
import time
import math
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any, Tuple
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as hf_logging
)

# NEW: Sentence-Transformer
from sentence_transformers import SentenceTransformer

###########################################################
# 0) Logging & NLTK
###########################################################
def log_info(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

hf_logging.set_verbosity_error()
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # 有些环境需要
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = True

###########################################################
# 1) Utilities: chunking（与训练/脚本1同风格）
###########################################################
def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]

def split_by_single_newline(paragraph: str) -> List[str]:
    return [ln.strip() for ln in paragraph.split("\n") if ln.strip()]

def chunk_by_tokens(paragraphs: List[str], tokenizer, max_tokens=1024) -> List[str]:
    """按 token 数量组合段落为块（超长段落退化为逐行拼接）"""
    chunks, current_tokens, current_count = [], [], 0
    for paragraph in paragraphs:
        p_ids = tokenizer(paragraph, add_special_tokens=False)["input_ids"]
        if len(p_ids) > max_tokens:
            for line in split_by_single_newline(paragraph):
                l_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
                l_len = len(l_ids)
                if l_len > max_tokens:
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
            p_len = len(p_ids)
            if current_count + p_len <= max_tokens:
                current_tokens.extend(p_ids); current_count += p_len
            else:
                chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
                current_tokens, current_count = p_ids, p_len
    if current_tokens:
        chunks.append(tokenizer.decode(current_tokens, skip_special_tokens=True).strip())
    return chunks

###########################################################
# 2) HyperLoRA（与训练脚本2一致）
###########################################################
TARGET_PATTERNS = [
    re.compile(r".*q_proj$"),
    re.compile(r".*k_proj$"),
    re.compile(r".*v_proj$"),
    re.compile(r".*o_proj$"),
]

def find_target_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if any(pat.match(name) for pat in TARGET_PATTERNS):
                targets.append((name, mod))
    targets.sort(key=lambda x: x[0])
    return targets

# ===== ADD: use the same GRUHyperLoRA as in training =====
def _sanitize(name: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_]', '_', name.replace('.', '__'))

class GRUHyperLoRA(nn.Module):
    def __init__(self, hidden_size: int, targets_meta, rank: int = 8,
                 dropout: float = 0.1, num_codes: int = 16, gru_layers: int = 2):
        super().__init__()
        self.rank = rank
        self.hidden = hidden_size
        self.targets_meta = targets_meta
        self.num_codes = num_codes
        self.gru_layers = gru_layers

        self.name2safe, self.safe2name = {}, {}
        self.B = nn.ParameterDict()
        self.target_dims = {}

        for name, in_f, out_f in targets_meta:
            safe = _sanitize(name)
            while safe in self.B:
                safe = safe + "_1"
            self.name2safe[name] = safe
            self.safe2name[safe] = name
            self.target_dims[name] = (in_f, out_f)

            B = nn.Parameter(torch.empty(out_f, rank))
            nn.init.xavier_uniform_(B)
            self.B[safe] = B

        self.global_embeddings = nn.Embedding(num_codes, 512)
        nn.init.normal_(self.global_embeddings.weight, mean=0.0, std=0.02)

        self.input_proj = nn.Linear(512, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        self.context_fusion = nn.Linear(hidden_size + hidden_size, hidden_size)

        self.a_generators = nn.ModuleDict()
        for name, in_f, _ in targets_meta:
            safe = self.name2safe[name]
            self.a_generators[safe] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, rank * in_f),
                nn.Tanh()
            )

    def forward(self, pooled_e: torch.Tensor,
                hidden_state: torch.Tensor = None):
        B = pooled_e.size(0)
        embeddings_seq = self.global_embeddings.weight
        gru_input = self.input_proj(embeddings_seq).unsqueeze(0).expand(B, -1, -1)

        if hidden_state is None:
            hidden_state = torch.zeros(
                self.gru_layers, B, self.hidden,
                device=pooled_e.device, dtype=pooled_e.dtype
            )

        gru_output, new_hidden_state = self.gru(gru_input, hidden_state)
        gru_features = gru_output[:, -1, :]
        fused = self.context_fusion(torch.cat([pooled_e, gru_features], dim=-1))

        A_map = {}
        for name, in_f, _ in self.targets_meta:
            safe = self.name2safe[name]
            A_flat = self.a_generators[safe](fused)          # [B, r*in_f]
            A_map[name] = A_flat.view(B, self.rank, in_f)    # [B, r, in_f]
        return A_map, new_hidden_state

    def get_B_by_name(self, name: str) -> torch.Tensor:
        return self.B[self.name2safe[name]]

    def reset_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.hidden, device=device)

# ===== REPLACE attach_hyperlora_hooks IN INFERENCE =====
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


###########################################################
# 3) NEW: 安全的 ST 分块编码（训练/推理同分布）
###########################################################
@torch.no_grad()
def encode_prompts_with_st(sent_model: SentenceTransformer,
                           prompts: List[str],
                           device,
                           dtype,
                           sent_max_len: int):
    """
    - 确保窗口大小 window = min(sent_max_len, backbone_max_pos)
    - 先按 ST 的 tokenizer 切 token，再 decode 回子串逐块 encode
    - 对块向量做均值池化，得到整条提示的句向量
    """
    try:
        backbone_max = int(getattr(getattr(sent_model, "auto_model", None), "config", None).max_position_embeddings)
        if backbone_max is None or backbone_max <= 0:
            backbone_max = 4096
    except Exception:
        backbone_max = 4096

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
# 4) 载入 ckpt（读取 sent_model & sent_max_len），准备 base + hyper + hooks
###########################################################
# ===== REPLACE load_hyper_from_pt_st IN INFERENCE =====
def load_hyper_from_pt_st(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    meta = ckpt["targets"]                      # [(name, in_f, out_f), ...]
    cfg = ckpt.get("config", {})
    rank = int(cfg.get("rank", 16))
    dropout = float(cfg.get("hyper_dropout", 0.1))
    alpha = float(cfg.get("alpha", 32.0))
    num_codes = int(cfg.get("num_codes", 24))
    gru_layers = int(cfg.get("gru_layers", 3))
    sent_model_name = ckpt.get("sent_model", "Qwen/Qwen3-Embedding-0.6B")
    sent_max_len = int(cfg.get("sent_max_len", 4096))

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

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
        gru_layers=gru_layers
    ).to(device=device, dtype=dtype)
    hyper.load_state_dict(ckpt["hyper"], strict=False)
    hyper.eval()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HyperLoRA checkpoint loaded successfully.")

    # 注册 3 元组 hook
    hooks = attach_hyperlora_hooks(targets, rank=rank)

    return tok, lm, st, hyper, targets, hooks, dtype, alpha, sent_max_len

###########################################################
# 5) 生成单条（分块）编辑文本 —— 与脚本1格式一致，但池化来源改为 ST
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
    sent_max_len: int = 4096,
    max_chunk_tokens: int = 2000,
    max_new_tokens: int = 2000,
):
    device = next(lm.parameters()).device
    dtype  = next(lm.parameters()).dtype

    # paragraphs = split_into_paragraphs(context)
    # context_chunks = chunk_by_tokens(paragraphs, tok, max_tokens=max_chunk_tokens)

    # 动态判断是否需要分块：优先整篇生成
    ctx_ids = tok.encode(context, add_special_tokens=False)
    if len(ctx_ids) <= 30000:  # 留余量，防止超过 32k 上下文
        context_chunks = [context]
    else:
        # 超长文本 fallback 分块
        paragraphs = split_into_paragraphs(context)
        context_chunks = chunk_by_tokens(paragraphs, tok, max_tokens=16000)
    # ===== 修改结束 =====

    outputs = []

    for chunk_text in context_chunks:
        prompt = (
            f"### Edit Request:\n{edit_request}\n\n"
            f"### Original Text:\n{chunk_text}\n\n"
            f"### Edited Content:\n"
        )

        # NEW: 用 ST 对 prompt 编码（安全分块 + 均值池化）
        pooled = encode_prompts_with_st(
            st, [prompt], device=device, dtype=dtype, sent_max_len=sent_max_len
        )  # [1, st_dim]

        # ===== inside generate_edited_text_st, after pooled is computed =====
        # 超网按 batch 生成 A；与训练一致，不做跨 batch 累积
        A_map, _ = hyper(pooled, hidden_state=None)  # dict[name] -> [B, r, in]
        alpha_t = torch.tensor(alpha, device=device, dtype=dtype)

        # 注入 (A_batch, B, alpha) —— 三元组
        for name, mod in targets:
            A_batch = A_map[name]                 # [1, r, in]
            Bm = hyper.get_B_by_name(name)        # [out, r]
            mod._hyper_state = (A_batch, Bm, alpha_t)

        # # 超网络生成每层的 rank 门控向量
        # s_map = hyper(pooled)  # {layer_name: [1, r]}

        # # 注入 (A,B,svec,alpha) 到目标层
        # alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
        # for name, mod in targets:
        #     A, Bm = hyper.get_AB_by_name(name)
        #     svec = s_map[name]                         # [1, r]
        #     mod._hyper_state = (A, Bm, svec, alpha_t)

        # 编码完整提示并生成（与脚本1一致）
        enc = tok(prompt, return_tensors="pt").to(device)
        dyn_max_new = max(128, min(max_new_tokens, int(len(tok.encode(chunk_text)) * 1.1)))
        gen_ids = lm.generate(
            **enc,
            max_new_tokens=dyn_max_new,
            temperature=0.7,
            top_p=0.9,
        )
        full_text = tok.decode(gen_ids[0], skip_special_tokens=True)
        # 只取 "### Edited Content:" 之后的部分（保持一致）
        part = full_text.split("### Edited Content:")
        outputs.append(part[-1].strip() if len(part) > 1 else full_text.strip())

        # 清理状态，避免影响下一个块
        clear_hyperlora_state(targets)

    return "\n\n".join(outputs).strip()

###########################################################
# 6) BLEU
###########################################################
def compute_sentence_bleu(reference_text, candidate_text):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([word_tokenize(reference_text)], word_tokenize(candidate_text), smoothing_function=smoothie)

###########################################################
# 7) 主流程：评测多文件（输出与脚本1完全一致）
###########################################################
def main():
    # === 路径设置 ===
    HYPER_CKPT_PATH = "/home/ieb24002/LLM_Code/LLM_Edit/hyperedit_light_qwen_2.5_2e-5_without_L2_pred/epoch2.pt"  # <== 改成你的 .pt
    EVAL_FILES = [
        "/home/ieb24002/LLM_Code/LLM_Edit/Evaluation_paper_use/code_benchmark_eval_filtered.json",
        "/home/ieb24002/LLM_Code/LLM_Edit/Evaluation_paper_use/latex_benchmark_eval_filtered.json",
        "/home/ieb24002/LLM_Code/LLM_Edit/Evaluation_paper_use/sql_benchmark_eval_filtered.json",
        "/home/ieb24002/LLM_Code/LLM_Edit/Evaluation_paper_use/wiki_benchmark_eval_filtered.json",
    ]
    OUTPUT_DIR = "HyperLoRA_output_pred_11_2_paper_use"  # <== 改成你想要的输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, lm, st, hyper, targets, hooks, dtype, alpha, sent_max_len = load_hyper_from_pt_st(HYPER_CKPT_PATH, device)
    log_info("Base model + ST + HyperLoRA loaded. Start evaluation ...")

    for eval_file in EVAL_FILES:
        if not os.path.exists(eval_file):
            log_info(f"Eval file not found, skip: {eval_file}")
            continue

        with open(eval_file, "r", encoding="utf-8") as f:
            data_eval = json.load(f)
        if isinstance(data_eval, dict) and "data" in data_eval:
            data_eval = data_eval["data"]

        base_name = os.path.splitext(os.path.basename(eval_file))[0]
        base_name_out = base_name.replace("_eval", "_generate")
        out_path = os.path.join(OUTPUT_DIR, f"{base_name_out}_output.json")
        bleu_path = os.path.join(OUTPUT_DIR, f"{base_name_out}_bleu.json")

        total_bleu, count = 0.0, 0
        refs_for_corpus, cands_for_corpus = [], []
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write("[\n")  # JSON 数组开头
            first = True  # 用于逗号控制

            for record in data_eval:
                sample_id = record["id"]
                context   = record["context"]
                edit_req  = record["edit_request"]
                ref_text  = record["edit_content"]

                print(f"\n============================")
                print(f"🧩 Sample ID: {sample_id}")
                print(f"✏️ Edit Request: {edit_req}")
                print(f"📄 Context (first 100 chars): {context[:100]}...")

                # ===== 计时开始 =====
                start_time = time.time()
                # 生成编辑结果
                gen_text = generate_edited_text_st(
                    context=context,
                    edit_request=edit_req,
                    lm=lm, tok=tok, st=st, hyper=hyper, targets=targets,
                    alpha=alpha, sent_max_len=sent_max_len,
                    max_chunk_tokens=2000, max_new_tokens=2000,
                )
                end_time = time.time()
                elapsed = end_time - start_time
                # ===== 计时结束 =====

                # 计算 BLEU
                ch_bleu = compute_sentence_bleu(ref_text, gen_text)

                # ===== 打印结果到控制台 =====
                print(f"✅ BLEU: {ch_bleu:.4f}")
                print(f"⏱️ Time: {elapsed:.2f} s")
                print(f"📝 Generated Edit (first 300 chars):\n{gen_text[:300]}...\n")
                # ==========================

                # ===== 实时写入 JSON 文件 =====
                record_out = {
                    "id": sample_id,
                    "edit_request": edit_req,
                    "generated_edit": gen_text,
                    "reference": ref_text,
                    "BLEU": ch_bleu,
                    "time_sec": round(elapsed, 2)
                }

                if not first:
                    fout.write(",\n")
                else:
                    first = False

                fout.write(json.dumps(record_out, ensure_ascii=False, indent=2))
                fout.flush()  # 立即写入磁盘
                # =====================================

            fout.write("\n]\n")  # JSON 数组结尾



        if count == 0:
            log_info(f"No samples in {eval_file}, skip BLEU.")
            continue

        avg_bleu = total_bleu / count
        smoothie = SmoothingFunction().method4
        corpus_score = corpus_bleu(refs_for_corpus, cands_for_corpus, smoothing_function=smoothie)

        with open(bleu_path, "w", encoding="utf-8") as fb:
            json.dump({
                "average_sentence_bleu": avg_bleu,
                "corpus_bleu": corpus_score,
                "num_samples": count
            }, fb, ensure_ascii=False, indent=2)

        log_info(f"Finished {eval_file}. Average BLEU={avg_bleu:.4f}, Corpus BLEU={corpus_score:.4f}")

    # 结束前移除 hooks（可选）
    for h in hooks:
        h.remove()
    log_info("All evaluation files processed. Exiting.")

if __name__ == "__main__":
    main()
