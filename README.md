# HyperEdit

**Unlocking Instruction-based Text Editing in LLMs via Hypernetworks**

## Abstract

Instruction-based text editing is increasingly critical for real-world applications such as code editors (e.g., Cursor), but Large Language Models (LLMs) continue to struggle with this task. Unlike free-form generation, editing requires faithfully implementing user instructions while preserving unchanged content, as even minor unintended modifications can break functionality. Existing approaches treat editing as generic text generation, leading to two key failures: they struggle to faithfully align edits with diverse user intents, and they often over-edit unchanged regions.

We propose **HyperEdit** to address both issues. First, we introduce **hypernetwork-based dynamic adaptation** that generates request-specific parameters, enabling the model to tailor its editing strategy to each instruction. Second, we develop **difference-aware regularization** that focuses supervision on modified spans, preventing over-editing while ensuring precise, minimal changes. HyperEdit achieves a **9%–30% relative improvement** in BLEU on modified regions over state-of-the-art baselines, despite utilizing only **3B parameters**.

## Getting Started

```bash
git clone https://github.com/StuRinDQB/hyper_edit.git
cd hyper_edit
pip install -r requirements.txt
```

## Citation

```bibtex
@article{zeng2025hyperedit,
  title={HyperEdit: Unlocking Instruction-based Text Editing in LLMs via Hypernetworks},
  author={Zeng, Yiming and Cao, Jinghan and Li, Zexin and Yu, Wanhao and Ye, Zhankai and Xiang, Dawei and Hua, Ting and Liu, Xin and Gao, Shangqian and Yu, Tingting},
  year={2025}
}
```
