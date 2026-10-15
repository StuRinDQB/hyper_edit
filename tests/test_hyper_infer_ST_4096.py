"""CPU checks for lightweight checkpoint loading and per-request LoRA state."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nltk
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The legacy evaluation entry point requests NLTK data on import.
with patch.object(nltk, 'download'):
    import hyper_infer_ST_4096 as infer
from Train_Light.hyper_light_train_gold import GRUHyperLoRA as GoldHyper
from Train_Light.hyper_light_train_pre import GRUHyperLoRA as PredHyper


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 4)
        self.v_proj = nn.Linear(3, 2)
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=0)


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        # Saved layer order matters for layer_id_embed; it need not be lexical.
        self.meta = [('v_proj', 3, 2), ('q_proj', 3, 4)]

    def training_model(self, kind=GoldHyper):
        model = kind(hidden_size=6, targets_meta=self.meta, rank=2,
                     dropout=0.0, num_codes=3, gru_layers=1, d_h=5).eval()
        with torch.no_grad():
            for parameter in model.B.values():
                parameter.normal_()
        return model

    def checkpoint(self, model):
        return {
            'model_name': 'test-base', 'sent_model': 'test-encoder',
            'targets': self.meta, 'hyper': copy.deepcopy(model.state_dict()),
            'config': {'rank': 2, 'alpha': 4.0, 'num_codes': 3,
                       'gru_layers': 1, 'hyper_dropout': 0.0, 'sent_max_len': 17},
        }

    def load(self, checkpoint, lm):
        st = Mock()
        st.parameters.return_value = []
        st.get_sentence_embedding_dimension.return_value = 6
        tok = SimpleNamespace(pad_token_id=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            torch.save(checkpoint, path)
            with patch.object(infer.AutoTokenizer, 'from_pretrained', return_value=tok), \
                 patch.object(infer.AutoModelForCausalLM, 'from_pretrained', return_value=lm), \
                 patch.object(infer, 'SentenceTransformer', return_value=st), \
                 patch.object(torch.cuda, 'is_available', return_value=True):
                return infer.load_hyper_from_pt_st(str(path), torch.device('cpu'))

    def test_gold_and_pred_checkpoints_match_training_forward(self):
        for kind in (GoldHyper, PredHyper):
            with self.subTest(training=kind.__module__):
                training = self.training_model(kind)
                result = self.load(self.checkpoint(training), TinyLM())
                _, lm, st, loaded, targets, hooks, dtype, alpha, sent_max_len = result
                try:
                    self.assertEqual(dtype, torch.float32)
                    self.assertEqual(alpha, 4.0)
                    self.assertEqual(sent_max_len, 17)
                    self.assertEqual(loaded.targets_meta, self.meta)
                    self.assertEqual(len(hooks), 2)
                    st.eval.assert_called_once()
                    pooled = torch.randn(2, 6)
                    expected, expected_hidden = training(pooled)
                    actual, actual_hidden = loaded(pooled)
                    torch.testing.assert_close(actual_hidden, expected_hidden)
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name])
                        torch.testing.assert_close(loaded.get_B_by_name(name), training.get_B_by_name(name))
                finally:
                    for hook in hooks:
                        hook.remove()

    def test_bad_checkpoints_fail_before_attaching_hooks(self):
        good = self.checkpoint(self.training_model())
        for fault in ('missing', 'unexpected', 'shape', 'duplicate', 'legacy'):
            with self.subTest(fault=fault):
                checkpoint = copy.deepcopy(good)
                if fault == 'missing':
                    del checkpoint['hyper']['a_proj.q_proj.weight']
                elif fault == 'unexpected':
                    checkpoint['hyper']['a_generators.old.weight'] = torch.zeros(1)
                elif fault == 'shape':
                    checkpoint['targets'] = [('v_proj', 3, 2), ('q_proj', 99, 4)]
                elif fault == 'duplicate':
                    checkpoint['targets'] = self.meta + [self.meta[0]]
                else:
                    del checkpoint['hyper']['layer_gru.weight_hh_l0']
                lm = TinyLM()
                with self.assertRaises((RuntimeError, ValueError)):
                    self.load(checkpoint, lm)
                self.assertFalse(lm.q_proj._forward_hooks)
                self.assertFalse(lm.v_proj._forward_hooks)

    def test_hook_applies_each_batch_adapter_and_clears(self):
        layer = nn.Linear(3, 4)
        targets = [('q_proj', layer)]
        hooks = infer.attach_hyperlora_hooks(targets, rank=2)
        try:
            x = torch.randn(2, 5, 3)
            base = nn.functional.linear(x, layer.weight, layer.bias)
            a = torch.randn(2, 2, 3)
            b = torch.randn(4, 2)
            layer._hyper_state = (a, b, torch.tensor(6.0))
            expected = base + torch.stack([x[i] @ a[i].T @ b.T for i in range(2)]) * 3.0
            torch.testing.assert_close(layer(x), expected)
            self.assertFalse(torch.allclose(layer(x), base))
            infer.clear_hyperlora_state(targets)
            torch.testing.assert_close(layer(x), base)
        finally:
            for hook in hooks:
                hook.remove()

    def test_generation_cleanup_and_continuation_decoding(self):
        lm = TinyLM()
        hyper = self.training_model()
        targets = infer.find_target_linear_modules(lm)
        hooks = infer.attach_hyperlora_hooks(targets, rank=2)
        tok = Mock()
        tok.encode.return_value = [1, 2, 3]
        tok.return_value = Batch(input_ids=torch.tensor([[1, 2, 3]]))
        decoded = []

        def decode(ids, **kwargs):
            decoded.append(ids.tolist())
            return 'kept ### Edited Content: text'

        tok.decode.side_effect = decode

        def generate(**kwargs):
            self.assertTrue(all(layer._hyper_state is not None for _, layer in targets))
            return torch.tensor([[1, 2, 3, 8, 9]])

        try:
            with patch.object(infer, 'encode_prompts_with_st', return_value=torch.randn(1, 6)):
                lm.generate = Mock(side_effect=RuntimeError('generation failed'))
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    infer.generate_edited_text_st('context', 'request', lm, tok, None, hyper, targets, 4.0)
                self.assertTrue(all(layer._hyper_state is None for _, layer in targets))
                lm.generate = generate
                result = infer.generate_edited_text_st('context', 'request', lm, tok, None, hyper, targets, 4.0)
                self.assertEqual(result, 'kept ### Edited Content: text')
                self.assertEqual(decoded, [[8, 9]])
                self.assertTrue(all(layer._hyper_state is None for _, layer in targets))
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == '__main__':
    unittest.main()
