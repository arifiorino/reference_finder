"""Optional real-library smoke tests using tiny, random Qwen weights.

Run in an environment with requirements-client.txt installed. No pretrained
weights, tokenizer files, or network requests are needed. These checks verify
the inference API and kernels; they do not evaluate the 4B model's quality.
"""
import json
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

from reference_finder.embedding_backend import SentenceEncoder
from reference_finder.embeddings import build_index, encoder_config, search_index


QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


class QwenRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "torch", "sentence_transformers", "transformers", "tokenizers", "accelerate"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest("Install embedding requirements to run the real Qwen smoke tests")
        import numpy as np
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
        cls.np, cls.torch = np, torch
        temporary = tempfile.TemporaryDirectory(prefix="tiny-qwen-")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        words = ["[UNK]", "[PAD]", "graph", "proof", "botnet", "privacy", "social",
                 "network", "zero", "knowledge", "absence", "of", "a", "the", "is", "safe"]
        raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
            model_max_length=128, padding_side="left",
        )
        tokenizer.save_pretrained(cls.directory)
        config = Qwen3Config(
            vocab_size=len(words), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
            use_cache=True, pad_token_id=1,
        )
        torch.manual_seed(17)
        model = Qwen3Model(config)
        model.save_pretrained(cls.directory)
        cls.write_json("modules.json", [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ])
        cls.write_json("sentence_bert_config.json", {"max_seq_length": 128, "do_lower_case": False})
        cls.write_json("config_sentence_transformers.json", {
            "prompts": {"query": QUERY_PROMPT, "document": ""},
            "default_prompt_name": None, "similarity_fn_name": "cosine",
        })
        cls.write_json("1_Pooling/config.json", {
            "word_embedding_dimension": 32, "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False, "pooling_mode_lasttoken": True,
            "include_prompt": True,
        })
        (cls.directory / "2_Normalize").mkdir()

    @classmethod
    def write_json(cls, name, value):
        destination = cls.directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value), encoding="utf-8")

    def encoder(self, device="cpu", dtype="float32"):
        return SentenceEncoder(str(self.directory), revision="a" * 40, max_tokens=64,
                               device=device, dtype=dtype)

    def test_actual_qwen_load_padding_pooling_and_normalization(self):
        encoder = self.encoder()
        documents = ["graph proof", "social network proof of botnet absence"]
        vectors = encoder.encode_documents(documents, batch_size=2)
        self.assertEqual(vectors.shape, (2, 32))
        self.assertEqual(encoder.tokenizer.padding_side, "left")
        self.assertFalse(encoder.model[0].auto_model.config.use_cache)
        self.np.testing.assert_allclose(self.np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        # Compare with native last-token pooling, independently of ST's pooler.
        tokens = encoder.model.tokenize(documents)
        with self.torch.inference_mode():
            hidden = encoder.model[0].auto_model(**tokens).last_hidden_state
            expected = self.torch.nn.functional.normalize(hidden[:, -1, :], p=2, dim=-1)
        self.np.testing.assert_allclose(vectors, expected.numpy(), atol=1e-6)
        singles = self.np.vstack([encoder.encode_documents([text]) for text in documents])
        self.np.testing.assert_allclose(vectors, singles, atol=1e-5)

    def test_query_instruction_applied_exactly_once(self):
        encoder = self.encoder()
        self.assertEqual(encoder.document_prompt, "")
        self.assertEqual(encoder.query_prompt, QUERY_PROMPT)
        query = "zero knowledge proof of botnet absence"
        actual = encoder.encode_queries([query])
        expected = encoder.encode_documents([QUERY_PROMPT + query])
        self.np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_real_vectors_roundtrip_through_sqlite_search(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.sqlite3"
            index = Path(directory) / "index.sqlite3"
            with sqlite3.connect(corpus) as conn:
                conn.execute("CREATE TABLE papers (id INTEGER, title TEXT, abstract TEXT)")
                conn.executemany("INSERT INTO papers VALUES (?,?,?)", [
                    (1, "graph proof", "zero knowledge proof of botnet absence"),
                    (2, "social network", "privacy proof"),
                    (3, "botnet", None),
                ])
            encoder = self.encoder()
            config = encoder_config(encoder, max_tokens=64, overlap=4)
            status = build_index(corpus, index, encoder, config, batch_size=2)
            self.assertEqual((status["embedded"], status["pending"], status["title_only"]), (3, 0, 1))
            matches = search_index(index, encoder, "graph proof", top_k=3)
            self.assertEqual({m["paper_id"] for m in matches}, {1, 2, 3})
            self.assertTrue(all(self.np.isfinite(m["score"]) for m in matches))
            self.assertEqual(sum(not m["has_abstract"] for m in matches), 1)

    def test_mps_fp16_matches_cpu_direction(self):
        if not self.torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable on this test host")
        query = "zero knowledge proof of botnet absence"
        reference = self.encoder().encode_queries([query])[0]
        encoder = self.encoder(device="mps", dtype="float16")
        actual = encoder.encode_queries([query])[0]
        encoder.synchronize()
        self.assertTrue(self.np.isfinite(actual).all())
        self.assertAlmostEqual(float(self.np.linalg.norm(actual)), 1.0, places=5)
        self.assertGreater(float(actual @ reference), 0.999)


if __name__ == "__main__":
    unittest.main()
