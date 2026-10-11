"""Optional local embedding backend; importing this module never loads a model.

Install requirements-client.txt or requirements-server.txt for the appropriate
PyTorch environment. Corpus and query embeddings must use the same model revision and
prompt settings. Chunk token counts include the document prompt and special tokens.
"""
from __future__ import annotations

import bisect
import logging
import re

LOG = logging.getLogger(__name__)
DEFAULT_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


def resolve_revision(model_id, revision=None):
    """Resolve a Hugging Face branch/tag to an immutable commit (lazily online)."""
    if revision and _COMMIT.fullmatch(revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-client.txt for local search, or "
            "requirements-server.txt for GPU embedding."
        ) from exc
    commit = HfApi().model_info(model_id, revision=revision).sha
    if not commit or not _COMMIT.fullmatch(commit):
        raise RuntimeError("Hugging Face did not return an immutable model commit.")
    return commit.lower()


def _token_count(tokenizer, text, special=True):
    return len(tokenizer.encode(text, add_special_tokens=special, truncation=False))


def _text_boundaries(tokenizer, text):
    """Prefer original-text token offsets, falling back to character boundaries.

    Decoding token slices can lose or change Unicode characters. Slicing the
    original text and checking the resulting encoding avoids that failure mode.
    """
    try:
        offsets = tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True,
        )["offset_mapping"]
        positions = [int(start) for start, end in offsets if 0 <= start < end <= len(text)]
        if positions and positions == sorted(positions):
            return sorted(set([0, len(text)] + positions))
    except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError):
        pass
    return range(len(text) + 1)


def chunk_text(tokenizer, text, prompt, max_tokens, overlap=64):
    """Split text without truncation, retaining the final tail and original bytes.

    Every returned ``token_count`` measures ``prompt + text`` with special tokens.
    Overlap is at most the requested number of unprompted text tokens. Some token
    boundaries can encode differently in isolation, so every final chunk is
    re-tokenized and checked rather than relying on additive prefix estimates.
    """
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a nonnegative integer")
    if not isinstance(text, str) or not isinstance(prompt, str):
        raise TypeError("text and prompt must be strings")
    capacity = max_tokens - _token_count(tokenizer, prompt)
    if capacity < 1:
        raise ValueError("max_tokens leaves no room after the prompt and special tokens")
    if overlap >= capacity:
        raise ValueError("overlap must be smaller than the available text token budget")
    if not text:
        return []
    count = _token_count(tokenizer, prompt + text)
    if count <= max_tokens:
        return [{"text": text, "token_count": count}]

    boundaries = _text_boundaries(tokenizer, text)
    start = 0
    chunks = []
    while start < len(text):
        first = bisect.bisect_right(boundaries, start)
        low, high = first, len(boundaries) - 1
        end_index = None
        # Token counts are approximately monotone; even for unusual tokenizers,
        # each accepted endpoint is independently checked, so no chunk truncates.
        while low <= high:
            middle = (low + high) // 2
            count = _token_count(tokenizer, prompt + text[start:boundaries[middle]])
            if count <= max_tokens:
                end_index = middle
                low = middle + 1
            else:
                high = middle - 1
        if end_index is None:
            # An unusually large token span may still contain fitting characters.
            if not isinstance(boundaries, range):
                boundaries = range(len(text) + 1)
                continue
            raise ValueError("max_tokens cannot fit even one text character after the prompt")
        end = boundaries[end_index]
        chunk = text[start:end]
        count = _token_count(tokenizer, prompt + chunk)
        if count > max_tokens:
            raise ValueError("Tokenizer produced an inconsistent token count")
        chunks.append({"text": chunk, "token_count": count})
        if end == len(text):
            break
        next_start = end
        if overlap:
            # Find a suffix whose *actual* token count fits the overlap budget.
            low, high = first, end_index
            while low <= high:
                middle = (low + high) // 2
                candidate = boundaries[middle]
                if _token_count(tokenizer, text[candidate:end], special=False) <= overlap:
                    next_start = candidate
                    high = middle - 1
                else:
                    low = middle + 1
        if next_start <= start:
            raise ValueError("Chunk overlap failed to make forward progress")
        start = next_start
    return chunks


class SentenceEncoder:
    """Sentence Transformers adapter with pinned weights and explicit prompts."""

    def __init__(self, model_id=DEFAULT_MODEL, revision=None, max_tokens=2048,
                 device="auto", dtype="auto"):
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        try:
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-client.txt for local search, or "
                "requirements-server.txt with CUDA-enabled PyTorch for GPU embedding."
            ) from exc
        self._np, self._torch = np, torch
        self.model_id = model_id
        self.revision = resolve_revision(model_id, revision)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.max_tokens = max_tokens
        LOG.info("Loading %s at %s on %s (%s)", model_id, self.revision, self.device, self.dtype)
        self.model = SentenceTransformer(
            model_id, revision=self.revision, device=self.device,
            model_kwargs={"dtype": getattr(torch, self.dtype), "attn_implementation": "sdpa"},
            trust_remote_code=False,
        )
        self.tokenizer = self.model.tokenizer
        context_limits = []
        # The embedding model can impose a stricter, trained sequence limit than
        # the underlying language model's positional embeddings (e.g. 32768 vs
        # 262144). Preserve that limit before setting our operational chunk size.
        embedding_limit = getattr(self.model, "max_seq_length", None)
        if isinstance(embedding_limit, int) and 0 < embedding_limit < 10**9:
            context_limits.append(embedding_limit)
        # Tokenizer limits can use huge sentinel values to mean unspecified.
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            context_limits.append(tokenizer_limit)
        first_module = self.model[0]
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        for name in ("max_position_embeddings", "n_positions"):
            value = getattr(config, name, None)
            if isinstance(value, int) and 0 < value < 10**9:
                context_limits.append(value)
        if not context_limits:
            raise ValueError("Cannot determine the model's supported context limit")
        if max_tokens > min(context_limits):
            raise ValueError(
                f"max_tokens={max_tokens} exceeds the model context limit {min(context_limits)}"
            )
        self.model.max_seq_length = max_tokens
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        prompts = self.model.prompts or {}
        self.document_prompt = next(
            (prompts[name] for name in ("document", "passage", "corpus") if name in prompts), ""
        )
        self.query_prompt = prompts.get("query", "")
        if not isinstance(self.document_prompt, str) or not isinstance(self.query_prompt, str):
            raise ValueError("The model's query and document prompts must be strings")

    def _resolve_device(self, device):
        torch = self._torch
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            return device
        if device == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise ValueError("MPS was requested but is unavailable in this PyTorch environment")
            return device
        if re.fullmatch(r"cuda(?::\d+)?", device):
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable in this PyTorch environment")
            index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} does not exist")
            return device
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")

    def _resolve_dtype(self, dtype):
        aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        dtype = aliases.get(dtype, dtype)
        cuda = self.device.startswith("cuda")
        if dtype == "auto":
            if cuda:
                with self._torch.cuda.device(self.device):
                    return "bfloat16" if self._torch.cuda.is_bf16_supported() else "float16"
            return "float32"
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if cuda and dtype == "bfloat16":
            with self._torch.cuda.device(self.device):
                if not self._torch.cuda.is_bf16_supported():
                    raise ValueError("bfloat16 is not supported on the selected CUDA device")
        return dtype

    def document_chunks(self, text, overlap=64):
        return chunk_text(self.tokenizer, text, self.document_prompt, self.max_tokens, overlap)

    def _encode(self, texts, batch_size, query):
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not a single string")
        texts = list(texts)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        prompt = self.query_prompt if query else self.document_prompt
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("Every input text must be a string")
            length = _token_count(self.tokenizer, prompt + text)
            if length > self.max_tokens:
                kind = "Query" if query else "Document"
                raise ValueError(
                    f"{kind} {index} uses {length} tokens including its prompt; "
                    f"the limit is {self.max_tokens}. Split it explicitly before encoding."
                )
        if not texts:
            return self._np.empty((0, self.dimension), dtype=self._np.float32)
        method = self.model.encode_query if query else self.model.encode_document
        batch_size = min(batch_size, len(texts))
        while True:
            try:
                with self._torch.inference_mode():
                    # Pass the chosen prompt exactly once. This also overrides
                    # default_prompt_name, avoiding hidden/default prefix drift.
                    vectors = method(
                        texts, prompt=prompt, batch_size=batch_size,
                        normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                result = self._np.ascontiguousarray(
                    vectors.detach().to(device="cpu", dtype=self._torch.float32).numpy(),
                    dtype=self._np.float32,
                )
                # Normalize again in float32: BF16 inference can round the
                # model's normalization enough to distort cosine dot products.
                norms = self._np.linalg.norm(result, axis=1, keepdims=True)
                if not self._np.isfinite(result).all() or (norms <= 0).any():
                    raise ValueError("The model returned non-finite or zero embedding vectors")
                result /= norms
                return result
            except self._torch.cuda.OutOfMemoryError:
                if not self.device.startswith("cuda") or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self._torch.cuda.empty_cache()
                LOG.warning("CUDA memory exhausted; retrying with batch_size=%d", batch_size)

    def encode_documents(self, texts, batch_size=32):
        """Encode already-budgeted chunks as normalized float32 document vectors."""
        return self._encode(texts, batch_size, query=False)

    def encode_queries(self, texts, batch_size=1):
        """Encode queries; overlong input raises instead of silently truncating."""
        return self._encode(texts, batch_size, query=True)

    def synchronize(self):
        """Wait for device work, so benchmark wall times include GPU execution."""
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self._torch.mps.synchronize()
