"""Process reward model (PRM) wrappers.

A small, uniform interface for per-step scoring, designed to drop
into the MCTS search loop:

    prm = QwenPRM(model_path)
    scores = prm.score(questions, answers, batch_size=4)
    #   questions: list[str]      one problem each
    #   answers  : list[list[str]]  per question, candidate answers,
    #                               each "\\n\\n"-joined steps
    #   scores   : list[list[list[float]]]  [question][answer][step]

The base class owns all the plumbing (flatten the question x answer
grid, chunk into batches, reshape back). A subclass implements just
two things:

    _load()        -> set self.model, self.tokenizer (+ token ids)
    _score_batch() -> per-step scores for a flat list of (question,
                      answer) pairs

Per-step *position masking* is model-specific and lives in
_score_batch. Aggregation (min / prod / last) is the caller's job;
score() returns raw per-step rewards.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Qwen2.5-Math-PRM-7B model-card system prompt.
QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer "
    "within \\boxed{}."
)


class PRM(ABC):
    """Per-step process reward model with a uniform score() entry.

    Subclasses implement _load() and _score_batch(); the base class
    handles batching and the question x answer reshaping.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        **model_kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._load(**model_kwargs)

    @abstractmethod
    def _load(self, **model_kwargs) -> None:
        """Set self.model and self.tokenizer (and any token ids)."""

    @abstractmethod
    def _score_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[list[float]]:
        """Score one batch of (question, answer) pairs.

        Each answer is a "\\n\\n"-joined string of steps. Returns one
        list of per-step floats per pair, in input order. Model-
        specific position masking lives here.
        """

    def _embed_batch(
        self,
        pairs: list[tuple[str, str]],
        system_prompt: str,
        layer: int = -1,
    ) -> list["torch.Tensor"]:
        """Per-token hidden states for one batch of (question, answer)
        pairs, for the embedding-diversity term in semantic MCTS v02.

        Returns one (seq_len, hidden_dim) tensor per pair (padding
        trimmed), taken from hidden-state `layer` (-1 = last). Unlike
        _score_batch, the forward pass is over the *plain* candidate
        chat (system / user(question) / assistant(answer)) — NOT the
        judge transcript _score_batch builds — so the pooled vector is
        a representation of the candidate as written, in the PRM's
        space. `system_prompt` is the generator's system prompt
        (config.gen.system_prompt), passed in so the embedded text
        matches v01's exactly and the v01-vs-v02 ablation stays clean
        (same text + pooling, only the model differs).

        Optional; only PRMs used as an embedding source need it. The
        default raises so an unconfigured PRM fails loudly rather than
        silently producing garbage embeddings.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _embed_batch; "
            "it cannot be used as an embeds_source for mcts_sem v02."
        )

    def embed(
        self,
        questions: list[str],
        answers: list[list[str]],
        system_prompt: str,
        batch_size: int = 8,
        layer: int = -1,
    ) -> list[list["torch.Tensor"]]:
        """Per-candidate pooled-input hidden states, indexed
        [question][answer]. Mirrors score()'s flatten/batch/reshape so
        the two stay aligned; delegates the model-specific work to
        _embed_batch. Each element is a (seq_len, hidden_dim) tensor.
        `system_prompt` (config.gen.system_prompt) is threaded through
        so the embedded chat matches what the generator embeds in v01.
        """
        pairs: list[tuple[str, str]] = []
        answer_counts: list[int] = []
        for question, cands in zip(questions, answers, strict=True):
            answer_counts.append(len(cands))
            for ans in cands:
                pairs.append((question, ans))

        flat: list["torch.Tensor"] = []
        for i in range(0, len(pairs), batch_size):
            flat.extend(
                self._embed_batch(
                    pairs[i : i + batch_size], system_prompt, layer
                )
            )

        out: list[list["torch.Tensor"]] = []
        cursor = 0
        for count in answer_counts:
            out.append(flat[cursor : cursor + count])
            cursor += count
        return out

    def score(
        self,
        questions: list[str],
        answers: list[list[str]],
        batch_size: int = 8,
    ) -> list[list[list[float]]]:
        """Per-step rewards, indexed [question][answer][step]."""
        # Flatten the question x answer grid into pairs, remembering
        # how many answers each question had so we can reshape back.
        pairs: list[tuple[str, str]] = []
        answer_counts: list[int] = []
        for question, cands in zip(questions, answers, strict=True):
            answer_counts.append(len(cands))
            for ans in cands:
                pairs.append((question, ans))

        # Score in batches.
        flat_scores: list[list[float]] = []
        for i in range(0, len(pairs), batch_size):
            flat_scores.extend(self._score_batch(pairs[i : i + batch_size]))

        # Reshape flat -> [question][answer][step].
        out: list[list[list[float]]] = []
        cursor = 0
        for count in answer_counts:
            out.append(flat_scores[cursor : cursor + count])
            cursor += count
        return out


class QwenPRM(PRM):
    """Qwen2.5-Math-PRM-7B.

    A reward head (loaded via AutoModel) emits an (incorrect, correct)
    probability pair at each `<extra_0>` separator token. We assemble
    system / user(problem) / assistant(steps + separators), run one
    forward pass, and read P(correct) at every separator.

    Loaded in fp16 for the V100 (the card recommends bf16, which sm_70
    lacks); fp16 preserves step rankings, drifts absolute scores a
    little. use_cache=False keeps a single forward pass light and
    sidesteps a removed cache API in the bundled remote code under
    newer transformers.
    """

    SEPARATOR = "<extra_0>"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        system_prompt: str = QWEN_SYSTEM_PROMPT,
        **model_kwargs,
    ):
        self.system_prompt = system_prompt
        super().__init__(model_path, device, dtype, **model_kwargs)

    def _load(self, **model_kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=self.dtype,
            trust_remote_code=True,
            **model_kwargs,
        ).eval()

        sep_ids = self.tokenizer.encode(
            self.SEPARATOR, add_special_tokens=False
        )
        if len(sep_ids) != 1:
            raise ValueError(
                f"Expected one separator token, got {sep_ids}"
            )
        self.sep_token_id = sep_ids[0]

    def _build_prompt(self, question: str, answer: str) -> str:
        # Each step ends with a separator, including the last, so the
        # final step gets its own score position.
        steps = answer.split("\n\n")
        assistant = self.SEPARATOR.join(steps) + self.SEPARATOR
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    def _score_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[list[float]]:
        prompts = [self._build_prompt(q, a) for q, a in pairs]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        with torch.no_grad():
            # Reward head: logits are (batch, seq_len, 2).
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )[0]
        probs = F.softmax(logits, dim=-1)[:, :, 1]  # P(correct)

        # Gather P(correct) at each separator position, per sample.
        sep_mask = input_ids == self.sep_token_id
        batch_scores: list[list[float]] = []
        for b in range(input_ids.shape[0]):
            batch_scores.append(
                probs[b][sep_mask[b]].detach().cpu().float().tolist()
            )
        return batch_scores

    def _embed_batch(
        self,
        pairs: list[tuple[str, str]],
        system_prompt: str,
        layer: int = -1,
    ) -> list[torch.Tensor]:
        """Last-(or `layer`-)layer hidden states over the PLAIN
        candidate chat, for mcts_sem v02's diversity term.

        Mirrors RLHFlowPRM._embed_batch so the v01/v02 source ablation
        stays clean: we render system / user(question) / assistant(answer)
        — the same shape v01 embeds with the generator — and crucially
        WITHOUT the <extra_0> separators _build_prompt inserts (those are
        a scoring-only artifact; the embedded text must match v01's, only
        the model differing). The inner Qwen2Model is the same backbone
        as a causal LM, so the final-norm hook works identically (the
        reward `score` head is bypassed — we never read its logits here).
        Returns one (seq_len, hidden_dim) tensor per pair, padding
        trimmed; the caller pools it. hidden_dim is 3584 here (vs 4096
        for the Llama PRM), so v02 with proj=none needs embeds_dim=3584.
        """
        convs = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            for q, a in pairs
        ]
        chat_texts = self.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False,
        )
        enc = self.tokenizer(
            chat_texts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(self.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Same memory trick as RLHFlowPRM: for layer=-1 hook the inner
        # model's final norm and capture only its output (verified bit-
        # identical to hidden_states[-1] for this checkpoint), instead of
        # output_hidden_states=True materializing all ~L layers. Hook
        # `model.model.norm` — the top-level module is
        # Qwen2ForProcessRewardModel (model: Qwen2Model + score: head),
        # so the backbone's final norm is one level deeper than in a
        # plain CausalLM but the path is the same name.
        if layer == -1:
            captured = {}

            def _hook(module, inputs, output):
                captured["hs"] = (
                    output[0] if isinstance(output, tuple) else output
                )

            handle = self.model.model.norm.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()
            hs = captured["hs"]
        else:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = out.hidden_states[layer]
            del out

        # Trim right-padding per row so pooling sees only real tokens.
        # Clone so the trimmed slices don't pin the full (batch, seq,
        # hidden) tensor, then drop it and free the cached blocks.
        embeds: list[torch.Tensor] = []
        for b in range(input_ids.shape[0]):
            n = int(attention_mask[b].sum().item())
            embeds.append(hs[b, :n].detach().clone())
        del hs
        torch.cuda.empty_cache()
        return embeds


class RLHFlowPRM(PRM):
    """RLHFlow Llama3.1-8B-PRM-Deepseek-Data.

    A causal LM trained to judge each step by predicting `+` or `-` in
    an assistant turn; the per-step reward is P(+) / (P(+) + P(-)).
    The conversation alternates user reasoning-step turns with
    assistant `+` turns. A parallel conversation swaps `+` for the
    unique marker token `ки` so we can locate each score position.
    """

    MARKER_TEXT = "ки"  # a single unique vocab token

    def _load(self, **model_kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=self.dtype,
            **model_kwargs,
        ).eval()

        # Llama ships no pad token; reuse EOS for batched calls.
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # encode("+") prepends BOS; [-1] picks the actual id.
        plus_id = self.tokenizer.encode("+")[-1]
        minus_id = self.tokenizer.encode("-")[-1]
        self.candidate_token_ids = [plus_id, minus_id]

        self.marker_token_id = (
            self.tokenizer(self.MARKER_TEXT, return_tensors="pt")
            .input_ids[0, 1]
            .item()
        )

    def _build_conversations(
        self, question: str, answer: str
    ) -> tuple[list[dict], list[dict]]:
        # Parallel conversations: one with `+`, one with the marker.
        conv, marker_conv = [], []
        for idx, step in enumerate(answer.split("\n\n")):
            # First user turn carries the problem; later turns are
            # the bare step, judged given the prior chain.
            text = question + " " + step if idx == 0 else step
            conv.append({"role": "user", "content": text})
            conv.append({"role": "assistant", "content": "+"})
            marker_conv.append({"role": "user", "content": text})
            marker_conv.append(
                {"role": "assistant", "content": self.MARKER_TEXT}
            )
        return conv, marker_conv

    def _score_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[list[float]]:
        convs, marker_convs = [], []
        for q, a in pairs:
            conv, marker_conv = self._build_conversations(q, a)
            convs.append(conv)
            marker_convs.append(marker_conv)

        # apply_chat_template already emits BOS, so don't add another.
        chat_texts = self.tokenizer.apply_chat_template(
            convs, tokenize=False,
        )
        marker_texts = self.tokenizer.apply_chat_template(
            marker_convs, tokenize=False,
        )
        input_ids = self.tokenizer(
            chat_texts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).input_ids.to(self.device)
        marker_input_ids = self.tokenizer(
            marker_texts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).input_ids.to(self.device)
        if input_ids.shape != marker_input_ids.shape:
            raise RuntimeError(
                f"Batched shape mismatch: {input_ids.shape} vs "
                f"{marker_input_ids.shape}"
            )

        with torch.no_grad():
            logits = self.model(input_ids).logits[
                :, :, self.candidate_token_ids
            ]
            probs = logits.softmax(dim=-1)[:, :, 0]  # P(+)

        # The model predicts token N from position N-1: a marker at
        # index k+1 of marker_input_ids[1:] is predicted by probs[k].
        batch_scores: list[list[float]] = []
        for b in range(input_ids.shape[0]):
            step_scores = probs[b, :-1][
                marker_input_ids[b, 1:] == self.marker_token_id
            ].detach().cpu().float().tolist()
            batch_scores.append(step_scores)
        return batch_scores

    def _embed_batch(
        self,
        pairs: list[tuple[str, str]],
        system_prompt: str,
        layer: int = -1,
    ) -> list[torch.Tensor]:
        """Last-(or `layer`-)layer hidden states over the PLAIN
        candidate chat, for mcts_sem v02's diversity term.

        Deliberately NOT the judge transcript _score_batch builds: we
        render system / user(question) / assistant(answer) — the same
        shape v01 embeds with the generator — so only the model differs
        between v01 and v02. Returns one (seq_len, hidden_dim) tensor
        per pair with right-padding trimmed; the caller pools it.
        """
        # Match v01's embedded text: a normal [system, user, assistant]
        # chat (sal.build_conv shape), continuing the assistant turn so
        # no generation prompt is appended.
        convs = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            for q, a in pairs
        ]
        chat_texts = self.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False,
        )
        enc = self.tokenizer(
            chat_texts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(self.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Memory: output_hidden_states=True materializes ALL ~L layers'
        # hidden states (an (L+1)-tuple) just to read one — costly for an
        # 8B model on long sequences. For the common layer=-1 case we
        # instead hook the final norm and capture only its output,
        # letting every other layer's activations free as usual. We hook
        # `model.norm` (NOT the last decoder layer): in current
        # transformers, hidden_states[-1] is the post-final-norm tensor,
        # so the last decoder layer's raw output differs from it (verified
        # — they're off by the RMSNorm). Any other `layer` falls back to
        # the full path; layer=-1 is the only value used in practice.
        if layer == -1:
            captured = {}

            def _hook(module, inputs, output):
                captured["hs"] = (
                    output[0] if isinstance(output, tuple) else output
                )

            handle = self.model.model.norm.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            finally:
                handle.remove()
            hs = captured["hs"]
        else:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            # hidden_states is a tuple (embeddings, layer_1, ..., layer_L);
            # index `layer`. Shape: (batch, seq_len, hidden).
            hs = out.hidden_states[layer]
            del out

        # Trim right-padding per row so pooling sees only real tokens.
        # Clone so the trimmed slices don't pin the full (batch, seq,
        # hidden) tensor, then drop it and free the cached blocks.
        embeds: list[torch.Tensor] = []
        for b in range(input_ids.shape[0]):
            n = int(attention_mask[b].sum().item())
            embeds.append(hs[b, :n].detach().clone())
        del hs
        torch.cuda.empty_cache()
        return embeds


# Registry mapping cfg.prm.kind -> wrapper class. Single source of
# truth for "which PRM kinds exist" so launchers (generate_mcts_cnt,
# generate_mcts_sem, prepare_scored_dataset, ...) don't each carry
# their own copy of this dict — add a new kind here once and every
# caller of build_prm() picks it up.
PRM_REGISTRY: dict[str, type[PRM]] = {
    "rlhflow": RLHFlowPRM,
    "qwen": QwenPRM,
}


def build_prm(kind: str, model_path: str, device: str = "cuda:0", **kwargs) -> PRM:
    """Construct the PRM wrapper registered under `kind`.

    Raises ValueError (not KeyError) on an unknown kind, listing the
    valid options — callers don't need their own existence check.
    """
    prm_cls = PRM_REGISTRY.get(kind)
    if prm_cls is None:
        raise ValueError(
            f"Unknown prm.kind: {kind!r}. "
            f"Expected one of {sorted(PRM_REGISTRY)}"
        )
    return prm_cls(model_path=model_path, device=device, **kwargs)
