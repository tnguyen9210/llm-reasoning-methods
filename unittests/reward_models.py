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
