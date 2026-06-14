"""RLHFlow PRM scoring wrapper."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RLHFlowPRM:
    """RLHFlow Llama3.1-8B-PRM wrapper with a uniform score() entry
    point that switches between per-step and batched forward passes.

    Mirrors core/reward_models.py:RLHFFlow but pared to what this
    smoke test needs (no embedding capture, no per-question logging).
    """

    def __init__(
        self,
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        **model_kwargs,
    ):
        self.model_path = model_path
        self.device_map = device_map
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            torch_dtype=torch_dtype, **model_kwargs,
        )

    def _load_model_and_tokenizer(self, torch_dtype, **model_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            **model_kwargs,
        ).eval()

        # Llama tokenizers ship without a pad token; reuse EOS so
        # batched inference doesn't error. Right-padding keeps the
        # final real token at a deterministic offset from the start.
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # encode("+") prepends BOS; [-1] picks the actual "+" id.
        # Restricting later softmax to {plus, minus} normalizes the
        # score as P(+) / (P(+) + P(-)).
        plus_token_id = tokenizer.encode("+")[-1]
        minus_token_id = tokenizer.encode("-")[-1]
        self.candidate_token_ids = [plus_token_id, minus_token_id]

        # "ки" is a Cyrillic bigram the Llama BPE merges into a single
        # token; we use it as a unique marker to locate `+` positions
        # in the chat-template output.
        self.marker_text = "ки"
        self.marker_token_id = (
            tokenizer(self.marker_text, return_tensors="pt")
            .input_ids[0, 1]
            .item()
        )

        return model, tokenizer

    def score(
        self, questions, outputs, batch_size=None,
    ):
        """Score reasoning steps with the PRM.

        Parameters
        ----------
        questions : list[str]
            Math problems, one per question.
        outputs : list[list[str]]
            For each question, a list of candidate answers. Each
            answer is a string with steps separated by "\\n\\n".


        Returns
        -------
        list[list[list[float]]]
            Indexed [question_idx][answer_idx][step_idx] = P(+).
        """
        if batch_size:
            return self._score_batched(questions, outputs, batch_size)
        return self._score_single(questions, outputs)

    def _score_single(self, questions, outputs):
        model = self.model
        tokenizer = self.tokenizer

        all_scores = []
        for q_idx, (question, answers) in enumerate(
            zip(questions, outputs, strict=True)
        ):
            question_scores = []
            for a_idx, ans in enumerate(answers):
                step_texts = ans.split("\n\n")
                conversation = []
                marker_conversation = []
                answer_scores = []
                for step_idx, step in enumerate(step_texts):
                    # First user turn carries the problem; later turns
                    # are just the step so the assistant judges that
                    # step alone given the prior chain.
                    text = (
                        question + " " + step if step_idx == 0 else step
                    )
                    conversation.append({"role": "user", "content": text})
                    conversation.append(
                        {"role": "assistant", "content": "+"}
                    )
                    marker_conversation.append(
                        {"role": "user", "content": text}
                    )
                    marker_conversation.append(
                        {"role": "assistant", "content": self.marker_text}
                    )

                    # Format to text first (tokenize=False), then
                    # encode with add_special_tokens=False — the chat
                    # template already emits BOS, so encode shouldn't
                    # add another.
                    chat_text = tokenizer.apply_chat_template(
                        conversation, tokenize=False,
                    )
                    marker_chat_text = tokenizer.apply_chat_template(
                        marker_conversation, tokenize=False,
                    )
                    input_ids = tokenizer.encode(
                        chat_text, return_tensors="pt",
                        add_special_tokens=False,
                    ).to(model.device)
                    marker_input_ids = tokenizer.encode(
                        marker_chat_text, return_tensors="pt",
                        add_special_tokens=False,
                    ).to(model.device)
                    if input_ids.shape != marker_input_ids.shape:
                        raise RuntimeError(
                            f"Marker shape mismatch: "
                            f"{input_ids.shape} vs "
                            f"{marker_input_ids.shape}"
                        )

                    with torch.no_grad():
                        logits = model(input_ids).logits[
                            :, :, self.candidate_token_ids
                        ]
                        probs = logits.softmax(dim=-1)[:, :, 0]

                    # Slicing [1:] drops BOS; a marker at original
                    # index k appears at index k-1 — also the position
                    # whose logit predicts k. The last marker is the
                    # current step's `+`.
                    marker_positions = (
                        marker_input_ids[0, 1:] == self.marker_token_id
                    ).nonzero(as_tuple=True)[0]
                    if marker_positions.numel() != step_idx + 1:
                        raise RuntimeError(
                            f"Expected {step_idx + 1} marker positions, "
                            f"found {marker_positions.numel()}"
                        )
                    score_pos = marker_positions[-1].item()
                    answer_scores.append(
                        probs[0, score_pos]
                        .detach().cpu().float().item()
                    )

                question_scores.append(answer_scores)
            all_scores.append(question_scores)
        return all_scores

    def _score_batched(self, questions, outputs, batch_size):
        model = self.model
        tokenizer = self.tokenizer

        # Flatten all (question, answer) pairs into two parallel
        # lists of conversations: one with `+`, one with the marker.
        # We un-flatten back to list[list[list[float]]] at the end
        # using the per-question answer counts in `outputs`.
        flat_conversations = []
        flat_marker_conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conv, marker_conv = [], []
                step_texts = ans.split("\n\n")
                for step_idx, step in enumerate(step_texts):
                    text = (
                        question + " " + step if step_idx == 0 else step
                    )
                    conv.append({"role": "user", "content": text})
                    conv.append({"role": "assistant", "content": "+"})
                    marker_conv.append({"role": "user", "content": text})
                    marker_conv.append(
                        {"role": "assistant", "content": self.marker_text}
                    )
                flat_conversations.append(conv)
                flat_marker_conversations.append(marker_conv)

        flat_scores = []
        device = model.device
        for i in range(0, len(flat_conversations), batch_size):
            convs = flat_conversations[i : i + batch_size]
            marker_convs = flat_marker_conversations[i : i + batch_size]

            # apply_chat_template on a list of conversations returns a
            # list of formatted strings; tokenizer(...) then batch-
            # tokenizes them with padding. add_special_tokens=False
            # since the chat template already emits BOS.
            chat_texts = tokenizer.apply_chat_template(
                convs, tokenize=False,
            )
            marker_chat_texts = tokenizer.apply_chat_template(
                marker_convs, tokenize=False,
            )
            input_ids = tokenizer(
                chat_texts, return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).input_ids.to(device)
            marker_input_ids = tokenizer(
                marker_chat_texts, return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).input_ids.to(device)
            if input_ids.shape != marker_input_ids.shape:
                raise RuntimeError(
                    f"Batched shape mismatch: "
                    f"{input_ids.shape} vs "
                    f"{marker_input_ids.shape}"
                )

            with torch.no_grad():
                logits = model(input_ids).logits[
                    :, :, self.candidate_token_ids
                ]
                probs = logits.softmax(dim=-1)[:, :, 0]

            for b in range(len(convs)):
                # probs[b, :-1] is positions 0..T-2; a True at index
                # k of marker_input_ids[b, 1:] means a marker at
                # original token k+1, predicted by probs[b, k]. So
                # this gather collects every per-step P(+) in one shot.
                step_scores = probs[b, :-1][
                    marker_input_ids[b, 1:] == self.marker_token_id
                ].detach().cpu().float().tolist()
                flat_scores.append(step_scores)

        # Un-flatten back into list[list[list[float]]].
        reshaped = []
        cursor = 0
        for answers in outputs:
            reshaped.append(flat_scores[cursor : cursor + len(answers)])
            cursor += len(answers)
        return reshaped
