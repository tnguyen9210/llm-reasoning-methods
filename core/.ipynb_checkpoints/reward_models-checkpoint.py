import gc
from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.config import Config
from sal.models.skywork_o1_prm.io_utils import (
    derive_step_rewards,
    prepare_batch_input_for_model,
    prepare_input,
)
from sal.models.skywork_o1_prm.prm_model import SkyworkPRMModel

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


class PRM:
    def __init__(self, model_path, device_map, **model_kwargs):
        self.model_path = model_path
        self.device_map = device_map
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError
        

class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        # model_id = "QuantFactory/Llama3.1-8B-PRM-Deepseek-Data-GGUF"
        # gguf_file = "Llama3.1-8B-PRM-Deepseek-Data.Q4_K_M.gguf"
        # tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)
        # model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=gguf_file).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch.float16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=8,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to("cuda")
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        device = self.model.device
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class SkyworkO1(PRM):
    # @classmethod
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = SkyworkPRMModel.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()

        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        # reference code: https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B#huggingface-inference
        all_scores = []
        for question, answers in zip(questions, outputs):
            processed_data = [
                prepare_input(
                    question, answer, tokenizer=self.tokenizer, step_token="\n"
                )
                for answer in answers
            ]
            input_ids, steps, reward_flags = zip(*processed_data)
            print([len(input_ids[i]) for i in range(len(input_ids))])
            # print(len(steps))
            input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(
                input_ids, reward_flags, self.tokenizer.pad_token_id
            )
            print(input_ids.shape)
            print(attention_mask.shape)
            device = self.model.pretrained_model.device
            device = "cuda"
            # print(device)
            inputs_id = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # print(len(input_ids))
            # print(len(steps))
            print('#--- memory:', torch.cuda.memory_allocated(0)/(1024**3))
            print('#--- memory:', torch.cuda.memory_allocated(1)/(1024**3))
            print('#--- memory:', torch.cuda.memory_allocated(2)/(1024**3))
            print('#--- memory:', torch.cuda.memory_allocated(3)/(1024**3))
            with torch.no_grad():
                _, _, rewards = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_probs=True,
                )
                # _, _, rewards = self.model(
                #     input_ids=input_ids.to(device),
                #     attention_mask=attention_mask.to(device),
                #     return_probs=True,
                # )
                stop
                all_step_scores = derive_step_rewards(
                    rewards.detach().to("cpu", dtype=torch.float32), reward_flags
                )
            all_scores.append(all_step_scores)
            # del(input_ids)
            # del(attention_mask)
            # del(steps)
            # del(reward_flags)
            # del(processed_data)
            torch.cuda.empty_cache()
        return all_scores


class SkyworkO1_1_5B(SkyworkO1):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        return SkyworkO1._load_model_and_tokenizer(prm_model_path, **model_kwargs)


class SkyworkO1_7B(SkyworkO1):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
        return SkyworkO1._load_model_and_tokenizer(prm_model_path, **model_kwargs)

