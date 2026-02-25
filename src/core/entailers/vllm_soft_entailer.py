"""VLLM-backed soft entailer for conditional probability estimation.

Uses a VLLM-served LLM (e.g. Zhengping/conditional-probability-regression)
that estimates p(hypothesis | premise) by decoding a distribution over
special label-level tokens and computing a weighted average score.
"""

import math
from typing import Text, List, Optional, Dict

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .entailer import Entailer
from ..utils.instances import EntailerInstance


class VLLMSoftEntailer(Entailer):
    """Soft entailer backed by a VLLM OpenAI-compatible API endpoint.

    The hosted model is expected to use special ``<|label_level_N|>`` tokens
    whose softmax-weighted midpoint scores yield a probability in [0, 1].
    This is the inference protocol used by
    ``Zhengping/conditional-probability-regression``.
    """

    _PROMPT_TEMPLATE = (
        '### Question: Given the premise "{premise}", '
        'how likely is it that the hypothesis "{hypothesis}" is true?\n\n'
    )
    _COMPLETION_PREFIX = "### Answer:"

    def __init__(
        self,
        model_name: Text,
        api_base: Text = "http://localhost:8000/v1",
        num_labels: int = 10,
        internal_batch_size: int = 16,
        cache_dir: Optional[Text] = None,
        top_logprobs: int = 20,
        api_key: Text = "EMPTY",
    ):
        super().__init__(
            model_name=model_name,
            device="cpu",
            internal_batch_size=internal_batch_size,
            max_length=512,
            cache_dir=cache_dir,
        )
        self._api_base = api_base.rstrip("/")
        self._num_labels = num_labels
        self._top_logprobs = max(top_logprobs, num_labels)

        self._client = OpenAI(
            base_url=self._api_base,
            api_key=api_key,
        )

        # Pre-compute label token strings and their midpoint score values.
        # Token format mirrors the vocabulary of the target model where each
        # ``<|label_level_i|>`` token is mapped to the midpoint of the i-th
        # uniform bin over [0, 1].
        self._label_tokens: List[Text] = [
            f" <|label_level_{i}|>" for i in range(num_labels)
        ]
        step_size = 1.0 / num_labels
        self._label_scores: List[float] = [
            i * step_size + 0.5 * step_size for i in range(num_labels)
        ]

    # ------------------------------------------------------------------
    # Override base-class hooks
    # ------------------------------------------------------------------

    def _load_model(self):
        """No local model to load — set sentinels so the base ``__call__``
        does not attempt to reload on every invocation."""
        self._model = "vllm"
        self._tokenizer = "vllm"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        """Numerically-stable softmax over a list of logprobs."""
        max_val = max(values)
        exps = [math.exp(v - max_val) for v in values]
        total = sum(exps)
        return [e / total for e in exps]

    def _extract_score(self, completion: ChatCompletion) -> float:
        """Compute the weighted-average probability from a chat completion.

        1. Collect the log-probabilities of every ``<|label_level_*|>``
           token that appears in the ``top_logprobs`` of the first
           generated token.
        2. Apply softmax **only** over those label tokens.
        3. Return the dot product with the pre-computed midpoint scores.
        """
        choice = completion.choices[0]

        if choice.logprobs is None or not choice.logprobs.content:
            return 0.5

        first_token_info = choice.logprobs.content[0]

        # Map token string → logprob from the top_logprobs list
        token_logprob_map: Dict[Text, float] = {
            entry.token: entry.logprob
            for entry in (first_token_info.top_logprobs or [])
        }

        # Look up each label token; use a very negative value for any
        # label that did not appear in the top-k.
        label_logprobs = [
            token_logprob_map.get(tok, -100.0) for tok in self._label_tokens
        ]

        probs = self._softmax(label_logprobs)
        score = sum(p * s for p, s in zip(probs, self._label_scores))
        return score

    # ------------------------------------------------------------------
    # Core batch call
    # ------------------------------------------------------------------

    def _call_batch(
        self, instances: List[EntailerInstance]
    ) -> List[float]:
        """Query the VLLM server for each instance and return scores."""
        scores: List[float] = []

        for instance in instances:
            messages = [
                {
                    "role": "user",
                    "content": self._PROMPT_TEMPLATE.format(
                        premise=instance.premise,
                        hypothesis=instance.hypothesis,
                    ),
                },
                {
                    "role": "assistant",
                    "content": self._COMPLETION_PREFIX,
                },
            ]

            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=self._top_logprobs,
                temperature=0,
                extra_body={
                    # vLLM-specific: continue from the assistant prefix
                    # rather than starting a new assistant turn.
                    "continue_final_message": True,
                },
            )
            scores.append(self._extract_score(completion))

        return scores
