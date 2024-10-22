""" This score scores a given claim by how much confidence boost stating that claim
explicitly gives to UNLI being able to hypothetically predict the claim. """

from typing import Text, Dict, List, Union, Optional
import numpy as np
from ..entailers.entailer import Entailer, EntailerInstance
from ..utils.instances import ScorerInstance
from .scorer import Scorer


class UNLIConfidenceBoostScorer(Scorer):
    """-log p(claim | bleached_context)"""

    __NAME__ = "unli-confidence-boost"

    def __init__(
        self,
        bleached_templates: List[Text],
        entailer: Entailer,
        cap_entailer: Optional[Entailer] = None,
    ):
        """We don't explicitly require the entailer to
        be soft, but practically people should always use a
        soft entailer for proper tie-breaking.
        """

        super().__init__()
        
        self._bleached_templates = bleached_templates
        self._entailer = entailer
        self._cap_entailer = cap_entailer

    def _score(self, instance: ScorerInstance, silent: bool = False) -> Dict[Text, Union[Text, float]]:
        """Here the scorer will score the instance into
        result dict
        """

        bleached_context = [bt.format(topic=instance.topic) for bt in self._bleached_templates]

        # pair each bleached_context with the claim
        # and score each pair

        inputs = [
            EntailerInstance(premise=bc, hypothesis=instance.text)
            for bc in bleached_context
        ]

        scores = self._entailer(inputs, silent=silent)
        # Adding lowerbound to cap_score to avoid log(0)
        score = max(*scores)

        cap_entailer_outputs = {}

        if self._cap_entailer is not None:
            # if cap_score == 1, then the claim needs to be
            # capped regardless of the score
            cap_scores = self._cap_entailer(inputs, silent=silent)
            cap_score = max(cap_scores)
            score = max(score, cap_score)

            cap_entailer_outputs["cap_scores"] = cap_scores

        # Zhengping 05/24/2025: Use - log(score) to align with CPMI
        parsed_score = (- np.log(score)).item()

        return {
            "premises": bleached_context,
            "hypothesis": instance.text,
            "parsed": parsed_score,
            "scores": scores,
            **cap_entailer_outputs,
        }

    def _batch_score(
        self, instances: List[ScorerInstance], silent: bool = False
    ) -> List[Dict[Text, Text | float]]:
        """Run scores in batch."""
        
        inputs = [
            EntailerInstance(premise=bt.format(topic=instance.topic), hypothesis=instance.text)
            for instance in instances
            for bt in self._bleached_templates
        ]

        all_scores = self._entailer(inputs, silent=silent)
        all_scores = [
            all_scores[
                i * len(self._bleached_templates) : (i + 1) * len(self._bleached_templates)
            ]
            for i in range(len(instances))
        ]

        all_cap_entailer_outputs = [{}] * len(instances)

        if self._cap_entailer is not None:
            cap_scores = self._cap_entailer(inputs, silent=silent)
            all_cap_entailer_outputs = [
                {
                    "cap_scores": cap_scores[
                        i
                        * len(self._bleached_templates) : (i + 1)
                        * len(self._bleached_templates)
                    ]
                }
                for i in range(len(instances))
            ]

        return [
            {
                "premises": [bt.format(topic=instance.topic) for bt in self._bleached_templates],
                "hypothesis": instance.text,
                "parsed": (
                    # 1 - self._epsilon - max(scores)
                    - np.log(max(*scores))
                    if "cap_scores" not in cap_entailer_outputs
                    else - np.log(max(*scores, *cap_entailer_outputs["cap_scores"]))
                ).item(),
                "scores": scores,
                **cap_entailer_outputs,
            }
            for instance, scores, cap_entailer_outputs in zip(
                instances, all_scores, all_cap_entailer_outputs
            )
        ]
