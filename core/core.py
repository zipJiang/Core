from typing import (
    Optional,
    TypeVar,
    Callable,
    List,
    Text,
    Tuple,
    Dict,
    ParamSpec,
)
from functools import partial
import numpy as np
import torch
from scipy.optimize import milp, Bounds, LinearConstraint
from .utils.instances import (
    ScorerInstance,
    CoreInstance,
    EntailerInstance,
)
from .utils.constants import __EPSILON__
from .utils.common import run_on_valid
from .scorers import Scorer, ConstantScorer, UNLIConfidenceBoostScorer
from .entailers import Entailer, SoftEntailer


_R = TypeVar("_R")
_P = ParamSpec("_P")


def default_result_parser(_rs: List[Text]) -> List[CoreInstance]:
    """ """
    return [
        CoreInstance(
            text=result,
            sent=None,
            topic=None,
        )
        for result in _rs
    ]


def default_result_merger(
    _selected: List[int], _inputs: List[CoreInstance], _rs: List[Text]
) -> List[Text]:
    """ """
    return [_inputs[idx].text for idx in _selected]


def Core(
    result_parser: Optional[Callable[[_R], List[CoreInstance]]] = None,
    result_merger: Optional[
        Callable[[List[int], List[CoreInstance], _R], _R]
    ] = None,
    sentence_level_checkworthy_scorer: Optional[Scorer] = None,
    claim_level_checkworthy_scorer: Optional[Scorer] = None,
    score_combinator: Optional[Callable[[float, float], float]] = None,
    overwrite_entailer: Optional[Entailer] = None,
    cache_dir: Optional[Text] = None,
    silent: bool = True,
):
    """Decorator factory for the Core deduplication algorithm. Add `@Core(...)` to your decomposition function with corresponding parameters to get a Core subselected decomposition.

    Parameters:
    -----------
    result_parser (Optional[Callable[[_R], List[CoreInstance]]): A function that takes the result and parse it into a list of `CoreInstance` that can be processed by Core.
    result_merger (Optional[Callable[[List[CoreInstance], _R], _R]): A function that takes the selected instances and merge it back to the original result.
    sentence_level_checkworthy_scorer (Optional[Scorer]): A scorer that scores the sentence level checkworthiness, defaults to `ConstantScorer(score=1.0)`.
    claim_level_checkworthy_scorer (Optional[Scorer]): A scorer that scores the claim level checkworthiness, defaults to `ConstantScorer(score=1.0)`.
    overwrite_entailer (Optional[Scorer]): A scorer that overwrites the default entailer, defaults to `None`, in this case, the default entailer is used which is NLI.
    """

    entailer = (
        overwrite_entailer
        if overwrite_entailer is not None
        else (
            Entailer(
                model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                # We select the device based on the availability of a GPU,
                # We leave it here as a placeholder so that people can easily,
                # overwrite with Entailer using different devices.
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                internal_batch_size=256,
                max_length=256,
                cache_dir=cache_dir,
            )
        )
    )

    if result_parser is None:
        result_parser = default_result_parser
    if result_merger is None:
        result_merger = default_result_merger

    if sentence_level_checkworthy_scorer is None:
        sentence_level_checkworthy_scorer = ConstantScorer(score=1.0)

    if claim_level_checkworthy_scorer is None:
        claim_level_checkworthy_scorer = ConstantScorer(score=1.0)

    if score_combinator is None:
        score_combinator = lambda a, b: a * b - __EPSILON__

    # solve the MILP problem
    def _solve_milp(
        pairwise_entailment: np.ndarray,
        weighting: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """ """

        if pairwise_entailment.size == 0:
            return np.array([], np.int16), 0.0

        or_mat = np.tril(
            np.bitwise_or(pairwise_entailment, np.transpose(pairwise_entailment)),
        )

        indices = np.nonzero(or_mat)

        # TODO: add logging information

        constraints = np.zeros((len(indices[0]), len(weighting)), dtype=np.float32)

        constraints[np.arange(len(indices[0])), indices[0]] = 1
        constraints[np.arange(len(indices[1])), indices[1]] = 1

        res = milp(
            c=-weighting,
            integrality=np.ones_like(weighting),
            bounds=Bounds(
                lb=np.zeros_like(weighting) - 1e-8, ub=np.ones_like(weighting) + 1e-8
            ),
            constraints=(
                LinearConstraint(
                    A=constraints,
                    ub=np.ones(len(indices[0])) + 1e-8,
                ),
            ),
        )

        selection = res.x
        result = res.fun

        return selection, result

    def _deduplicate(instances: List[CoreInstance]) -> List[int]:
        """Takes in a list of instances and do the deduplication."""
        
        sent_checkworthy_instances = [
            ScorerInstance(text=instance.sent, topic=instance.topic)
            for instance in instances
        ]
        sent_checkworthy_scores = run_on_valid(
            batch_func=partial(sentence_level_checkworthy_scorer.__call__, silent=silent),
            items=sent_checkworthy_instances,
            validity_func=lambda instance: instance.text is not None,
            filler=1.0,
        )

        claim_checkworthy_instances = [
            ScorerInstance(text=instance.text, topic=instance.topic)
            for instance in instances
        ]
        claim_checkworthy_scores = run_on_valid(
            batch_func=partial(claim_level_checkworthy_scorer.__call__, silent=silent),
            items=claim_checkworthy_instances,
            validity_func=lambda instance: instance.text is not None,
            filler=1.0,
        )

        sent_ent_instances = [
            EntailerInstance(premise=instance.sent, hypothesis=instance.text)
            for instance in instances
        ]
        sent_ent_results = run_on_valid(
            batch_func=partial(entailer.__call__, silent=silent),
            items=sent_ent_instances,
            validity_func=lambda instance: instance.premise is not None,
            filler=1.0,
        )
        
        # filter out claims that are not entailed
        # TODO: Check whether this binarizer should be optional
        instances_wreal_idx = [
            (tidx, instance)
            for tidx, (instance, entailed) in enumerate(
                zip(instances, sent_ent_results)
            )
            if entailed > 0.5
        ]

        # create pairwise entailment instances
        # if not in the result is 1
        finding_pair: Dict[Tuple[int, int], int] = {}
        pairwise_entailment_instances = []

        for i in range(len(instances_wreal_idx)):
            for j in range(len(instances_wreal_idx)):
                if i == j:
                    continue
                elif instances_wreal_idx[i][1].text == instances_wreal_idx[j][1].text:
                    finding_pair[(i, j)] = -1  # automatic entailment
                else:
                    finding_pair[(i, j)] = len(pairwise_entailment_instances)
                    pairwise_entailment_instances.append(
                        EntailerInstance(
                            premise=instances_wreal_idx[i][1].text,
                            hypothesis=instances_wreal_idx[j][1].text,
                        )
                    )

        pairwise_entailment_scoring = entailer(pairwise_entailment_instances, silent=silent)

        intra_ent_mat = np.array(
            [
                [
                    (
                        (
                            # TODO: Check whether this threshold is optimal
                            pairwise_entailment_scoring[finding_pair[(i, j)]] > 0.5
                            if finding_pair[(i, j)] >= 0
                            else True
                        )
                        if i != j
                        else False
                    )
                    for j in range(len(instances_wreal_idx))
                ]
                for i in range(len(instances_wreal_idx))
            ],
            dtype=np.int16,
        )

        weighting = np.array(
            [
                score_combinator(
                    sent_checkworthy, claim_checkworthy
                )  # This penalize non-checkworthy claims
                for sent_checkworthy, claim_checkworthy in zip(
                    sent_checkworthy_scores, claim_checkworthy_scores
                )
            ],
            np.float32,
        )

        selection, _ = _solve_milp(
            pairwise_entailment=intra_ent_mat, weighting=weighting
        )

        non_zero_selection_indices = sorted(np.nonzero(selection)[0].tolist())
        non_zero_selection_indices = [
            instances_wreal_idx[idx][0] for idx in non_zero_selection_indices
        ]

        # use the non_zero_selection_indices to back translate to the original instances
        return non_zero_selection_indices

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        """ """

        def wrapper(*args: _P.args, **kwargs: _P.kwargs):
            results = func(*args, **kwargs)

            inputs = result_parser(results)
            selected = _deduplicate(inputs)

            return result_merger(selected, inputs, results)

        return wrapper

    return decorator
