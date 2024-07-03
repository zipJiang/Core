""" Common utilities that can be applied at multiple places in the project. """


from typing import List, Any, Callable, TypeVar, Dict, Text, Union
from tqdm import tqdm


T = TypeVar("T")
ITEM = TypeVar("ITEM")
RESULT = TypeVar("RESULT")


def paginate_func(
    items: List[Any],
    page_size: int,
    func: Callable[..., T],
    combination: Callable[[List[T]], T],
    silent: bool = False
) -> T:
    
    results = []
    
    iterator = range(0, len(items), page_size)
    if not silent:
        iterator = tqdm(iterator, desc="Paginating")
        
    for i in iterator:
        results.append(
            func(
                items[i:i+page_size]
            )
        )
        
    return combination(results)


def run_on_valid(
    batch_func: Callable[[List[ITEM]], List[RESULT]],
    items: List[ITEM],
    validity_func: Callable[[ITEM], bool],
    filler: RESULT
) -> List[RESULT]:
    """Take a batch function and run it on the valid items in the list.
    And use filler_value for the invalid items.
    """
    
    input_tuples = [(idx, item) for idx, item in enumerate(items) if validity_func(item)]
    inputs = [item for _, item in input_tuples]
    partial_results = batch_func(inputs)
    
    results = [filler for _ in items]
    for (idx, _), result in zip(input_tuples, partial_results):
        results[idx] = result
        
    return results