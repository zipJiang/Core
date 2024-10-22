from typing import Text, List, Union, Dict, AsyncGenerator, Tuple, Optional
from .scorer import Scorer
from ..utils.instances import ScorerInstance


class ConstantScorer(Scorer):
    
    __NAME__ = "constant"
    
    def __init__(self, score: Optional[float] = 1.0):
        """
        """
        
        super().__init__()
        self._default_score = score
        
    def _score(self, instance: ScorerInstance, silent: bool = False) -> Dict[Text, Union[Text, float]]:
        return {"parsed": self._default_score, "raw": None}
