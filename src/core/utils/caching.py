""" Define the cahing mechanism for entailer calling """

import sqlite3
import logging
import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Text,
    List,
    Union,
    Tuple,
    Optional
)


logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


class EntailmentCache:
    """ """
    def __init__(
        self,
        cache_dir: Text,
    ):
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        self._cache_conn = sqlite3.connect(
            os.path.join(self._cache_dir, "entailment_cache.db")
        )
        self._cursor = self._cache_conn.cursor()
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entailment_cache (
                premise TEXT,
                hypothesis TEXT,
                model_name TEXT,
                entailment_score REAL
            )
            """
        )

    def query(
        self,
        premise: Text,
        hypothesis: Text,
        model_name: Text
    ) -> Optional[float]:
        self._cursor.execute(
            """
            SELECT entailment_score FROM entailment_cache
            WHERE premise = ? AND hypothesis = ? AND model_name = ?
            """,
            (premise, hypothesis, model_name)
        )
        result = self._cursor.fetchone()
        return result[0] if result else None
    
    def insert(
        self,
        premises: List[Text],
        hypotheses: List[Text],
        model_names: Union[Text, List[Text]],
        entailment_scores: List[float]
    ):
        """ """

        if isinstance(model_names, Text):
            model_names = [model_names] * len(premises)

        assert len(premises) == len(hypotheses) == len(model_names) == len(entailment_scores), "All lists must have the same length."

        insertion_dict = {}

        for premise, hypothesis, model_name, entailment_score in zip(premises, hypotheses, model_names, entailment_scores):
            if (premise, hypothesis, model_name) in insertion_dict:
                if insertion_dict[(premise, hypothesis, model_name)] != entailment_score:
                    logger.warning(
                        f"Duplicate entries for premise: {premise}, hypothesis: {hypothesis}, model_name: {model_name} (inconsistent)."
                    )
                continue
            insertion_dict[(premise, hypothesis, model_name)] = entailment_score
            
        data = []

        for (premise, hypothesis, model_name), entailment_score in insertion_dict.items():
            data.append((premise, hypothesis, model_name, entailment_score))

        self._cursor.executemany(
            """INSERT OR IGNORE INTO entailment_cache (premise, hypothesis, model_name, entailment_score)
            VALUES (?, ?, ?, ?)""", data
        )
        self._cache_conn.commit()
        
    def __del__(self):
        self._cache_conn.close()