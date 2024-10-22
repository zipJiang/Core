""" Define the cahing mechanism for entailer calling """

import sqlite3
import hashlib
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
                hash_key TEXT PRIMARY KEY,
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
        
        hash_key = self.hash_query(premise, hypothesis, model_name)

        self._cursor.execute(
            """
            SELECT entailment_score FROM entailment_cache
            WHERE hash_key = ?
            """,
            (hash_key,)
        )
        result = self._cursor.fetchone()
        return result[0] if result else None
    
    def hash_query(
        self,
        premise: Text,
        hypothesis: Text,
        model_name: Text
    ) -> Text:
        """ """
        return hashlib.md5(f"{premise}::{hypothesis}::{model_name}".encode()).hexdigest()
    
    def batch_query(
        self,
        premises: List[Text],
        hypotheses: List[Text],
        model_names: Union[Text, List[Text]]
    ) -> List[Optional[float]]:
        
        # query at once, if not found, return None
        if isinstance(model_names, Text):
            model_names = [model_names] * len(premises)
            
        assert len(premises) == len(hypotheses) == len(model_names), "All lists must have the same length."

        hash_keys = [self.hash_query(premise, hypothesis, model_name) for premise, hypothesis, model_name in zip(premises, hypotheses, model_names)]

        # select both hash_key and entailment_score, then filter out the ones not found
        self._cursor.execute(
            """
            SELECT hash_key, entailment_score FROM entailment_cache
            WHERE hash_key IN ({})
            """.format(", ".join(["?"] * len(hash_keys))),
            hash_keys
        )

        results = {hash_key: score for hash_key, score in self._cursor.fetchall()}
        return [results.get(hash_key, None) for hash_key in hash_keys]
    
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
            hash_key = self.hash_query(premise, hypothesis, model_name)
            if hash_key in insertion_dict:
                if insertion_dict[hash_key] != entailment_score:
                    logger.warning(
                        f"Duplicate entries for premise: {premise}, hypothesis: {hypothesis}, model_name: {model_name} (inconsistent)."
                    )
                continue
            insertion_dict[hash_key] = entailment_score
            
        self._cursor.executemany(
            """INSERT OR IGNORE INTO entailment_cache (hash_key, entailment_score)
            VALUES (?, ?)""", 
            [(key, value) for key, value in insertion_dict.items()]
        )
        self._cache_conn.commit()
        
    def __del__(self):
        self._cache_conn.close()