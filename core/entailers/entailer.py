"""Entailer will be used to judge the entailment relationship between two setences"""

from typing import Text
import torch
from timeit import timeit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import Text, Dict, List
from ..utils.common import paginate_func
from ..utils.instances import EntailerInstance


class Entailer:
    __LABEL_MAP__ = [1, 0, 0]

    def __init__(
        self,
        model_name: Text,
        device: Text = "cuda",
        internal_batch_size: int = 16,
        max_length: int = 512,
    ):
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._internal_batch_size = internal_batch_size
        self._max_length = max_length
        
    def _load_model(self):
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def _collate_fn(
        self, instances: List[EntailerInstance]
    ) -> Dict[Text, torch.Tensor]:
        """Notice that we are requiring this to run entailer instances."""

        premises = [instance.premise for instance in instances]
        hypotheses = [instance.hypothesis for instance in instances]
        
        tokenized = self._tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        return {
            "input_ids": tokenized["input_ids"].to(self._device),
            "attention_mask": tokenized["attention_mask"].to(self._device),
            "token_type_ids": (
                tokenized["token_type_ids"].to(self._device)
                if "token_type_ids" in tokenized
                else None
            ),
        }

    def __call__(
        self,
        instances: List[EntailerInstance],
        silent: bool = False
    ) -> List[float]:
        """ Lazy load the model and tokenizer to be friendly to
        parallel processing in our case.
        """

        if self._model is None or self._tokenizer is None:
            self._load_model()

        return paginate_func(
            items=instances,
            page_size=self._internal_batch_size,
            func=self._call_batch,
            combination=lambda x: [xxx for xx in x for xxx in xx],
            silent=silent
        )

    def _call_batch(self, instances: List[EntailerInstance]) -> List[float]:
        """This is the actual calling function of the model."""

        assert len(instances) <= self._internal_batch_size, "Batch size is too large."

        with torch.no_grad():
            inputs = self._collate_fn(instances)
            outputs = self._model(**inputs)

        indices = torch.argmax(outputs.logits, dim=1).int().cpu().numpy().tolist()

        return [float(self.__LABEL_MAP__[index]) for index in indices]