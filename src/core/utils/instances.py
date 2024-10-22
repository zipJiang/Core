"""
"""

from dataclasses import dataclass, field
from typing import Text, Union, Optional, List


@dataclass(frozen=True, eq=True)
class Instance:
    def __iter__(self):
        for field in self.__dataclass_fields__:
            yield (field, getattr(self, field))


@dataclass(frozen=True, eq=True)
class ScorerInstance(Instance):
    text: Text
    topic: Union[None, Text]


@dataclass(frozen=True, eq=True)
class EntailerInstance:
    premise: Text
    hypothesis: Text


@dataclass(frozen=True, eq=True)
class CoreInstance(ScorerInstance):
    sent: Union[None, Text]