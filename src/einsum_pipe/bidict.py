from typing import Dict, Generic, Iterator, TypeVar
from collections.abc import MutableMapping


K = TypeVar('K')
V = TypeVar('V')


class _BiDict(MutableMapping, Generic[K, V]):
    """A custom dictionary that keeps track of the inverse mapping from values to keys. Values must be unique too
    """

    def __init__(self):
        self.store: Dict[K, V] = {}
        self.inverse: Dict[V, K] = {}

    def __getitem__(self, key: K) -> V:
        return self.store[key]

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            del self.inverse[self[key]]
        self.inverse[value] = key
        self.store[key] = value

    def __delitem__(self, key: K) -> None:
        del self.inverse[self[key]]
        del self.store[key]

    def values(self):
        return self.inverse.keys()

    def __iter__(self) -> Iterator[K]:
        return iter(self.store)

    def __len__(self):
        return len(self.store)
