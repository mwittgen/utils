# This file is part of utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

__all__ = (
    "FrozenGroupedSet",
    "GroupedSet",
    "MutableGroupedSet",
)

from collections import defaultdict
from typing import (
    AbstractSet,
    Any,
    Callable,
    Container,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    final,
)


_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V", bound=Hashable)
_I = TypeVar("_I", bound=Hashable)


class GroupedSet(Generic[_K, _V, _I]):

    __slots__ = ("groups", "_unpack", "_pack")

    def __init__(
        self,
        groups: Mapping[_K, AbstractSet[_V]],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ):
        self.groups = groups
        self._pack = pack
        self._unpack = unpack

    @staticmethod
    def from_iterable(
        iterable: Iterable[_I],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ) -> GroupedSet[_K, _V, _I]:
        result = MutableGroupedSet[_K, _V, _I]({}, pack, unpack)
        for item in iterable:
            result.add(item)
        return result

    def __contains__(self, item: _I) -> bool:
        k, v = self._unpack(item)
        if (values := self.groups.get(k)) is None:
            return False
        else:
            return v in values

    def __iter__(self) -> Iterator[_I]:
        for key, values in self.groups.items():
            for v in values:
                yield self._pack(key, v)

    def __len__(self) -> int:
        return sum(len(values) for values in self.groups.values())

    def __eq__(self, other: Any) -> bool:
        ...

    def __getitem__(self, key: _K) -> AbstractSet[_V]:
        return self.groups[key]

    def issubset(self, other: Container[_I]) -> bool:
        return all(item in other for item in self)

    def issuperset(self, other: Iterable[_I]) -> bool:
        return all(item in self for item in other)

    def isdisjoint(self, other: Iterable[_I]) -> bool:
        return not any(item in self for item in other)

    def union(self, *others: Iterable[_I]) -> GroupedSet[_K, _V, _I]:
        result = self.copy().mutable()
        result.update(*others)
        return result

    def intersection(self, *others: Iterable[_I]) -> GroupedSet[_K, _V, _I]:
        result = self.copy().mutable()
        result.intersection_update(*others)
        return result

    def copy(self) -> GroupedSet[_K, _V, _I]:
        """Return a set with the same contents as ``self``.

        Returns
        -------
        copy : `GroupedSet`
            A set that is guaranteed not to have any other owners that could
            change its contents.  This may actually be ``self`` for immutable
            derived classes, since those can never have any owners that can
            change their contents.
        """
        return MutableGroupedSet(self.groups, self._pack, self._unpack)

    def mutable(self) -> MutableGroupedSet[_K, _V, _I]:
        """Return ``self`` if ``self`` is a `MutableGroupedSet`, or a
        mutable copy otherwise.

        Returns
        -------
        mutable : `MutableGroupedSet`
            A mutable set.
        """
        return MutableGroupedSet(self.groups, self._pack, self._unpack)

    def frozen(self) -> FrozenGroupedSet[_K, _V, _I]:
        """Return ``self`` if ``self`` is a `FrozenGroupedSet`, or an
        immutable copy otherwise.

        Returns
        -------
        frozen : `FrozenGroupedSet`
            An immutable set.
        """
        return FrozenGroupedSet(self.groups, self._pack, self._unpack)


@final
class FrozenGroupedSet(GroupedSet[_K, _V, _I]):

    __slots__ = ()

    groups: Mapping[_K, frozenset[_V]]

    def __init__(
        self,
        groups: Mapping[_K, AbstractSet[_V]],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ):
        super().__init__({k: frozenset(v) for k, v in groups.items()}, pack, unpack)

    @staticmethod
    def from_iterable(
        iterable: Iterable[_I],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ) -> FrozenGroupedSet[_K, _V, _I]:
        return GroupedSet[_K, _V, _I].from_iterable(iterable, pack, unpack).frozen()

    def __getitem__(self, key: _K) -> frozenset[_V]:
        return self.groups[key]

    def union(self, *others: Iterable[_I]) -> FrozenGroupedSet[_K, _V, _I]:
        return super().union(*others).frozen()

    def intersection(self, *others: Iterable[_I]) -> FrozenGroupedSet[_K, _V, _I]:
        return super().intersection(*others).frozen()

    def copy(self) -> FrozenGroupedSet[_K, _V, _I]:
        return self

    def frozen(self) -> FrozenGroupedSet[_K, _V, _I]:
        return self


@final
class MutableGroupedSet(GroupedSet[_K, _V, _I]):

    __slots__ = ()

    groups: defaultdict[_K, set[_V]]

    def __init__(
        self,
        groups: Mapping[_K, AbstractSet[_V]],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ):
        super().__init__(defaultdict(set, {k: set(v) for k, v in groups.items()}), pack, unpack)

    @staticmethod
    def from_iterable(
        iterable: Iterable[_I],
        pack: Callable[[_K, _V], _I],
        unpack: Callable[[_I], tuple[_K, _V]],
    ) -> MutableGroupedSet[_K, _V, _I]:
        return GroupedSet[_K, _V, _I].from_iterable(iterable, pack, unpack).mutable()

    def __getitem__(self, key: _K) -> set[_V]:
        return self.groups[key]

    def union(self, *others: Iterable[_I]) -> MutableGroupedSet[_K, _V, _I]:
        return super().union(*others).mutable()

    def intersection(self, *others: Iterable[_I]) -> MutableGroupedSet[_K, _V, _I]:
        return super().intersection(*others).mutable()

    def update(self, *others: Iterable[_I]) -> None:
        for other in others:
            if (groups := getattr(other, "groups", None)) is not None:
                for k, v in groups.items():
                    self.groups[k].add(v)
            else:
                for item in other:
                    k, v = self._unpack(item)
                    self.groups[k].add(v)

    def intersection_update(self, *others: Iterable[_I]) -> None:
        for other in others:
            if (groups := getattr(other, "groups", None)) is None:
                groups = GroupedSet[_K, _V, _I].from_iterable(other, self._pack, self._unpack).groups
            for k in self.groups.keys() - groups.keys():
                del self.groups[k]
            for k in self.groups.keys() & groups.keys():
                values = self.groups[k]
                values.intersection_update(groups[k])
                if not values:
                    del self.groups[k]

    def add(self, item: _I) -> None:
        k, v = self._unpack(item)
        self.groups[k].add(v)

    def discard(self, item: _I) -> None:
        k, v = self._unpack(item)
        values = self.groups[k]
        values.discard(v)
        if not values:
            del self.groups[k]

    def remove(self, item: _I) -> None:
        k, v = self._unpack(item)
        values = self.groups[k]
        try:
            values.remove(v)
        finally:
            if not values:
                del self.groups[k]

    def clear(self) -> None:
        self.groups.clear()

    def copy(self) -> MutableGroupedSet[_K, _V, _I]:
        return super().copy().mutable()

    def mutable(self) -> MutableGroupedSet[_K, _V, _I]:
        return self
