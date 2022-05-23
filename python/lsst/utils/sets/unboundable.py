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
    "FrozenUnboundableSet",
    "MutableUnboundableSet",
    "UnboundableSet",
)

from collections import defaultdict
from typing import AbstractSet, Any, ClassVar, Generic, Hashable, Iterable, TypeVar, Union, cast, final

from .ellipsis import Ellipsis, EllipsisType

_T = TypeVar("_T", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class UnboundableSet(Generic[_T]):
    """A set-like class with a special state representing "all elements" in a
    abstract sense.

    Parameters
    ----------
    values : `AbstractSet` or ``...``
        Either a true set-like object containing the values in the object or
        ``...`` to indicate all possible values.  This is not copied, so future
        changes to ``values`` will the `UnboundedSet`.

    Notes
    -----
    This class is not abstract, but in most contexts it is best to treat it as
    an interface, but actually construct `FrozenUnboundableSet` or
    `MutableUnboundableSet` instead.  The exception is when an `UnboundableSet`
    view into into some other set-like type (like a dictionary's `KeysView`) is
    desired.

    This class does not inherit from `AbstractSet` itself because it is not
    always iterable, sized, or capable of removing items.  Its set-operation
    methods accept `AbstractSet` instances and ``...`` in addition to
    `UnboundableSet` instances for convenience, but note that calling the same
    methods on true set-like objects with `UnboundableSet` arguments does not
    work.  Regular named methods (e.g. `issubset`, `union`) are used
    exclusively by `UnboundableSet`, instead of providing operator support as
    well (e.g.  ``<=``, ``|``) in order to mitigate confusion about this lack
    of reversibility.
    """

    __slots__ = ("values",)

    def __init__(self, values: Union[AbstractSet[_T], EllipsisType]):
        self.values = values

    values: Union[AbstractSet[_T], EllipsisType]
    """The set-like object or sentinel ``...`` backing the `UnboundableSet`.
    """

    # Most set-returning methods on UnboundableSet actually return
    # MutableUnboundableSet instances but annotate the return type with the
    # base class.  That annotation is necessary to allow other subclasses (like
    # FrozenUnboundableSet) to return instances of their own type, as is
    # usually expected.  Actually returning MutableUnboundableSet is not
    # necessary (and should be considered an implementation detail), but it
    # often what's used internally in the base class implementation, and doing
    # it across the board is good for consistency.

    @classmethod
    def make_empty(cls) -> UnboundableSet[_T]:
        """Construct an `UnboundableSet` that contains no values.

        Returns
        -------
        empty : `UnboundableSet`
            Empty set.
        """
        return MutableUnboundableSet(set())

    @classmethod
    def make_full(cls) -> UnboundableSet[_T]:
        """Construct an `UnboundableSet` that contains all possible values.

        Returns
        -------
        empty : `UnboundableSet`
            Full set.
        """
        return MutableUnboundableSet(Ellipsis)

    @classmethod
    def coerce(cls, other: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> UnboundableSet[_T]:
        """Coerce the given set-like object to an `UnboundableSet`.

        Parameters
        ----------
        other : `AbstractSet`, ``...```, or `UnboundableSet`
            Set-like object to coerce.

        Returns
        -------
        set : `UnboundableSet`
            An unboundable set instance.  May be the same instance as ``other``
            if that is already an `UnboundableSet`.
        """
        # Implementation relies on the fact that _extract_values passes through
        # anything but an UnboundableSet unchanged, and the assumption made
        # there that of they types we could be passed, only UnboundableSet has
        # a .values attributes.
        if cls._extract_values(other) is other:
            return UnboundableSet(cast(AbstractSet[_T] | EllipsisType, other))
        else:
            return cast(UnboundableSet, other)

    @staticmethod
    def _extract_values(
        other: AbstractSet[Any] | EllipsisType | UnboundableSet[Any],
    ) -> AbstractSet[Any] | EllipsisType:
        """Return the ``.values`` attribute of an `UnboundableSet` if the
        given object is one, or the given object itself if it is not.
        """
        # MyPy has to worry about the possibility of an `AbstractSet` that
        # also happens to have a `.values` attribute of some arbitrary type,
        # but here we're content to assume that anything passed here with that
        # attribute is an `UnboundableSet`.
        return cast(Union[AbstractSet[_T], EllipsisType], getattr(other, "values", other))

    def __contains__(self, key: _T) -> bool:
        return self.values is Ellipsis or key in self.values

    def __bool__(self) -> bool:
        return self.values is Ellipsis or bool(self.values)

    def __eq__(self, other: Any) -> bool:
        other_values = self._extract_values(other)
        if other_values is Ellipsis:
            return self.values is Ellipsis
        return self.values == other_values

    def with_concrete_bounds(self, bounds: AbstractSet[_U]) -> AbstractSet[_T | _U]:
        """Transform this into a true set by comparing to the concrete set of
        all possible values.

        Parameters
        ----------
        bounds : `AbstractSet`
            Concrete set of values to consider equivalent to ``...``

        Returns
        -------
        bounded : `AbstractSet`
            A set equivalent to ``self`` on if ``bounds`` contains all possible
            values.

        Raises
        ------
        LookupError
            Raised if ``self`` actually countains concrete values not in
            ``bounds``.  Value of the exception is the set of missing values.
        """
        if self.values is Ellipsis:
            return bounds
        elif self.values <= bounds:
            return self.values
        else:
            raise LookupError(frozenset(self.values - bounds))

    def issubset(self, other: AbstractSet[Any] | EllipsisType | UnboundableSet[Any]) -> bool:
        """Test whether all elements in ``self`` are also in ``other``.

        Parameters
        ----------
        other : `AbstractSet`, ``...``, or `UnboundableSet`
            Other set to compare against.

        Returns
        -------
        issubset : `bool`
            Whether ``self`` is a subset of ``other``.

        Notes
        -----
        The full set (``...``) is a subset only of itself, and the empty set
        is a subset of all sets, including itself.
        """
        other_values = self._extract_values(other)
        if other_values is Ellipsis:
            return True
        if self.values is Ellipsis:
            return False
        return self.values <= other_values

    def issuperset(self, other: AbstractSet[Any] | EllipsisType | UnboundableSet[Any]) -> bool:
        """Test whether any elements in ``self`` are also in ``other``.

        Parameters
        ----------
        other : `AbstractSet`, ``...``, or `UnboundableSet`
            Other set to compare against.

        Returns
        -------
        issuperset : `bool`
            Whether ``self`` is a superset of ``other``.

        Notes
        -----
        The full set (``...``) is a subset all other sets, including itself,
        and the empty set is a superset only of itself.
        """
        if self.values is Ellipsis:
            return True
        other_values = self._extract_values(other)
        if other_values is Ellipsis:
            return False
        return self.values >= other_values

    def isdisjoint(self, other: AbstractSet[Any] | EllipsisType | UnboundableSet[Any]) -> bool:
        """Test whether any elements in ``other`` are also in ``self``.

        Parameters
        ----------
        other : `AbstractSet`, ``...``, or `UnboundableSet`
            Other set to compare against.

        Returns
        -------
        isdisjoint : `bool`
            Whether ``self`` has any elements in common with ``other``.

        Notes
        -----
        The full set (``...``) is disjoint only with the empty set, which is
        disjoint with all sets, including itself.
        """
        other_values = self._extract_values(other)
        if self.values is Ellipsis:
            return not other_values
        if other_values is Ellipsis:
            return not self.values
        return self.values.isdisjoint(other_values)

    def union(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> UnboundableSet[_T]:
        """Return a set with all elements in any of the given sets.

        Parameters
        ----------
        *args : `AbstractSet`, ``...``, or `UnboundableSet`
            Sets to combine.

        Returns
        -------
        union : `UnboundableSet`
            Set containing all elements in any of the given sets.  If called
            with no arguments, this is the empty set.  If any argument is the
            full set (``...``), the result will be the full set.
        """
        result = MutableUnboundableSet[_T].make_empty()
        result.update(*args)
        return result

    def intersection(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> UnboundableSet[_T]:
        """Return a set with only elements that are in all of the given sets.

        Parameters
        ----------
        *args : `AbstractSet`, ``...``, or `UnboundableSet`
            Sets to combine.

        Returns
        -------
        intersection : `UnboundableSet`
            Set containing only elements in all of the given sets.  If called
            with no arguments, this is the full set.
        """
        result = MutableUnboundableSet[_T].make_full()
        result.intersection_update(*args)
        return result

    def copy(self) -> UnboundableSet[_T]:
        """Return a set with the same contents as ``self``.

        Returns
        -------
        copy : `UnboundableSet`
            A set that is guaranteed not to have any other owners that could
            change its contents.  This may actually be ``self`` for immutable
            derived classes, since those can never have any owners that can
            change their contents.
        """
        return MutableUnboundableSet(Ellipsis if self.values is Ellipsis else set(self.values))

    def mutable(self) -> MutableUnboundableSet[_T]:
        """Return ``self`` if ``self`` is a `MutableUnboundableSet`, or a
        mutable copy otherwise.

        Returns
        -------
        mutable : `MutableUnboundableSet`
            A mutable set.
        """
        return MutableUnboundableSet(Ellipsis if self.values is Ellipsis else set(self.values))

    def frozen(self) -> FrozenUnboundableSet[_T]:
        """Return ``self`` if ``self`` is a `FrozenUnboundableSet`, or an
        immutable copy otherwise.

        Returns
        -------
        frozen : `FrozenUnboundableSet`
            An immutable set.
        """
        return FrozenUnboundableSet(Ellipsis if self.values is Ellipsis else frozenset(self.values))


@final
class FrozenUnboundableSet(UnboundableSet[_T]):
    """An immutable implementation of `UnboundableSet`.

    Notes
    -----
    While `UnboundableSet` is not directly mutable, it does not guarantee that
    it cannot be mutated by other holders of its internal `set`-like object;
    `FrozenUnboundableSet` does.
    """

    __slots__ = ()

    values: Union[frozenset[_T], EllipsisType]
    """The ``frozenset`` or sentinel ``...`` backing the
    `FrozenUnboundableSet`.
    """

    empty: ClassVar[FrozenUnboundableSet]
    """The empty set that contains no elements."""

    full: ClassVar[FrozenUnboundableSet]
    """The full set that contains all elements."""

    @classmethod
    def make_empty(cls) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return cls.empty

    @classmethod
    def make_full(cls) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return cls.full

    @classmethod
    def coerce(cls, other: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return super().coerce(other).frozen()

    def __hash__(self) -> int:
        return hash(self.values)

    def union(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet.union(*args).frozen()

    def intersection(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet.intersection(*args).frozen()

    def copy(self) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return self

    def frozen(self) -> FrozenUnboundableSet[_T]:
        # Docstring inherited.
        return self


FrozenUnboundableSet.empty = FrozenUnboundableSet(frozenset())
FrozenUnboundableSet.full = FrozenUnboundableSet(Ellipsis)


@final
class MutableUnboundableSet(UnboundableSet[_T]):

    __slots__ = ()

    values: Union[set[_T], EllipsisType]
    """The ``set`` or sentinel ``...`` backing the `MutableUnboundableSet`.
    """

    @classmethod
    def make_empty(cls) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet[_T].make_empty().mutable()

    @classmethod
    def make_full(cls) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet[_T].make_full().mutable()

    @classmethod
    def coerce(cls, other: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return super().coerce(other).mutable()

    @classmethod
    def make_empty_defaultdict(
        cls, items: Iterable[tuple[_U, AbstractSet[_T] | EllipsisType | UnboundableSet[_T]]] = ()
    ) -> defaultdict[_U, MutableUnboundableSet[_T]]:
        """Construct a mapping whose default value is an empty set.

        Returns
        -------
        empty_mapping : `defaultdict`
            A `defaultdict` that returns an empty `MutableUnboundableSet` for
            missing keys.  Accessing with a missing key modifies the mapping by
            inserting a new empty set into it.
        """
        return defaultdict[_U, MutableUnboundableSet[_T]](cls.make_empty)

    @classmethod
    def make_full_defaultdict(cls) -> defaultdict[_U, MutableUnboundableSet[_T]]:
        """Construct a mapping whose default value is a full set.

        Returns
        -------
        full_mapping : `defaultdict`
            A `defaultdict` that returns a full `MutableUnboundableSet` for
            missing keys.  Accessing with a missing key modifies the mapping by
            inserting a new full set into it.
        """
        return defaultdict[_U, MutableUnboundableSet[_T]](cls.make_full)

    def union(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet.union(*args).mutable()

    def intersection(*args: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return UnboundableSet.intersection(*args).mutable()

    def update(self, *others: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> None:
        """Update ``self`` to include all elements in any of the given sets,
        including ``self``.

        Parameters
        ----------
        *others : `AbstractSet`, ``...``, or `UnboundableSet`
            Other sets to combine.
        """
        if self.values is Ellipsis:
            return
        for other in others:
            other_values = self._extract_values(other)
            if other_values is Ellipsis:
                self.values = Ellipsis
                return
            self.values.update(other_values)

    def intersection_update(self, *others: AbstractSet[_T] | EllipsisType | UnboundableSet[_T]) -> None:
        """Update ``self`` to include only elements in all of the given sets,
        including ``self``.

        Parameters
        ----------
        *others : `AbstractSet`, ``...``, or `UnboundableSet`
            Other sets to combine.
        """
        for other in others:
            other_values = self._extract_values(other)
            if other_values is Ellipsis:
                continue
            if self.values is Ellipsis:
                self.values = set(other_values)
            else:
                self.values.intersection_update(other_values)

    def add(self, item: _T) -> None:
        """Add a single item to the set.

        Parameters
        ----------
        item
            Item to add to the set.
        """
        if self.values is not Ellipsis:
            self.values.add(item)

    def clear(self) -> None:
        """Remove all items from the set."""
        if self.values is Ellipsis:
            self.values = set()
        else:
            self.values.clear()

    def copy(self) -> MutableUnboundableSet[_T]:
        # Docstring inherited.
        return super().copy().mutable()

    def mutable(self) -> MutableUnboundableSet[_T]:
        # Docstring inherited.x
        return self
