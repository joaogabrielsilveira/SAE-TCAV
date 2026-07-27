"""Consistent, optional progress reporting for long comparison stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:  # Keep existing library imports usable before extras install.
    tqdm = None


T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    enabled: bool,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
) -> Iterator[T]:
    """Wrap an iterable in a consistently configured tqdm progress bar."""

    if not enabled or tqdm is None:
        return iter(iterable)
    return iter(
        tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
        )
    )
