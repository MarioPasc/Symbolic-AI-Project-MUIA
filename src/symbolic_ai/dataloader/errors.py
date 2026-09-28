"""Exceptions raised by the dataloader."""

from __future__ import annotations


class DataLoaderError(Exception):
    """Base class of every dataloader error."""


class DatabaseValidationError(DataLoaderError):
    """The database violates its schema or a validation rule; ``problems`` lists every violation."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} database problem(s):\n" + "\n".join(problems))


class UnknownIdError(DataLoaderError, KeyError):
    """An identifier that does not exist in the database was requested."""
