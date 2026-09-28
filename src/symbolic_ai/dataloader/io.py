"""Reading of ``datapackage.json`` and the CSV resources it describes.

Two row-reading modes are provided. :func:`read_raw_rows` keeps every cell as ``str | None`` (an
empty cell becomes ``None``, per the database's CSV convention) so that :mod:`.validation` can
inspect every row without raising on the first malformed value. :func:`read_rows` performs the same
read and additionally converts each cell to its schema type, raising ``ValueError`` on a malformed
value; callers use it only after validation has confirmed every value parses.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeAlias

FieldValue: TypeAlias = str | int | bool | date | None

_TRUE_LITERAL = "true"
_FALSE_LITERAL = "false"
_DESCRIPTOR_FILE_NAME = "datapackage.json"


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """One field of a resource's Table Schema."""

    name: str
    type: str
    required: bool = False
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ForeignKeySchema:
    """One foreign key: ``field`` references ``reference_field`` of another resource."""

    field: str
    reference_resource: str
    reference_field: str


@dataclass(frozen=True, slots=True)
class ResourceSchema:
    """The descriptor entry of one CSV resource."""

    name: str
    path: str
    fields: tuple[FieldSchema, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeySchema, ...] = ()

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the field names, in declaration order."""
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> FieldSchema:
        """Return the schema of one field by name.

        Raises
        ------
        KeyError
            If no field of that name is declared.
        """
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Resource {self.name!r} has no field {name!r}.")


@dataclass(frozen=True, slots=True)
class DataPackage:
    """The parsed ``datapackage.json`` descriptor."""

    name: str
    version: str
    resources: tuple[ResourceSchema, ...] = field(default_factory=tuple)

    def resource(self, name: str) -> ResourceSchema:
        """Return the schema of one resource by name.

        Raises
        ------
        KeyError
            If no resource of that name is declared.
        """
        for candidate in self.resources:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Resource {name!r} is not declared in the data package descriptor.")

    def resource_for_self_reference(
        self, resource: ResourceSchema, reference_resource: str
    ) -> ResourceSchema:
        """Resolve a foreign key's target resource, where ``""`` means ``resource`` itself.

        Parameters
        ----------
        resource : ResourceSchema
            The resource declaring the foreign key.
        reference_resource : str
            The foreign key's ``reference.resource``; empty for a self-reference.

        Returns
        -------
        ResourceSchema
            ``resource`` if ``reference_resource`` is empty, else that named resource.
        """
        return resource if reference_resource == "" else self.resource(reference_resource)


def parse_semver(value: str) -> tuple[int, int, int]:
    """Parse a semantic version string ``MAJOR.MINOR.PATCH`` into a comparable tuple.

    Parameters
    ----------
    value : str
        The version string to parse.

    Returns
    -------
    tuple[int, int, int]
        ``(major, minor, patch)``, comparable numerically.

    Raises
    ------
    ValueError
        If ``value`` is not exactly three dot-separated non-negative integers.
    """
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(f"invalid semantic version {value!r} (expected MAJOR.MINOR.PATCH)")
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid semantic version {value!r}: {exc}") from exc
    return (major, minor, patch)


def _field_schema(raw: dict[str, Any]) -> FieldSchema:
    constraints: dict[str, Any] = raw.get("constraints", {})
    enum = constraints.get("enum")
    return FieldSchema(
        name=str(raw["name"]),
        type=str(raw["type"]),
        required=bool(constraints.get("required", False)),
        enum=tuple(str(v) for v in enum) if enum is not None else None,
    )


def _foreign_key_schema(raw: dict[str, Any]) -> ForeignKeySchema:
    reference: dict[str, Any] = raw["reference"]
    return ForeignKeySchema(
        field=str(raw["fields"]),
        reference_resource=str(reference["resource"]),
        reference_field=str(reference["fields"]),
    )


def _primary_key(raw: object) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(v) for v in raw)
    return (str(raw),)


def _resource_schema(raw: dict[str, Any]) -> ResourceSchema:
    schema: dict[str, Any] = raw["schema"]
    fields = tuple(_field_schema(f) for f in schema["fields"])
    foreign_keys = tuple(_foreign_key_schema(fk) for fk in schema.get("foreignKeys", []))
    return ResourceSchema(
        name=str(raw["name"]),
        path=str(raw["path"]),
        fields=fields,
        primary_key=_primary_key(schema["primaryKey"]),
        foreign_keys=foreign_keys,
    )


def read_descriptor(data_dir: Path) -> DataPackage:
    """Read and parse ``datapackage.json``.

    Parameters
    ----------
    data_dir : Path
        Root directory of the database.

    Returns
    -------
    DataPackage
        The parsed descriptor: name, version and every resource's schema.
    """
    descriptor_path = data_dir / _DESCRIPTOR_FILE_NAME
    with descriptor_path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    resources = tuple(_resource_schema(r) for r in raw["resources"])
    return DataPackage(name=str(raw["name"]), version=str(raw["version"]), resources=resources)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def read_raw_rows(data_dir: Path, resource: ResourceSchema) -> tuple[dict[str, str | None], ...]:
    """Read one resource's CSV as raw strings; an empty cell becomes ``None``.

    Parameters
    ----------
    data_dir : Path
        Root directory of the database.
    resource : ResourceSchema
        The resource's schema, as returned by :meth:`DataPackage.resource`.

    Returns
    -------
    tuple[dict[str, str | None], ...]
        One dictionary per row, keyed by field name, in file order.
    """
    path = data_dir / resource.path
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(
            {name: _as_optional_str(raw_row.get(name)) for name in resource.field_names}
            for raw_row in reader
        )


def parse_value(raw: str | None, field_type: str) -> FieldValue:
    """Convert one raw cell to its declared schema type.

    Parameters
    ----------
    raw : str | None
        The raw cell value; ``None`` for an empty cell.
    field_type : str
        One of ``"string"``, ``"integer"``, ``"boolean"``, ``"date"``.

    Returns
    -------
    FieldValue
        ``None`` for an empty cell, else the converted value.

    Raises
    ------
    ValueError
        If ``raw`` is not empty and does not parse as ``field_type``.
    """
    if raw is None:
        return None
    if field_type == "integer":
        return int(raw)
    if field_type == "boolean":
        if raw == _TRUE_LITERAL:
            return True
        if raw == _FALSE_LITERAL:
            return False
        raise ValueError(
            f"invalid boolean {raw!r} (expected {_TRUE_LITERAL!r} or {_FALSE_LITERAL!r})"
        )
    if field_type == "date":
        return date.fromisoformat(raw)
    return raw


def read_rows(data_dir: Path, resource: ResourceSchema) -> tuple[dict[str, FieldValue], ...]:
    """Read one resource's CSV, with every cell converted to its schema type.

    Parameters
    ----------
    data_dir : Path
        Root directory of the database.
    resource : ResourceSchema
        The resource's schema, as returned by :meth:`DataPackage.resource`.

    Returns
    -------
    tuple[dict[str, FieldValue], ...]
        One dictionary per row, keyed by field name, in file order.

    Raises
    ------
    ValueError
        If a cell does not parse as its declared type. Callers use this function only after
        validation (:func:`symbolic_ai.dataloader.validation.collect_problems`) has confirmed every
        value parses.
    """
    raw_rows = read_raw_rows(data_dir, resource)
    return tuple(
        {name: parse_value(value, resource.field(name).type) for name, value in row.items()}
        for row in raw_rows
    )
