from __future__ import annotations
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MicroplateLayout:
    """Representation of a microwell plate layout.

    Attributes:
        layout: Sequence of Well objects representing the plate configuration.
    """

    layout: Sequence[Well]

    @classmethod
    def from_csv(cls, csv_path: Path) -> MicroplateLayout:
        """Load a microplate layout from a CSV file.

        Args:
            csv_path: Path to CSV file containing well_id, sample, and optional property columns.

        Returns:
            MicroplateLayout instance with wells parsed from the CSV.
        """
        layout = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                well = Well.from_dict(row)
                layout.append(well)

        return cls(layout)


@dataclass
class Well:
    """Represents a single well in a microplate.

    Attributes:
        row: Single uppercase letter (A-Z) indicating the row.
        col: Integer column number.
        sample: Sample identifier or name in this well.
        properties: Additional metadata or properties for this well.
    """

    row: str
    col: int
    sample: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate inputs after initialization."""
        # Validate and normalize row
        self.row = self.row.upper()

        # Ensure row is a single uppercase letter
        if not isinstance(self.row, str) or len(self.row) != 1 or not "A" <= self.row <= "Z":
            raise ValueError("Row must be a single uppercase letter (A-Z)")

        # Ensure column is an integer
        if not isinstance(self.col, int):
            try:
                self.col = int(self.col)
            except (ValueError, TypeError) as error:
                raise TypeError(
                    f"Column must be convertible to an integer, got {type(self.col)}"
                ) from error

    def __str__(self) -> str:
        """Return string representation of the well with the column integer padded with a zero."""
        return f"{self.row}{self.col:02d}"

    def __repr__(self) -> str:
        """Return a string that could be used to recreate this object."""
        return (
            f"Well(row='{self.row}', "
            f"column={self.col}, "
            f"sample='{self.sample}', "
            f"properties={repr(self.properties)})"
        )

    @property
    def id(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, well_id: str, sample: str, properties: Mapping[str, Any]) -> Well:
        """Create a Well from a well ID string.

        Args:
            well_id: Well identifier (e.g., 'A1', 'B12').
            sample: Sample identifier or name.
            properties: Additional metadata for this well.

        Returns:
            Well instance parsed from the well ID string.

        Raises:
            ValueError: If well_id is invalid or column cannot be parsed.
        """
        if not well_id or len(well_id) < 2:
            raise ValueError("Well ID must be at least 2 characters (e.g., 'A1' or 'A01')")

        row = well_id[0].upper()
        try:
            col = int(well_id[1:])
        except ValueError as e:
            raise ValueError(f"Could not parse column number from '{well_id}'") from e

        return cls(row, col, sample, properties)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Well:
        """Create a Well from a dictionary.

        Args:
            data: Dictionary containing 'well_id' key and optional 'sample' and property keys.

        Returns:
            Well instance created from the dictionary.

        Raises:
            ValueError: If 'well_id' key is missing from the dictionary.
        """
        if "well_id" not in data.keys():
            raise ValueError("Dictionary must contain 'well_id' key")
        well_id = data["well_id"]

        sample = data.get("sample", "")
        properties = {k: v for k, v in data.items() if k not in ["well_id", "sample"]}

        return cls.from_string(well_id, sample, properties)
