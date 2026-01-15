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

    @property
    def rows(self) -> list:
        """Unique rows in the plate layout."""
        return sorted({well.row for well in self.layout})

    @property
    def columns(self) -> list:
        """Unique columns in the plate layout."""
        return sorted({well.column for well in self.layout})

    @property
    def well_ids(self):
        """Return a list of all well IDs in the layout."""
        return sorted({well.id for well in self.layout})

    def __getitem__(self, well_id: str):
        """Get a well from the layout by its ID.

        This method allows accessing wells using dictionary-style notation:
        plate["A01"] instead of searching through the layout list.

        Args:
            well_id: The well ID to retrieve (e.g., "A01", "H12")

        Returns:
            The Well object corresponding to the given ID

        Raises:
            KeyError: If the well ID doesn't exist in the layout
        """
        for well in self.layout:
            if well.id == well_id:
                return well

        raise KeyError(f"Well ID '{well_id}' not found in plate layout.")

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

    def display(self, max_width: int = 12, empty: str = "-") -> str:
        """Display the plate layout as a grid table.

        Args:
            max_width: Maximum width for each cell (sample names will be truncated).
            empty: String to display for empty/missing wells.

        Returns:
            Formatted string representation of the plate grid.
        """
        rows = self.rows
        columns = self.columns

        if not rows or not columns:
            return "Empty plate layout"

        # Create a lookup dictionary for quick access
        well_map = {(well.row, well.column): well.sample for well in self.layout}

        # Calculate column width (at least 3 for row label, max_width for content)
        col_width = min(max_width, max(len(str(c)) for c in columns))
        col_width = max(col_width, len(empty))

        # Adjust based on actual sample name lengths
        if well_map:
            max_sample_len = max(len(sample) for sample in well_map.values() if sample)
            col_width = max(col_width, min(max_width, max_sample_len))

        # Build the header row with column numbers
        header = "   " + " ".join(f"{col:>{col_width}}" for col in columns)
        separator = "   " + " ".join("-" * col_width for _ in columns)

        lines = [header, separator]

        # Build each row
        for row in rows:
            row_cells = []
            for col in columns:
                sample = well_map.get((row, col), "")
                if not sample:
                    cell_value = empty
                else:
                    # Truncate if too long
                    cell_value = sample[:col_width] if len(sample) > col_width else sample
                row_cells.append(f"{cell_value:>{col_width}}")

            row_line = f"{row}| " + " ".join(row_cells)
            lines.append(row_line)

        return "\n".join(lines)


@dataclass
class Well:
    """Represents a single well in a microplate.

    Attributes:
        row: Single uppercase letter (A-Z) indicating the row.
        column: Integer column number.
        sample: Sample identifier or name in this well.
        properties: Additional metadata or properties for this well.
    """

    row: str
    column: int
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
        if not isinstance(self.column, int):
            try:
                self.column = int(self.column)
            except (ValueError, TypeError) as error:
                raise TypeError(
                    f"Column must be convertible to an integer, got {type(self.column)}"
                ) from error

    def __str__(self) -> str:
        """Return string representation of the well with the column integer padded with a zero."""
        return f"{self.row}{self.column:02d}"

    def __repr__(self) -> str:
        """Return a string that could be used to recreate this object."""
        return (
            f"Well(row='{self.row}', "
            f"column={self.column}, "
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
            column = int(well_id[1:])
        except ValueError as e:
            raise ValueError(f"Could not parse column number from '{well_id}'") from e

        return cls(row, column, sample, properties)

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
