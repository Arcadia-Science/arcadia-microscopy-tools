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

    # CSV column name constants
    WELL_ID_KEY = "well_id"
    SAMPLE_KEY = "sample"

    layout: Sequence[Well]
    _well_lookup: dict[str, Well] = field(default_factory=dict, init=False, repr=False)
    _rows: list[str] = field(default_factory=list, init=False, repr=False)
    _columns: list[int] = field(default_factory=list, init=False, repr=False)
    _well_ids: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Validate and initialize the layout."""
        # Check for duplicate well IDs
        well_ids = [well.id for well in self.layout]
        duplicates = [well_id for well_id in set(well_ids) if well_ids.count(well_id) > 1]
        if duplicates:
            raise ValueError(f"Duplicate well IDs found: {', '.join(sorted(duplicates))}")

        # Build lookup cache for efficient access
        self._well_lookup = {well.id: well for well in self.layout}

        # Cache property values
        self._rows = sorted({well.row for well in self.layout})
        self._columns = sorted({well.column for well in self.layout})
        self._well_ids = sorted({well.id for well in self.layout})

    @property
    def rows(self) -> list[str]:
        """Unique rows in the plate layout."""
        return self._rows

    @property
    def columns(self) -> list[int]:
        """Unique columns in the plate layout."""
        return self._columns

    @property
    def well_ids(self) -> list[str]:
        """Return a list of all well IDs in the layout."""
        return self._well_ids

    def __getitem__(self, well_id: str) -> Well:
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
        if well_id not in self._well_lookup:
            raise KeyError(f"Well ID '{well_id}' not found in plate layout.")
        return self._well_lookup[well_id]

    def __len__(self) -> int:
        """Return the number of wells in the layout."""
        return len(self.layout)

    def __contains__(self, well_id: str) -> bool:
        """Check if a well ID exists in the layout.

        Args:
            well_id: The well ID to check (e.g., "A01", "H12")

        Returns:
            True if the well exists, False otherwise
        """
        return well_id in self._well_lookup

    def __iter__(self):
        """Iterate over wells in the layout."""
        return iter(self.layout)

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
            non_empty_samples = [sample for sample in well_map.values() if sample]
            if non_empty_samples:
                max_sample_len = max(len(sample) for sample in non_empty_samples)
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
    def from_string(
        cls,
        well_id: str,
        sample: str = "",
        properties: Mapping[str, Any] | None = None,
    ) -> Well:
        """Create a Well from a well ID string.

        Args:
            well_id: Well identifier (e.g., 'A1', 'B12').
            sample: Sample identifier or name. Defaults to empty string.
            properties: Additional metadata for this well. Defaults to empty dict.

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

        if properties is None:
            properties = {}

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
        if MicroplateLayout.WELL_ID_KEY not in data.keys():
            raise ValueError(f"Dictionary must contain '{MicroplateLayout.WELL_ID_KEY}' key")
        well_id = data[MicroplateLayout.WELL_ID_KEY]

        sample = data.get(MicroplateLayout.SAMPLE_KEY, "")
        properties = {
            k: v
            for k, v in data.items()
            if k not in [MicroplateLayout.WELL_ID_KEY, MicroplateLayout.SAMPLE_KEY]
        }

        return cls.from_string(well_id, sample, properties)
