from __future__ import annotations
import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Well:
    """Represents a single well in a microplate.

    Attributes:
        id: Well identifier (e.g., "A01", "B12").
        sample: Sample identifier or name in this well.
        properties: Additional metadata or properties for this well.
    """

    id: str
    sample: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize the well ID."""
        if not self.id or len(self.id) < 2:
            raise ValueError("Well ID must be at least 2 characters (e.g., 'A1' or 'A01')")

        row = self.id[0].upper()
        if not "A" <= row <= "Z":
            raise ValueError(f"Row must be A-Z, got '{row}'")

        try:
            column = int(self.id[1:])
        except ValueError as e:
            raise ValueError(f"Could not parse column number from '{self.id}'") from e

        # Normalize to zero-padded format (A1 -> A01)
        normalized = f"{row}{column:02d}"
        if normalized != self.id:
            object.__setattr__(self, "id", normalized)

    @property
    def row(self) -> str:
        """Extract row letter from well ID."""
        return self.id[0]

    @property
    def column(self) -> int:
        """Extract column number from well ID."""
        return int(self.id[1:])

    def __str__(self) -> str:
        """Return well ID string."""
        return self.id

    def __repr__(self) -> str:
        """Return a string that could be used to recreate this object."""
        props = f", properties={self.properties!r}" if self.properties else ""
        return f"Well(id='{self.id}', sample='{self.sample}'{props})"

    @classmethod
    def from_string(
        cls,
        id: str,
        sample: str = "",
        properties: Mapping[str, Any] | None = None,
    ) -> Well:
        """Create a Well from a well ID string.

        Args:
            id: Well identifier (e.g., 'A1', 'B12').
            sample: Sample identifier or name. Defaults to empty string.
            properties: Additional metadata for this well. Defaults to empty dict.

        Returns:
            Well instance parsed from the well ID string.
        """
        return cls(id, sample, properties or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Well:
        """Create a Well from a dictionary (e.g., from CSV row).

        Args:
            data: Dictionary containing 'well_id' key and optional 'sample' and property keys.
                  CSV files should have a 'well_id' column.

        Returns:
            Well instance created from the dictionary.

        Raises:
            ValueError: If 'well_id' key is missing from the dictionary.
        """
        if "well_id" not in data:
            raise ValueError("Dictionary must contain 'well_id' key")

        id = data["well_id"]
        sample = data.get("sample", "")
        properties = {k: v for k, v in data.items() if k not in ("well_id", "sample")}

        return cls(id, sample, properties)


@dataclass(frozen=True)
class MicroplateLayout:
    """Representation of a microwell plate layout.

    Attributes:
        wells: Mapping of well IDs to Well objects.
    """

    wells: Mapping[str, Well]

    def __post_init__(self):
        """Validate the layout."""
        # Convert to dict if not already
        if not isinstance(self.wells, dict):
            object.__setattr__(self, "wells", dict(self.wells))

        # Verify well IDs match keys
        for well_id, well in self.wells.items():
            if well.id != well_id:
                raise ValueError(f"Well ID mismatch: key '{well_id}' != well.id '{well.id}'")

    @property
    def rows(self) -> list[str]:
        """Unique rows in the plate layout."""
        return sorted({well.row for well in self.wells.values()})

    @property
    def columns(self) -> list[int]:
        """Unique columns in the plate layout."""
        return sorted({well.column for well in self.wells.values()})

    @property
    def well_ids(self) -> list[str]:
        """Return a list of all well IDs in the layout."""
        return sorted(self.wells.keys())

    def __getitem__(self, well_id: str) -> Well:
        """Get a well by its ID.

        Args:
            well_id: The well ID to retrieve (e.g., "A01", "H12")

        Returns:
            The Well object corresponding to the given ID

        Raises:
            KeyError: If the well ID doesn't exist in the layout
        """
        try:
            return self.wells[well_id]
        except KeyError:
            raise KeyError(f"Well ID '{well_id}' not found in plate layout.") from None

    def __len__(self) -> int:
        """Return the number of wells in the layout."""
        return len(self.wells)

    def __contains__(self, well_id: str) -> bool:
        """Check if a well ID exists in the layout.

        Args:
            well_id: The well ID to check (e.g., "A01", "H12")

        Returns:
            True if the well exists, False otherwise
        """
        return well_id in self.wells

    def __iter__(self):
        """Iterate over wells in the layout."""
        return iter(self.wells.values())

    @classmethod
    def from_csv(cls, csv_path: Path) -> MicroplateLayout:
        """Load a microplate layout from a CSV file.

        Args:
            csv_path: Path to CSV file containing well_id, sample, and optional property columns.

        Returns:
            MicroplateLayout instance with wells parsed from the CSV.
        """
        wells = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                well = Well.from_dict(row)
                wells[well.id] = well

        return cls(wells)

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

        # Build well lookup by (row, column)
        well_map = {(well.row, well.column): well.sample for well in self.wells.values()}

        # Calculate column width
        samples = [s for s in well_map.values() if s]
        max_sample = max((len(s) for s in samples), default=0)
        col_width = max(
            len(empty), min(max_width, max_sample), max((len(str(c)) for c in columns), default=0)
        )

        # Header
        header = "   " + " ".join(f"{col:>{col_width}}" for col in columns)
        separator = "   " + " ".join("-" * col_width for _ in columns)
        lines = [header, separator]

        # Rows
        for row in rows:
            cells = []
            for col in columns:
                sample = well_map.get((row, col), "")
                display = sample[:col_width] if sample else empty
                cells.append(f"{display:>{col_width}}")
            lines.append(f"{row}| " + " ".join(cells))

        return "\n".join(lines)
