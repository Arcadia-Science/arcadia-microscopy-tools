from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


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

        # Normalize to capital letter, zero-padded format (a1 -> A01)
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
        layout: Mapping of well IDs to Well objects.
    """

    layout: Mapping[str, Well]

    def __post_init__(self):
        """Validate the layout."""
        # Convert to dict if not already
        if not isinstance(self.layout, dict):
            object.__setattr__(self, "layout", dict(self.layout))

        # Verify well IDs match keys
        for well_id, well in self.layout.items():
            if well.id != well_id:
                raise ValueError(f"Well ID mismatch: key '{well_id}' != well.id '{well.id}'")

    @property
    def wells(self) -> list[Well]:
        """Return a sorted list of all wells in the layout."""
        return sorted(self.layout.values(), key=lambda well: well.id)

    @property
    def rows(self) -> list[str]:
        """Unique rows in the plate layout."""
        return sorted({well.row for well in self.layout.values()})

    @property
    def columns(self) -> list[int]:
        """Unique columns in the plate layout."""
        return sorted({well.column for well in self.layout.values()})

    @property
    def well_ids(self) -> list[str]:
        """Return a list of all well IDs in the layout."""
        return sorted(self.layout.keys())

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
            return self.layout[well_id]
        except KeyError:
            raise KeyError(f"Well ID '{well_id}' not found in plate layout.") from None

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
        return well_id in self.layout

    def __iter__(self):
        """Iterate over wells in the layout."""
        return iter(self.layout.values())

    @classmethod
    def from_csv(cls, csv_path: Path, **kwargs) -> MicroplateLayout:
        """Load a microplate layout from a CSV file using pandas.

        Args:
            csv_path: Path to CSV file containing well_id, sample, and optional property columns.
            **kwargs: Additional arguments passed to pd.read_csv (e.g., encoding, dtype).

        Returns:
            MicroplateLayout instance with wells parsed from the CSV.
        """
        df = pd.read_csv(csv_path, **kwargs)
        layout = {}
        for _, row in df.iterrows():
            well = Well.from_dict(row.to_dict())
            layout[well.id] = well

        return cls(layout)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert plate layout to a pandas DataFrame with all well data.

        Returns:
            DataFrame with columns: well_id, row, column, sample, and any additional properties.
            One row per well in the layout.
        """
        if not self.layout:
            return pd.DataFrame()

        data = []
        for well in self.layout.values():
            row_data = {
                "well_id": well.id,
                "row": well.row,
                "column": well.column,
                "sample": well.sample,
            }
            # Add any additional properties
            row_data.update(well.properties)
            data.append(row_data)

        return pd.DataFrame(data)

    def display(self) -> str:
        """Display the plate layout as a formatted grid table.

        Returns:
            String representation of the plate as a pivot table with rows and columns.
        """
        df = self.to_dataframe()
        if df.empty:
            return "Empty plate layout"

        # Create pivot table: rows as index, columns as columns, sample as values
        pivot = df.pivot(index="row", columns="column", values="sample")
        pivot = pivot.fillna("-")
        return pivot.to_string()
