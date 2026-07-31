"""Imaging channel model.

A channel is the atomic acquisition unit: one detection configuration that
yields one image plane per (z, t, position). A channel's identity is its
physical/optical configuration -- the contrast :class:`Modality` plus only the
facets that are meaningful for that modality (spectral bands, detection
geometry). Human-readable names and display colors are deliberately *not* part
of a channel's identity:

- every channel exposes a human-readable ``name`` (metadata, never identity)
- display colors live in :mod:`arcadia_microscopy_tools.display`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class Modality(Enum):
    """Physical contrast mechanism that forms an image."""

    BRIGHTFIELD = "brightfield"
    DIC = "dic"
    PHASE = "phase"
    FLUORESCENCE = "fluorescence"
    SRS = "srs"
    CARS = "cars"
    SHG = "shg"


class DetectionGeometry(Enum):
    """Direction in which signal is collected relative to illumination."""

    EPI = "epi"  # reflected / backward (incl. non-descanned epi detection)
    FORWARD = "forward"  # transmitted


@dataclass(frozen=True)
class Channel(ABC):
    """An imaging channel: one detection configuration producing one image plane.

    Subclasses model a single contrast :class:`Modality` and expose only the
    facets relevant to it. Equality and hashing are based on the identifying
    facets (never on the human-readable name or display color), so two channels
    compare equal iff they describe the same optical configuration.
    """

    modality: ClassVar[Modality]
    # ``geometry`` is declared on each concrete subclass: a ``ClassVar`` where the
    # detection geometry is fixed by physics (label-free, fluorescence, SRS) and an
    # identifying field where it is a free choice (CARS, SHG).

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for the channel (metadata, not identity)."""


# ── Label-free transmitted-contrast channels ────────────────────────────────


@dataclass(frozen=True)
class BrightfieldChannel(Channel):
    """Transmitted-light brightfield contrast."""

    modality: ClassVar[Modality] = Modality.BRIGHTFIELD
    geometry: ClassVar[DetectionGeometry] = DetectionGeometry.FORWARD

    @property
    def name(self) -> str:
        return "BRIGHTFIELD"


@dataclass(frozen=True)
class DICChannel(Channel):
    """Differential interference contrast."""

    modality: ClassVar[Modality] = Modality.DIC
    geometry: ClassVar[DetectionGeometry] = DetectionGeometry.FORWARD

    @property
    def name(self) -> str:
        return "DIC"


@dataclass(frozen=True)
class PhaseChannel(Channel):
    """Phase contrast."""

    modality: ClassVar[Modality] = Modality.PHASE
    geometry: ClassVar[DetectionGeometry] = DetectionGeometry.FORWARD

    @property
    def name(self) -> str:
        return "PHASE"


# ── Spectral channels (carry excitation/emission wavelengths) ────────────────


@dataclass(frozen=True, kw_only=True)
class SpectralChannel(Channel):
    """Base for channels whose identity includes excitation/emission wavelengths.

    Attributes:
        excitation_nm: Excitation wavelength in nanometers, if known.
        emission_nm: Emission wavelength in nanometers, if known.
        label: Optional human-readable name (e.g. ``"DAPI"``). Does not
            participate in equality or hashing.
    """

    excitation_nm: float | None = None
    emission_nm: float | None = None
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.excitation_nm is not None and self.excitation_nm <= 0:
            raise ValueError("excitation_nm must be positive")
        if self.emission_nm is not None and self.emission_nm <= 0:
            raise ValueError("emission_nm must be positive")

    @property
    def name(self) -> str:
        return self.label or self._default_name()

    @abstractmethod
    def _default_name(self) -> str:
        """Canonical name derived from the channel's identifying facets."""


@dataclass(frozen=True, kw_only=True)
class FluorescenceChannel(SpectralChannel):
    """Widefield or confocal fluorescence, identified by its spectral bands."""

    modality: ClassVar[Modality] = Modality.FLUORESCENCE
    geometry: ClassVar[DetectionGeometry] = DetectionGeometry.EPI

    def _default_name(self) -> str:
        if self.excitation_nm is not None and self.emission_nm is not None:
            return f"Ex{self.excitation_nm:.0f}/Em{self.emission_nm:.0f}"
        if self.excitation_nm is not None:
            return f"Ex{self.excitation_nm:.0f}"
        if self.emission_nm is not None:
            return f"Em{self.emission_nm:.0f}"
        return self.modality.name


@dataclass(frozen=True, kw_only=True)
class SRSChannel(SpectralChannel):
    """Stimulated Raman scattering (loss-based; detected in transmission)."""

    modality: ClassVar[Modality] = Modality.SRS
    geometry: ClassVar[DetectionGeometry] = DetectionGeometry.FORWARD

    def _default_name(self) -> str:
        return "SRS"


@dataclass(frozen=True, kw_only=True)
class CARSChannel(SpectralChannel):
    """Coherent anti-Stokes Raman scattering.

    Detection geometry (epi vs forward) is an identifying facet here: ``E-CARS``
    and ``F-CARS`` are distinct channels of the same modality.
    """

    modality: ClassVar[Modality] = Modality.CARS
    geometry: DetectionGeometry = DetectionGeometry.EPI

    def _default_name(self) -> str:
        prefix = "E" if self.geometry is DetectionGeometry.EPI else "F"
        return f"{prefix}-CARS"


@dataclass(frozen=True, kw_only=True)
class SHGChannel(SpectralChannel):
    """Second-harmonic generation.

    Detection geometry (epi vs forward) is an identifying facet here: ``E-SHG``
    and ``F-SHG`` are distinct channels of the same modality.
    """

    modality: ClassVar[Modality] = Modality.SHG
    geometry: DetectionGeometry = DetectionGeometry.EPI

    def _default_name(self) -> str:
        prefix = "E" if self.geometry is DetectionGeometry.EPI else "F"
        return f"{prefix}-SHG"


# ── Predefined channel presets ──────────────────────────────────────────────
#
# These are convenient, ready-made channel instances. The nonlinear presets
# (SRS / *-CARS / *-SHG) carry no wavelengths -- loaders fill those in per
# acquisition (e.g. via ``dataclasses.replace``), which preserves geometry and
# label while producing a distinct, fully-specified channel.

BRIGHTFIELD = BrightfieldChannel()
DIC = DICChannel()
PHASE = PhaseChannel()
DAPI = FluorescenceChannel(excitation_nm=405, emission_nm=450, label="DAPI")
FITC = FluorescenceChannel(excitation_nm=488, emission_nm=512, label="FITC")
TRITC = FluorescenceChannel(excitation_nm=561, emission_nm=595, label="TRITC")
CY5 = FluorescenceChannel(excitation_nm=640, emission_nm=665, label="CY5")
SRS = SRSChannel()
E_CARS = CARSChannel(geometry=DetectionGeometry.EPI)
F_CARS = CARSChannel(geometry=DetectionGeometry.FORWARD)
E_SHG = SHGChannel(geometry=DetectionGeometry.EPI)
F_SHG = SHGChannel(geometry=DetectionGeometry.FORWARD)

CHANNELS: dict[str, Channel] = {
    ch.name: ch
    for ch in [
        BRIGHTFIELD,
        DIC,
        PHASE,
        DAPI,
        FITC,
        TRITC,
        CY5,
        SRS,
        E_CARS,
        F_CARS,
        E_SHG,
        F_SHG,
    ]
}
