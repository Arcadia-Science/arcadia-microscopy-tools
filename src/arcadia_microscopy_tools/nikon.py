"""Parse Nikon ND2 files into :class:`InstrumentMetadata`.

The public entry point is :func:`parse_nd2`, which reads only the metadata of an
ND2 file (no pixel data) and returns a fully-populated
:class:`~arcadia_microscopy_tools.microscopy.InstrumentMetadata` with one
:class:`~arcadia_microscopy_tools.metadata_structures.ChannelMetadata` per channel.
"""

from __future__ import annotations
import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nd2
import numpy as np

from .channels import (
    BrightfieldChannel,
    Channel,
    DICChannel,
    FluorescenceChannel,
    PhaseChannel,
)
from .exceptions import MetadataWarning
from .metadata_structures import (
    AcquisitionSettings,
    ChannelMetadata,
    DimensionFlags,
    MeasuredDimensions,
    MicroscopeConfig,
    NominalDimensions,
)
from .microscopy import InstrumentMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence

# Julian Day Number of the Unix epoch (1970-01-01T00:00:00Z).
_JD_UNIX_EPOCH = 2440587.5
_MS_PER_DAY = 86_400_000.0

# Dimensions that are *not* iterated frame-by-frame (handled within a frame or
# as the in-plane image axes).
_NON_LOOP_DIMS = frozenset({"C", "Y", "X"})

# Label-free modalities, matched against tokens in the channel name. ND2 modality
# flags are unreliable for these (e.g. a DIC channel is tagged "fluorescence"),
# so the human-readable channel name is the only dependable signal.
_LABEL_FREE_TOKENS: tuple[tuple[frozenset[str], type[Channel]], ...] = (
    (frozenset({"dic"}), DICChannel),
    (frozenset({"phase", "ph", "phc"}), PhaseChannel),
    (
        frozenset({"brightfield", "bright", "bf", "trans", "transmitted", "td", "bria"}),
        BrightfieldChannel,
    ),
)

# Per-line maximum optical output (mW), keyed by (light-source key, wavelength nm).
# Values are estimates of the source output (at the fiber/output port), derived
# from the manufacturer specs:
#   - LUN-F (Nikon LUN-F XL laser unit): per-line output estimated from the
#     Nikon laser-unit spec. The LUN-F XL is the higher-power variant, so its
#     per-line maxima exceed the ~15 mW of the LU-N3/LU-N4 units on the page.
#     https://www.microscope.healthcare.nikon.com/products/light-sources/lu-n3-lu-n4-laser-units
#   - Lida (Lumencor Lida): per-line output estimated from the Lumencor datasheet.
#     https://www.biovis.com/images/light_lumencor/54-10030_LIDA_IJ.pdf
#
# NOTE: these are source-side maxima. The actual power delivered at the sample
# can be up to ~30% lower due to losses through the optical path.
_MAX_LINE_POWER_MW: dict[tuple[str, int], float] = {
    ("LUN-F", 405): 50.0,
    ("LUN-F", 488): 60.0,
    ("LUN-F", 561): 50.0,
    ("LUN-F", 640): 40.0,
    ("Lida", 450): 120.0,
    ("Lida", 550): 230.0,
    ("Lida", 640): 170.0,
}


def parse_nd2(
    path: str | Path, channels: list[Channel] | None = None
) -> InstrumentMetadata:
    """Parse an ND2 file and return its :class:`InstrumentMetadata`.

    Args:
        path: Path to a ``.nd2`` file.
        channels: Optional list of :class:`Channel` objects (one per image
            channel, in order) overriding automatic channel detection. When not
            provided, channels are inferred from the ND2 optical configuration.

    Returns:
        An :class:`InstrumentMetadata` describing the image dimensions and the
        per-channel acquisition/optical configuration.
    """
    with nd2.ND2File(str(path)) as nf:
        return _parse_open(nf, channels)


def load_nd2(
    path: str | Path, channels: list[Channel] | None = None
) -> tuple[np.ndarray, InstrumentMetadata]:
    """Load an ND2 file's pixel data and metadata.

    Args:
        path: Path to a ``.nd2`` file.
        channels: Optional list of :class:`Channel` objects (one per image
            channel, in order) overriding automatic channel detection. When not
            provided, channels are inferred from the ND2 optical configuration.

    Returns:
        A ``(intensities, metadata)`` tuple, where ``intensities`` is a NumPy
        array with axes in :attr:`InstrumentMetadata.sizes` order and ``metadata``
        is the :class:`InstrumentMetadata` for the file.
    """
    with nd2.ND2File(str(path)) as nf:
        intensities = nf.asarray()
        metadata = _parse_open(nf, channels)
    return intensities, metadata


def _parse_open(nf: Any, channel_overrides: list[Channel] | None = None) -> InstrumentMetadata:
    """Build :class:`InstrumentMetadata` from an already-open :class:`nd2.ND2File`."""
    sizes = dict(nf.sizes)
    voxel = nf.voxel_size()
    channels = nf.metadata.channels or []
    unstructured = nf.unstructured_metadata()

    if channel_overrides is not None and len(channel_overrides) != len(channels):
        raise ValueError(
            f"Expected {len(channels)} channels but got {len(channel_overrides)} "
            "in channels list"
        )

    flags = _dimension_flags(sizes, channels)
    t_step_ms = _time_step_ms(nf.experiment)
    sample_settings = _sample_settings(unstructured)
    events = _events_table(nf)

    # Frame-level coordinate arrays (shared across channels in a frame),
    # preferring the per-frame "Recorded Data" in events over nominal/looped
    # values, with graceful fallbacks when events are unavailable.
    z_values = _frame_z_values_um(events, sizes, voxel.z) if flags.is_zstack else None
    t_values = _frame_t_values_ms(events, nf, sizes) if flags.is_timelapse else None
    if flags.is_montage:
        x_values, y_values = _frame_xy_values_um(events, nf, sizes)
    else:
        x_values, y_values = None, None

    first_frame: Any = nf.frame_metadata(0)

    channel_metadata_list = [
        _build_channel_metadata(
            index=index,
            channel_meta=channel_meta,
            frame_channel=first_frame.channels[index],
            plane_settings=sample_settings.get(str(index), {}),
            sizes=sizes,
            voxel=voxel,
            flags=flags,
            t_step_ms=t_step_ms,
            z_values=z_values,
            t_values=t_values,
            x_values=x_values,
            y_values=y_values,
            channel_override=channel_overrides[index] if channel_overrides else None,
        )
        for index, channel_meta in enumerate(channels)
    ]

    return InstrumentMetadata(sizes=sizes, channel_metadata_list=channel_metadata_list)


def _build_channel_metadata(
    *,
    index: int,
    channel_meta: Any,
    frame_channel: Any,
    plane_settings: dict[str, Any],
    sizes: dict[str, int],
    voxel: Any,
    flags: DimensionFlags,
    t_step_ms: float | None,
    z_values: np.ndarray | None,
    t_values: np.ndarray | None,
    x_values: np.ndarray | None,
    y_values: np.ndarray | None,
    channel_override: Channel | None = None,
) -> ChannelMetadata:
    name = channel_meta.channel.name or f"Channel{index}"
    excitation_meta = channel_meta.channel.excitationLambdaNm
    emission_meta = channel_meta.channel.emissionLambdaNm

    device_setting = plane_settings.get("DeviceSetting", {})
    light_source, line_wavelength, power_percent = _active_excitation(device_setting)
    power_mw = _line_power_mw(light_source, line_wavelength, power_percent)

    channel = channel_override or _build_channel(
        name=name,
        excitation_meta=excitation_meta,
        emission_meta=emission_meta,
        line_wavelength=line_wavelength,
    )

    resolution = NominalDimensions(
        x_size_px=sizes["X"],
        y_size_px=sizes["Y"],
        xy_step_um=voxel.x,
        z_size_px=sizes.get("Z") if flags.is_zstack else None,
        z_step_um=voxel.z if flags.is_zstack else None,
        t_size_px=sizes.get("T") if flags.is_timelapse else None,
        t_step_ms=t_step_ms if flags.is_timelapse else None,
    )

    measured = MeasuredDimensions(
        x_values_um=x_values if flags.is_montage else None,
        y_values_um=y_values if flags.is_montage else None,
        z_values_um=z_values if flags.is_zstack else None,
        t_values_ms=t_values if flags.is_timelapse else None,
    )

    acquisition = _acquisition_settings(plane_settings, channel_meta.microscope)
    optics = _microscope_config(channel_meta.microscope, light_source, power_mw)

    return ChannelMetadata(
        channel=channel,
        timestamp=_julian_to_datetime(frame_channel.time.absoluteJulianDayNumber),
        dimensions=flags,
        resolution=resolution,
        measured=measured,
        acquisition=acquisition,
        optics=optics,
    )


def _build_channel(
    *,
    name: str,
    excitation_meta: float | None,
    emission_meta: float | None,
    line_wavelength: float | None,
) -> Channel:
    """Map an ND2 channel to a :class:`Channel`.

    Label-free contrast (DIC/phase/brightfield) is detected by name; everything
    else is treated as fluorescence, with the excitation wavelength taken from
    the channel metadata when present, otherwise from the active light source.
    """
    tokens = set(_tokenize(name))
    for keywords, channel_cls in _LABEL_FREE_TOKENS:
        if tokens & keywords:
            return channel_cls()

    # ND2 uses 0 (falsy) to mean "no excitation wavelength recorded", so fall
    # back to the active light source's line wavelength in that case too.
    excitation_nm = excitation_meta if excitation_meta else line_wavelength
    return FluorescenceChannel(
        excitation_nm=excitation_nm,
        emission_nm=emission_meta or None,
        label=name,
    )


def _dimension_flags(sizes: dict[str, int], channels: Sequence[Any]) -> DimensionFlags:
    flags = DimensionFlags.SPATIAL_2D
    if sizes.get("Z", 1) > 1:
        flags |= DimensionFlags.Z_STACK
    if sizes.get("T", 1) > 1:
        flags |= DimensionFlags.TIMELAPSE
    if sizes.get("P", 1) > 1:
        flags |= DimensionFlags.MONTAGE
    # Only 3-component (RGB) data is flagged; SPECTRAL (lambda) acquisitions and
    # 4-component (RGBA) data are not currently detected here.
    component_count = channels[0].volume.componentCount if channels else 1
    if component_count == 3:
        flags |= DimensionFlags.RGB
    return flags


def _time_step_ms(experiment: Sequence[Any]) -> float | None:
    """Extract the nominal time step (ms) from a time loop in the experiment."""
    for loop in experiment:
        params = getattr(loop, "parameters", None)
        if params is None:
            continue
        periods = getattr(params, "periods", None)
        if periods:
            return periods[0].periodMs
        period_ms = getattr(params, "periodMs", None)
        if period_ms is not None:
            return period_ms
    return None


def _events_table(nf: Any) -> list[dict[str, Any]]:
    """Per-frame "Recorded Data" rows, or ``[]`` if unavailable.

    ``events()`` is unsupported for legacy ND2 files and may raise; callers fall
    back to nominal/looped values in that case.
    """
    try:
        rows = nf.events(orient="records")
    except Exception:  # noqa: BLE001 - any failure means "no events", fall back.
        return []
    return rows if isinstance(rows, list) else []


# Candidate column names (by hardware) for the per-frame index of each axis.
# Matched case-sensitively, exact first then as a substring (see _find_column).
_Z_INDEX_COLS = ("Z Index",)
_T_INDEX_COLS = ("T Index",)
_P_INDEX_COLS = ("P Index", "Point Index", "Multipoint Index", "Position Index")
# Candidate column names for the recorded stage coordinate / timestamp.
# "Z-Series" is the focus-relative offset NIS records per plane (centered on 0
# at focus); "Z Coord" is the absolute stage position.
_ZSERIES_COLS = ("Z-Series",)
_Z_COORD_COLS = ("Z Coord",)
_X_COORD_COLS = ("X Coord",)
_Y_COORD_COLS = ("Y Coord",)
_TIME_COLS = ("Time [s]", "Time")


def _find_column(events: list[dict[str, Any]], candidates: tuple[str, ...]) -> str | None:
    """Return the first event column matching ``candidates`` (exact, then substring)."""
    if not events:
        return None
    keys = list(events[0].keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    for candidate in candidates:
        for key in keys:
            if candidate in key:
                return key
    return None


def _axis_values(
    events: list[dict[str, Any]],
    index_cols: tuple[str, ...],
    value_cols: tuple[str, ...],
    size: int,
    *,
    scale: float = 1.0,
    zero: bool = False,
    center: bool = False,
) -> np.ndarray | None:
    """One recorded value per axis index, taken from the events table.

    Rows are per (flattened) frame, so the first row seen for each index along
    the target axis is used. Returns ``None`` (so the caller can fall back) when
    the needed columns are missing or any index/value is absent, non-numeric, or
    ``NaN``.

    ``zero`` shifts values so the first index is 0; ``center`` shifts them so the
    midpoint of the range is 0 (i.e. symmetric +/- about the stack center).
    """
    index_col = _find_column(events, index_cols)
    value_col = _find_column(events, value_cols)
    if index_col is None or value_col is None:
        return None

    by_index: dict[int, Any] = {}
    for row in events:
        raw_index = row.get(index_col)
        value = row.get(value_col)
        if raw_index is None or value is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        by_index.setdefault(index, value)

    if any(i not in by_index for i in range(size)):
        return None
    try:
        values = np.array([by_index[i] for i in range(size)], dtype=float) * scale
    except (TypeError, ValueError):
        # A recorded value wasn't numeric; treat as unavailable and fall back.
        return None
    if np.isnan(values).any():
        return None
    if zero:
        return values - values[0]
    if center:
        return values - (values.max() + values.min()) / 2.0
    return values


def _frame_z_values_um(
    events: list[dict[str, Any]], sizes: dict[str, int], z_step_um: float
) -> np.ndarray:
    """Measured Z positions (um, +/- about focus), else nominal centered spacing.

    Prefers the focus-relative ``Z-Series`` column, then the absolute ``Z Coord``
    centered on the stack midpoint, then a centered nominal range.
    """
    z_size = sizes["Z"]
    series = _axis_values(events, _Z_INDEX_COLS, _ZSERIES_COLS, z_size)
    if series is not None:
        return series
    measured = _axis_values(events, _Z_INDEX_COLS, _Z_COORD_COLS, z_size, center=True)
    if measured is not None:
        return measured
    return _nominal_z_values_um(sizes, z_step_um)


def _frame_t_values_ms(events: list[dict[str, Any]], nf: Any, sizes: dict[str, int]) -> np.ndarray:
    """Measured elapsed time (ms, zeroed to the first timepoint) from events, else frames."""
    measured = _axis_values(events, _T_INDEX_COLS, _TIME_COLS, sizes["T"], scale=1000.0, zero=True)
    if measured is not None:
        return measured
    return _measured_t_values_ms(nf, sizes)


def _frame_xy_values_um(
    events: list[dict[str, Any]], nf: Any, sizes: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Measured per-position stage X/Y (absolute um).

    Prefers the recorded ``X Coord``/``Y Coord`` columns in the events table,
    then the per-frame stage position from frame metadata. If neither is
    available the positions are unknown: a :class:`MetadataWarning` is issued and
    NaN-filled arrays are returned so metadata construction can still proceed
    (``MONTAGE`` requires non-``None`` X/Y values).
    """
    p_size = sizes.get("P", 1)
    x_values = _axis_values(events, _P_INDEX_COLS, _X_COORD_COLS, p_size)
    y_values = _axis_values(events, _P_INDEX_COLS, _Y_COORD_COLS, p_size)
    if x_values is not None and y_values is not None:
        return x_values, y_values

    measured = _measured_xy_values_um(nf, sizes)
    if measured is not None:
        return measured

    warnings.warn(
        "Montage stage positions are unavailable from both the events table and "
        "frame metadata; X/Y coordinate values are unknown and reported as NaN.",
        MetadataWarning,
        stacklevel=2,
    )
    return np.full(p_size, np.nan), np.full(p_size, np.nan)


def _nominal_z_values_um(sizes: dict[str, int], z_step_um: float) -> np.ndarray:
    """Nominal Z offsets (um) centered on the stack midpoint (+/- about focus)."""
    z_size = sizes["Z"]
    return (np.arange(z_size, dtype=float) - (z_size - 1) / 2.0) * z_step_um


def _measured_t_values_ms(nf: Any, sizes: dict[str, int]) -> np.ndarray:
    """Per-frame elapsed time (ms), zeroed to the first timepoint."""
    t_size = sizes["T"]
    values = np.empty(t_size, dtype=float)
    for t in range(t_size):
        frame_index = _frame_index(sizes, {"T": t})
        values[t] = nf.frame_metadata(frame_index).channels[0].time.relativeTimeMs
    return values - values[0]


def _measured_xy_values_um(nf: Any, sizes: dict[str, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-position stage X/Y (um) from frame metadata, or ``None`` if unavailable."""
    p_size = sizes.get("P", 1)
    x_values = np.empty(p_size, dtype=float)
    y_values = np.empty(p_size, dtype=float)
    for p in range(p_size):
        frame_index = _frame_index(sizes, {"P": p})
        try:
            stage = nf.frame_metadata(frame_index).channels[0].position.stagePositionUm
        except Exception:  # noqa: BLE001 - any failure means "no positions", fall back.
            return None
        x_values[p] = stage.x
        y_values[p] = stage.y
    if np.isnan(x_values).any() or np.isnan(y_values).any():
        return None
    return x_values, y_values


def _frame_index(sizes: dict[str, int], coord: dict[str, int]) -> int:
    """Row-major linear index over the looped (non C/Y/X) dimensions."""
    looped = [dim for dim in sizes if dim not in _NON_LOOP_DIMS]
    index = 0
    for position, dim in enumerate(looped):
        stride = 1
        for later in looped[position + 1 :]:
            stride *= sizes[later]
        index += coord.get(dim, 0) * stride
    return index


def _acquisition_settings(plane_settings: dict[str, Any], microscope: Any) -> AcquisitionSettings:
    exposure_ms = plane_settings.get("ExposureTime")
    exposure_time_s = exposure_ms / 1000.0 if exposure_ms else None

    camera = plane_settings.get("CameraSetting", {})
    binning = _binning(camera)

    zoom = getattr(microscope, "zoomMagnification", None)

    return AcquisitionSettings(
        exposure_time_s=exposure_time_s,
        zoom=zoom,
        binning=binning,
    )


def _binning(camera: dict[str, Any]) -> str | None:
    for fmt in ("FormatFast", "FormatQuality"):
        desc = camera.get(fmt, {}).get("Desc", {})
        bin_x = desc.get("BinningX")
        bin_y = desc.get("BinningY")
        if bin_x and bin_y and bin_x > 0 and bin_y > 0:
            return f"{int(bin_x)}x{int(bin_y)}"
    return None


def _microscope_config(
    microscope: Any,
    light_source: str | None,
    power_mw: float | None,
) -> MicroscopeConfig:
    magnification = microscope.objectiveMagnification
    return MicroscopeConfig(
        magnification=int(round(magnification)) if magnification else 0,
        numerical_aperture=microscope.objectiveNumericalAperture or 0.0,
        objective=microscope.objectiveName or None,
        light_source=light_source,
        power_mw=power_mw,
    )


def _sample_settings(unstructured: dict[str, Any]) -> dict[str, Any]:
    """Per-plane ``SampleSetting`` dict keyed by plane index (as a string)."""
    seq = _first_value(unstructured, "ImageMetadataSeqLV")
    if not isinstance(seq, dict):
        return {}
    sample_setting = (
        seq.get("SLxPictureMetadata", {}).get("PicturePlanes", {}).get("SampleSetting", {})
    )
    return sample_setting if isinstance(sample_setting, dict) else {}


def _line_power_mw(
    light_source: str | None,
    wavelength_nm: float | None,
    power_percent: float | None,
) -> float | None:
    """Convert a recorded laser-line percentage to absolute mW.

    Uses :data:`_MAX_LINE_POWER_MW` (a hardware spec, since the ND2 only records
    a percentage of each line's maximum). Returns ``None`` when the light source,
    wavelength, or percentage is unknown, or when the line's max power is not in
    the table -- never a bare percentage masquerading as mW.
    """
    if light_source is None or wavelength_nm is None or power_percent is None:
        return None
    max_mw = _MAX_LINE_POWER_MW.get((light_source, int(round(wavelength_nm))))
    if max_mw is None:
        return None
    return power_percent / 100.0 * max_mw


def _active_excitation(
    device_setting: dict[str, Any],
) -> tuple[str | None, float | None, float | None]:
    """Return ``(light_source, wavelength_nm, power_percent)`` for the active line.

    Scans the ``MultiLaser`` device settings for lines flagged as used and
    returns the highest-power one (device name, excitation wavelength, and the
    line power as a *percentage* of the device maximum). Use :func:`_line_power_mw`
    to convert the percentage to absolute mW.
    """
    used_lines: list[tuple[str | None, float | None, float]] = []
    prefix = "MultiLaser_LineUsedLineUsed"
    for key, used in device_setting.items():
        if not (key.startswith(prefix) and used):
            continue
        suffix = key[len(prefix) :]  # e.g. "0-03"
        device_index = suffix.split("-", 1)[0]
        wavelength = device_setting.get(f"MultiLaser_LineUsedLineWavelength{suffix}")
        power_percent = device_setting.get(f"MultiLaser_PowerLinePower{suffix}") or 0.0
        device = device_setting.get(f"MultiLaser_PowerKey{device_index}") or device_setting.get(
            f"MultiLaser_LineUsedKey{device_index}"
        )
        used_lines.append((device, wavelength, power_percent))

    if not used_lines:
        return None, None, None

    device, wavelength, power_percent = max(used_lines, key=lambda line: line[2])
    return device, wavelength, power_percent or None


def _first_value(mapping: dict[str, Any], prefix: str) -> Any:
    """Return the first value whose key starts with ``prefix`` (handles ``|`` suffixes)."""
    for key, value in mapping.items():
        if key.startswith(prefix):
            return value
    return None


def _tokenize(name: str) -> list[str]:
    """Lowercase alphanumeric runs in ``name`` (splitting on any other char)."""
    return re.findall(r"[^\W_]+", name.lower())


def _julian_to_datetime(julian_day: float) -> datetime:
    unix_seconds = (julian_day - _JD_UNIX_EPOCH) * 86_400.0
    return datetime.fromtimestamp(unix_seconds, tz=UTC)
