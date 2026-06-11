# Design proposal: separating imaging modality from channel

Status: Implemented in 0.5.0
Scope: `channels.py`, `display.py` (new), `blending.py`, `nikon.py`, `leica.py` (public `Channel` API)

## Problem

The library uses a single flat type, `Channel`, to represent several
distinct concepts. The code itself acknowledges this: the Leica parser holds a
set of `Channel` objects in a variable named `_CRS_LASER_MODALITIES`.

```python
# leica.py
_CRS_LASER_MODALITIES = {SRS, E_CARS, F_CARS, E_SHG, F_SHG}
```

A `Channel` today is `(name, color, excitation_nm, emission_nm)`, and the
predefined constants in `channels.py` quietly span four unrelated categories:

| Constant(s)                          | What it actually represents                                   |
| ------------------------------------ | ------------------------------------------------------------- |
| `BRIGHTFIELD`, `DIC`, `PHASE`        | Label-free **contrast techniques** (no ex/em; color is white/gray) |
| `DAPI`, `FITC`, `TRITC`, `CY5`       | **Fluorophores/dyes** with intrinsic ex/em (a sample-prep concept) |
| `SRS`, `E_CARS`, `F_CARS`, `E_SHG`, `F_SHG` | Nonlinear **contrast modalities**; ex/em are computed from the CRS laser at load time, not intrinsic |
| `E_*` vs `F_*` prefixes              | **Detection geometry** (epi vs forward) baked into the name   |

So five orthogonal ideas are flattened into one frozen dataclass keyed by `name`:

1. **Modality / contrast mechanism** — how contrast is formed.
2. **Detection geometry** — epi (reflected / non-descanned) vs forward (trans).
3. **Fluorophore / label** — what is in the sample.
4. **Spectral properties** — excitation / emission wavelengths.
5. **Display color** — a pure visualization concern.

## Why it bites in practice

### 1. Predefined constants are templates, not values

`SRS`/`E_CARS`/etc. carry no wavelengths. The Leica loader maps a detector to
the `SRS` template, then constructs a **new** `Channel` with computed
wavelengths:

```python
# leica.py (infer_channel_from_detector → _compute_crs_wavelengths)
return Channel(
    name=channel.name,
    excitation_nm=round(pump_wavelength_nm, 1),
    emission_nm=round(emission_nm, 1),
)
```

Because `Channel` is `frozen=True` and compared by all fields, the SRS channel
that ends up in `image.channels` is **not equal** to the `SRS` constant. Any
downstream code doing `if ch == SRS` or `ch in CHANNELS.values()` silently
fails. The internal `channel in (E_CARS, F_CARS)` check only works because it
runs on the *template* before the rebuild.

### 2. Detection geometry leaks into identity

`E_CARS`/`F_CARS` share a color and the same contrast modality (CARS), differing
only in detection geometry. Same for `E_SHG`/`F_SHG`. Encoding geometry in the
channel name forces the detection map to disambiguate `BRIGHTFIELD` vs `F_SHG`
on the same detector with a runtime warning.

### 3. Display color is part of physical identity

`color` is a rendering choice, but it participates in equality/hashing. Two
acquisitions of the same fluorophore rendered with different LUTs would compare
as different channels.

## Model: channel-as-identity with modality-dependent facets

A channel is **not** "composed of" modality + geometry. A channel is the
**atomic acquisition unit** — one detection configuration that yields one image
plane — and its *identity* is that configuration. Modality, detection geometry,
and spectral bands are *descriptive facets*, and crucially **the set of relevant
facets depends on the modality**:

| Modality | Identifying facets beyond the modality itself |
| --- | --- |
| Brightfield / DIC / phase | (none — the technique is the identity) |
| Fluorescence | excitation / emission bands |
| SRS | excitation / emission (Raman) wavelengths |
| CARS / SHG | wavelengths **and** detection geometry (epi vs forward) |

Detection geometry is therefore **not a universal co-equal axis**: it is a
free, identifying choice only for CARS and SHG. For the other modalities it is
fixed by physics (transmitted for label-free/SRS, epi for fluorescence) and is
exposed as a constant rather than stored.

This is expressed as a small class hierarchy rather than one flat dataclass with
many optional fields, so each modality carries exactly the facets that are real
for it:

```python
class Modality(Enum):
    BRIGHTFIELD; DIC; PHASE; FLUORESCENCE; SRS; CARS; SHG

class DetectionGeometry(Enum):
    EPI       # reflected / non-descanned
    FORWARD   # transmitted

@dataclass(frozen=True)
class Channel(ABC):                       # identity = optical configuration
    modality: ClassVar[Modality]
    # geometry is declared per concrete leaf: ClassVar where fixed, field where free
    @property
    @abstractmethod
    def name(self) -> str: ...

class BrightfieldChannel(Channel): ...    # no fields; geometry = FORWARD
class DICChannel(Channel): ...
class PhaseChannel(Channel): ...

@dataclass(frozen=True, kw_only=True)
class SpectralChannel(Channel):           # adds spectral facets
    excitation_nm: float | None = None
    emission_nm: float | None = None
    label: str = field(default="", compare=False)   # name is metadata, NOT identity

class FluorescenceChannel(SpectralChannel): ...     # geometry = EPI (fixed)
class SRSChannel(SpectralChannel): ...              # geometry = FORWARD (fixed)
class CARSChannel(SpectralChannel):
    geometry: DetectionGeometry = DetectionGeometry.EPI   # identifying field
class SHGChannel(SpectralChannel):
    geometry: DetectionGeometry = DetectionGeometry.EPI   # identifying field
```

Resolved decisions:

- **Modality drives behavior, not a set of constants.** `_CRS_LASER_MODALITIES`
  became `_NONLINEAR_MODALITIES = {Modality.SRS, Modality.CARS, Modality.SHG}`,
  and the Leica loader fills wavelengths with `dataclasses.replace(template, …)`,
  which preserves modality, geometry, and label.
- **Fluorophore is sample-prep, not a channel facet.** It is intentionally *not*
  modeled on `Channel`. A fluorescence channel is identified by its spectral
  bands; the dye that happens to emit there belongs to sample metadata.
- **Display color is not on `Channel`.** It moved to `display.channel_color()`
  (derived from emission wavelength for spectral channels; fixed neutral colors
  for label-free modalities), with an optional per-`Layer` override.
- **Channels are hashable and usable as dict keys.** Frozen dataclasses; `label`
  is `compare=False` so loader-rebuilt channels compare/hash equal to their
  presets, and `dict[Channel, …]` (e.g. `overlay_channels`) behaves intuitively.

## Resolved open questions

- *Should fluorophore live on `Channel`?* No — it is sample prep; removed.
- *Where should display color live?* In `display.py`, resolved at viz time.
- *Do we need stable, hashable channel identity?* Yes — identity is the optical
  configuration (excluding the human-readable label), so channels are safe dict
  keys.
