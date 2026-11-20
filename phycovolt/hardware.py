"""What the ADALM1000 (M1K) is, and how the program talks to any instrument.

Questions it answers:

- Which fixed numbers describe the M1K (spec appendix A.1)? This file is the ONLY place in the code
  for them; the measured values of one particular unit are in `unit.ini` (appendix A.3), read by
  `load_unit`.
- Which contract does an instrument follow (spec 7.2)? `Instrument` is implemented by the real M1K
  (`m1k.py`) and by the simulator (`simulator.py`); an own USB driver would be a third
  implementation. The rest of the program only sees this contract.

Main choices: the contract is small on purpose (identify, read the calibration, read at rest, stream
100 ms chunks, close). Every value is in volts and amperes as libsmu returns them, that is, with the
stored calibration already applied. An instrument error carries a short `code` for the program to
act on; the text the operator reads is written by checks.py and printed by __main__.py.
"""

from __future__ import annotations

import configparser
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ContextManager, Iterator, Protocol

import numpy as np

# --- Sampling and reading -------------------------------------------------------------------------

SAMPLE_RATE = 100_000                 # samples/s per channel; below it libsmu 1.0.3 hangs (L-12)
CHANNELS = ("A", "B")
CHUNK_SAMPLES = 10_000                # 100 ms: the unit of every read (I-05)
CHUNK_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE
READ_TIMEOUT_MS = 2_000               # per chunk; a chunk that does not arrive in time is an error
QUEUE_SAMPLES = 1_000_000             # libsmu's queue: the longest cyclic block and burst (10 s, L-08)
MAX_STREAM_CHUNKS = QUEUE_SAMPLES // CHUNK_SAMPLES
PARK_SAMPLES = 1_000                  # 10 ms in HI_Z put the DAC at the rest code (I-02)

# USB transfers: libsmu resets the "previous sample" of its current correction at the start of each
# transfer (read in libsmu 1.0.3's handle_in_transfer).
USB_PACKET_SAMPLES = 256
USB_TRANSFERS = 2
USB_BUFFER_SECONDS = {"win32": 0.050}  # every other platform: 0.020
USB_BUFFER_SECONDS_DEFAULT = 0.020
USB_ID = "064b:784c"
USB_ID_FIRMWARE_UPDATE = "03eb:6124"

# --- Ranges ---------------------------------------------------------------------------------------

V_MIN = 0.0                           # V, source and measure
V_MAX = 5.0
I_MIN = -0.2                          # A, source; the corrected request is clamped to this
I_MAX = 0.2
CHANNEL_CURRENT_LIMIT = 0.2           # A, the limiter; the ADM1177 records an overcurrent
PIN_CURRENT_LIMIT = 0.2               # A, per fixed pin
HI_Z_INPUT_OHMS = 1e6

# --- Converters -----------------------------------------------------------------------------------

CODES = 65_536                        # 16-bit DAC and ADC
CODE_MAX = CODES - 1
V_STEP = V_MAX / CODES                # 76.29 uV: ADC and DAC voltage step
ADC_I_SPAN = 0.4                      # ADC current word: I = (word * 0.4/65536 - 0.195) * 1.25
ADC_I_OFFSET = 0.195
ADC_I_GAIN = 1.25
ADC_I_STEP = ADC_I_SPAN / CODES * ADC_I_GAIN          # 7.629 uA
ADC_I_MIN = -ADC_I_OFFSET * ADC_I_GAIN                # -0.24375 A at code 0
ADC_I_MAX = ADC_I_MIN + CODE_MAX * ADC_I_STEP         # +0.25624 A at code 65535
ADC_V_MAX = CODE_MAX * V_STEP                         # 4.99992 V
DAC_I_BASE = 2.0 / 5.0               # DAC current code: trunc(65536 * (0.4 + 1.6 * I))
DAC_I_SCALE = 1.6
# libsmu writes the factor as 0.8*0.2*20.*0.5, which in double precision is 1.6000000000000003;
# libsmu_math uses this exact value so that its codes are libsmu's, truncation included.
DAC_I_SCALE_LIBSMU = 0.8 * 0.2 * 20.0 * 0.5
DAC_I_STEP = 1 / (CODES * DAC_I_SCALE)                # 9.537 uA
DAC_REST_CODE = 32_768 * 4 // 5       # 26 214: the code in HI_Z; 2.0 V in SVMI, -3.8 uA in SIMV (L-04)

# --- Pins and firmware ----------------------------------------------------------------------------

PINS = ("GND", "2.5V", "5V")
PIN_NOMINAL = {"GND": 0.0, "2.5V": 2.5, "5V": 5.0}   # V; the real ones differ (H-04): unit.ini
FIRMWARE_FOR_CALIBRATION = (2, 6)     # the EEPROM calibration
FIRMWARE_FOR_OVERCURRENT = (2, 7)     # libsmu reads the ADM1177 only from 2.07 on (L-09)


def samples_per_transfer(platform: str = sys.platform) -> int:
    """How many samples libsmu 1.0.3 puts in one USB transfer at SAMPLE_RATE (1024 on Linux)."""
    buffer_seconds = USB_BUFFER_SECONDS.get(platform, USB_BUFFER_SECONDS_DEFAULT)
    packets = math.ceil(buffer_seconds / (USB_PACKET_SAMPLES / SAMPLE_RATE) / USB_TRANSFERS)
    return packets * USB_PACKET_SAMPLES


def firmware_tuple(firmware: str) -> tuple[int, int]:
    """'2.17' -> (2, 17). Raises ValueError for a text that is not major.minor."""
    major, minor = firmware.strip().split(".")[:2]
    return int(major), int(minor)


# --- The calibration records ----------------------------------------------------------------------


class Kind(Enum):
    """One function of a channel, each with its own calibration record."""

    MEASURE_V = "measure V"
    MEASURE_I = "measure I"
    SOURCE_V = "source V"
    SOURCE_I = "source I"


@dataclass(frozen=True)
class Record:
    """One calibration record: raw value = true value / gain + offset (gain+ or gain- by the sign)."""

    offset: float = 0.0
    gain_pos: float = 1.0
    gain_neg: float = 1.0


# The order of the 8 records in the EEPROM, in calib.txt and in pysmu's `calibration`.
RECORD_ORDER = tuple((channel, kind) for channel in CHANNELS for kind in Kind)
DEFAULT_RECORDS = tuple(Record() for _ in RECORD_ORDER)


def record_index(channel: str, kind: Kind) -> int:
    """Position of a channel's record in RECORD_ORDER (CH A measure V = 0 … CH B source I = 7)."""
    return RECORD_ORDER.index((channel, kind))


# --- The instrument contract (spec 7.2) -----------------------------------------------------------


class Role(Enum):
    """What a channel does: the M1K mode it uses is in MODE_OF_ROLE."""

    HI_Z = "hi-z"
    SOURCE_V = "source-v"
    SOURCE_I = "source-i"


MODE_OF_ROLE = {Role.HI_Z: "HI_Z", Role.SOURCE_V: "SVMI", Role.SOURCE_I: "SIMV"}
SOURCE_KIND = {Role.SOURCE_V: Kind.SOURCE_V, Role.SOURCE_I: Kind.SOURCE_I}   # the record a role uses


@dataclass(frozen=True)
class ChannelSetup:
    """What one channel does during a stream.

    `block` is one period (or 100 ms of a level) at SAMPLE_RATE, in V for SOURCE_V and A for
    SOURCE_I, repeated cyclically for the whole stream; None for HI_Z.
    """

    role: Role = Role.HI_Z
    block: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.role is Role.HI_Z:
            if self.block is not None:
                raise ValueError("a hi-z channel has no block")
            return
        if self.block is None:
            raise ValueError(f"a {self.role.value} channel needs a block")
        block = np.array(self.block, dtype=float).reshape(-1)
        if not 1 <= block.size <= QUEUE_SAMPLES:
            raise ValueError(f"a block has 1 to {QUEUE_SAMPLES} samples, not {block.size}")
        if not np.all(np.isfinite(block)):
            raise ValueError("a block has a value that is not a number")
        block.setflags(write=False)
        object.__setattr__(self, "block", block)


@dataclass(frozen=True)
class Chunk:
    """Samples of both channels, as libsmu returns them (stored calibration applied); V and A."""

    index: int                        # chunk number since the start of the stream (0 for a rest read)
    v_a: np.ndarray
    i_a: np.ndarray
    v_b: np.ndarray
    i_b: np.ndarray

    @property
    def samples(self) -> int:
        return int(self.v_a.size)

    def channel(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(voltage, current) of channel "A" or "B"."""
        return (self.v_a, self.i_a) if name == "A" else (self.v_b, self.i_b)


class InstrumentError(Exception):
    """The instrument could not do what was asked; `code` says why, for the program to act on.

    Codes: "no_pysmu", "no_board", "several_boards", "firmware", "lost_sample", "timeout", "usb",
    "invalid_request".
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class Stream(Protocol):
    """The chunks of one burst. Valid only inside the `with` block that created it."""

    overcurrent: bool                 # valid after the stream ends

    def __iter__(self) -> Iterator[Chunk]: ...

    def stop(self) -> None:           # ends at the current chunk
        ...


class Instrument(Protocol):
    """What the program asks of an instrument (spec 7.2)."""

    description: str                  # "serial … · fw 2.17 · hw D" or "simulated (no board) · fw 2.17 · hw D"
    serial: str
    firmware: str
    hardware: str

    def calibration(self) -> list[Record]:
        """The 8 records stored in the EEPROM, in RECORD_ORDER."""
        ...

    def rest_read(self, seconds: float) -> Chunk:
        """Read with both channels explicitly in HI_Z (I-03)."""
        ...

    def stream(self, a: ChannelSetup, b: ChannelSetup, chunks: int) -> ContextManager[Stream]:
        """Source and measure for `chunks` chunks; leaving the `with` by any path leaves both
        channels in HI_Z; a lost sample raises InstrumentError, never a silent gap."""
        ...

    def close(self) -> None:
        """Both channels in HI_Z; the device released."""
        ...


# --- One unit's measured values (unit.ini) --------------------------------------------------------

UNIT_FILE = Path(__file__).with_name("unit.ini")


@dataclass(frozen=True)
class Unit:
    """The measured values of one M1K (spec A.3), used by the simulator to rehearse."""

    name: str
    serial: str
    firmware: str
    hardware: str
    true_records: tuple[Record, ...]          # in RECORD_ORDER
    stored_offset_shift: float                # the simulator's EEPROM = true records + these shifts
    stored_gain_shift: float                  # relative (40 ppm = 40e-6)
    pins: dict[str, float] = field(default_factory=dict)   # V, real voltage of each fixed pin
    open_input: float = 0.0                   # V read in HI_Z with nothing connected
    current_zero_slope: float = 0.0           # A/V: measured-current zero against the channel voltage
    voltage_sigma: dict[str, float] = field(default_factory=dict)   # V, white noise per sample, by channel
    current_sigma: float = 0.0                # A
    voltage_hum: float = 0.0                  # V, amplitude of the mains hum
    current_hum: float = 0.0                  # A
    open_input_pickup: float = 0.0            # V, hum on an open HI_Z input
    source_nonlinearity: dict[str, float] = field(default_factory=dict)   # V below the line at 4.5 V (H-06)
    mains: float = 60.0                       # Hz
    voltage_slow: float = 0.0                 # V, standard deviation of the slow noise
    current_slow: float = 0.0                 # A
    block_correlation: float = 0.0            # lag-1 correlation of the slow noise's 100 ms means


def _record(text: str) -> Record:
    offset, gain_pos, gain_neg = (float(part) for part in text.split(","))
    return Record(offset, gain_pos, gain_neg)


def load_unit(path: Path | None = None) -> Unit:
    """Read a unit file (default: the project's unit, `unit.ini` next to this file)."""
    parser = configparser.ConfigParser(inline_comment_prefixes=(";",))
    with open(path or UNIT_FILE, encoding="utf-8") as file:
        parser.read_file(file)
    unit = parser["unit"]
    records = parser["true_records"]
    true_records = tuple(
        _record(records[f"{channel.lower()}_{kind.name.lower()}"]) for channel, kind in RECORD_ORDER
    )
    stored = parser["stored_calibration"]
    pins = {name: parser["pins"].getfloat(name.lower()) for name in PINS}
    measure = parser["measure"]
    noise = parser["noise"]
    return Unit(
        name=unit["name"],
        serial=unit["serial"],
        firmware=unit["firmware"],
        hardware=unit["hardware"],
        true_records=true_records,
        stored_offset_shift=stored.getfloat("offset_shift"),
        stored_gain_shift=stored.getfloat("gain_shift_ppm") * 1e-6,
        pins=pins,
        open_input=measure.getfloat("open_input"),
        current_zero_slope=measure.getfloat("current_zero_slope"),
        voltage_sigma={channel: noise.getfloat(f"voltage_sigma_{channel.lower()}") for channel in CHANNELS},
        current_sigma=noise.getfloat("current_sigma"),
        voltage_hum=noise.getfloat("voltage_hum"),
        current_hum=noise.getfloat("current_hum"),
        open_input_pickup=noise.getfloat("open_input_pickup"),
        source_nonlinearity={channel: parser["source_nonlinearity"].getfloat(channel.lower()) for channel in CHANNELS},
        mains=noise.getfloat("mains"),
        voltage_slow=noise.getfloat("voltage_slow"),
        current_slow=noise.getfloat("current_slow"),
        block_correlation=noise.getfloat("block_correlation"),
    )
