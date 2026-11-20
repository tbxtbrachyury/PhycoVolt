"""What is connected to each channel, and its physics: expected voltage and current, power.

Questions it answers: for a channel's circuit and what its DAC delivers, which voltage and current
should the channel see, where is the other end of the circuit, how much does each resistor
dissipate, and what reaches the culture through the i-v-i board?

Main choices: a circuit is one small description (spec 3.3.2), used by the checks and by the
simulator. Physics only, with Ohm's law and the real pin voltage (H-04): every judgement (windows,
limits) is in checks.py. The current sign follows the M1K: positive leaves the channel's terminal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from phycovolt import hardware as hw
from phycovolt import libsmu_math, units, waveforms
from phycovolt.hardware import Record, Role
from phycovolt.waveforms import BOARD, Waveform


class CircuitKind(Enum):
    RESISTOR = "resistor"             # source-i: channel -> R -> return pin
    LOAD_SHUNT = "load-shunt"         # source-v: channel -> load -> shunt -> return pin
    IVI_BOARD = "ivi-board"           # source-i: channel -> i-v-i board input (about 2.5 V)
    FREE = "free"                     # hi-z: measures a node; no physics check
    SHUNT_NODE = "shunt-node"         # hi-z: the node between the OTHER channel's load and shunt
    SHUNT = "shunt"                   # hi-z: a known resistor R whose other end is on a pin


ROLE_OF = {
    CircuitKind.RESISTOR: Role.SOURCE_I, CircuitKind.IVI_BOARD: Role.SOURCE_I,
    CircuitKind.LOAD_SHUNT: Role.SOURCE_V,
    CircuitKind.FREE: Role.HI_Z, CircuitKind.SHUNT_NODE: Role.HI_Z, CircuitKind.SHUNT: Role.HI_Z,
}
# The returns offered, the default first. GND is not offered for a current source on a resistor:
# near 0 A the pin would sit at 0 V, where saturation cannot be told from a good measurement.
RETURNS = {
    CircuitKind.RESISTOR: ("2.5V", "5V"),
    CircuitKind.LOAD_SHUNT: ("GND", "2.5V", "5V"),
    CircuitKind.IVI_BOARD: (BOARD,),
    CircuitKind.FREE: (None,),
    CircuitKind.SHUNT_NODE: (None,),
    CircuitKind.SHUNT: ("GND", "2.5V", "5V"),
}


def circuits_for(role: Role) -> list[CircuitKind]:
    return [kind for kind, circuit_role in ROLE_OF.items() if circuit_role is role]


@dataclass(frozen=True)
class Circuit:
    """One channel's circuit, as the operator describes it (ohms, volts)."""

    kind: CircuitKind
    ret: str | None = None            # return pin, BOARD, or None (free, shunt-node)
    r: float = 0.0                    # resistor: R; load-shunt: the load; shunt: R
    shunt: float = 0.0                # load-shunt: the shunt (0: no shunt)
    scale: float = 0.0                # ivi-board: culture current / M1K current, signed
    board_input: float = hw.PIN_NOMINAL["2.5V"]   # ivi-board: the board's input voltage

    def __post_init__(self) -> None:
        if self.ret not in RETURNS[self.kind]:
            raise ValueError(f"{self.kind.value} takes the returns {RETURNS[self.kind]}, not {self.ret!r}")
        if self.kind in (CircuitKind.RESISTOR, CircuitKind.LOAD_SHUNT, CircuitKind.SHUNT) and not self.r > 0:
            raise ValueError(f"{self.kind.value} needs a resistance above 0 ohm")
        if self.shunt < 0:
            raise ValueError("the shunt cannot be negative")
        if self.kind is CircuitKind.IVI_BOARD and self.scale == 0:
            raise ValueError("the ivi-board needs its scale factor (culture current / M1K current)")

    @property
    def ohms(self) -> float:
        """Resistance from the channel's terminal to the return (inf: no path; 0: the board's input)."""
        if self.kind is CircuitKind.LOAD_SHUNT:
            return self.r + self.shunt
        if self.kind in (CircuitKind.RESISTOR, CircuitKind.SHUNT):
            return self.r
        if self.kind is CircuitKind.IVI_BOARD:
            return 0.0
        return math.inf

    @property
    def nominal_return(self) -> float | None:
        """The return's nominal voltage (None without a return)."""
        if self.ret is None:
            return None
        return self.board_input if self.ret == BOARD else hw.PIN_NOMINAL[self.ret]


def default_circuit(role: Role) -> Circuit:
    """resistor 10 ohm to 2.5V (source-i); 47 ohm + 10 ohm shunt to GND (source-v); free (hi-z)."""
    if role is Role.SOURCE_I:
        return Circuit(CircuitKind.RESISTOR, "2.5V", r=10.0)
    if role is Role.SOURCE_V:
        return Circuit(CircuitKind.LOAD_SHUNT, "GND", r=47.0, shunt=10.0)
    return Circuit(CircuitKind.FREE)


# The typed fields of each circuit: key -> (Circuit attribute, unit). The ivi-board's scale factor
# has no default: it must come from the operator (a wrong one would mislabel the culture current).
CIRCUIT_FIELDS = {
    CircuitKind.RESISTOR: {"r": ("r", "Ω"), "return": ("ret", "")},
    CircuitKind.LOAD_SHUNT: {"load": ("r", "Ω"), "shunt": ("shunt", "Ω"), "return": ("ret", "")},
    CircuitKind.IVI_BOARD: {"scale": ("scale", ""), "input": ("board_input", "V")},
    CircuitKind.FREE: {},
    CircuitKind.SHUNT_NODE: {},
    CircuitKind.SHUNT: {"r": ("r", "Ω"), "return": ("ret", "")},
}
_STARTING = {
    CircuitKind.RESISTOR: {"r": 10.0}, CircuitKind.LOAD_SHUNT: {"r": 47.0, "shunt": 10.0},
    CircuitKind.SHUNT: {"r": 10.0},
}


def circuit_from_text(kind: CircuitKind, typed: dict[str, str], base: Circuit | None = None) -> Circuit:
    """A circuit from typed fields ("r": "4,7k", "return": "5V"), starting from `base` when it is of
    the same kind, or from the usual values (10 ohm; 47 ohm + 10 ohm; the first return offered)."""
    if base is not None and base.kind is kind:
        values = {"ret": base.ret, "r": base.r, "shunt": base.shunt, "scale": base.scale,
                  "board_input": base.board_input}
    else:
        values = {"ret": RETURNS[kind][0], **_STARTING.get(kind, {})}
    fields = CIRCUIT_FIELDS[kind]
    for key, text in typed.items():
        if key not in fields:
            raise ValueError(f"{kind.value} has no field {key!r} (fields: {list(fields)})")
        attribute, unit = fields[key]
        if attribute == "ret":
            values["ret"] = text.strip()
        elif unit:
            values[attribute] = units.value(text, unit)
        else:
            values[attribute] = float(text.replace(",", "."))
    return Circuit(kind, **values)


@dataclass(frozen=True)
class ChannelConfig:
    """What one channel is set to do: its role, its circuit and (when it sources) its waveform."""

    role: Role = Role.HI_Z
    circuit: Circuit = Circuit(CircuitKind.FREE)
    waveform: Waveform | None = None

    def __post_init__(self) -> None:
        if ROLE_OF[self.circuit.kind] is not self.role:
            raise ValueError(f"{self.circuit.kind.value} is not a {self.role.value} circuit")
        if (self.waveform is None) != (self.role is Role.HI_Z):
            raise ValueError("a sourcing channel needs a waveform; a hi-z channel has none")


def check_pair(a: ChannelConfig, b: ChannelConfig) -> None:
    """Refuse a pair of channels that cannot work together (a shunt-node without a shunt)."""
    for this, other in ((a, b), (b, a)):
        if this.circuit.kind is CircuitKind.SHUNT_NODE and not (
                other.circuit.kind is CircuitKind.LOAD_SHUNT and other.circuit.shunt > 0):
            raise ValueError("shunt-node needs the other channel on a load-shunt circuit with a shunt")


# --- What the DAC delivers -------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """What a sourcing channel will do over one period, before it is switched on."""

    requested: np.ndarray             # the waveform asked for (V or A)
    codes: np.ndarray                 # the DAC codes libsmu will send, with the stored calibration
    delivered: np.ndarray             # what the DAC delivers, if the stored calibration is exact
    voltage: np.ndarray               # the channel's expected voltage
    current: np.ndarray               # the channel's expected current (+ leaves the terminal)
    return_voltage: float             # the real voltage of the return used for the expectation
    timing: waveforms.Timing


def predict(config: ChannelConfig, channel: str, records: list[Record], return_voltage: float) -> Prediction:
    """The prediction of one period for a sourcing channel (C.3 and Ohm's law)."""
    kind = hw.SOURCE_KIND[config.role]
    record = records[hw.record_index(channel, kind)]
    requested = waveforms.period(config.waveform)
    codes = libsmu_math.dac_code(requested, kind, record)
    delivered = libsmu_math.source_output(codes, kind, record)
    voltage, current = expected(config, delivered, return_voltage)
    return Prediction(requested, codes, delivered, voltage, current, return_voltage,
                      waveforms.timing(config.waveform))


# --- Physics --------------------------------------------------------------------------------------


def pin_from_rest(rest_voltage: float, ohms: float, open_input: float) -> float:
    """Where the other end of a circuit is, from the channel's rest reading in HI_Z.

    With no current the resistors drop nothing, but the 1 MOhm input towards `open_input` (0.34 V
    on this unit) makes a divider with the circuit: undo it.
    """
    if not math.isfinite(ohms):
        return rest_voltage
    return open_input + (rest_voltage - open_input) * (hw.HI_Z_INPUT_OHMS + ohms) / hw.HI_Z_INPUT_OHMS


def expected(config: ChannelConfig, delivered: np.ndarray, return_voltage: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """(voltage, current) the channel should see for what its DAC delivers (V or A), with the real
    return voltage; no rails, no limiter (the checks look at those)."""
    circuit = config.circuit
    delivered = np.asarray(delivered, dtype=float)
    if config.role is Role.SOURCE_I:
        return return_voltage + delivered * circuit.ohms, delivered
    if config.role is Role.SOURCE_V:
        return delivered, (delivered - return_voltage) / circuit.ohms
    raise ValueError("a hi-z channel sources nothing")


def resistor_powers(config: ChannelConfig, current: np.ndarray) -> dict[str, float]:
    """Mean power (W) in each resistor of the circuit for a current waveform: I_rms^2 x R."""
    circuit = config.circuit
    mean_square = float(np.mean(np.square(current)))
    if circuit.kind is CircuitKind.RESISTOR:
        return {"resistor": mean_square * circuit.r}
    if circuit.kind is CircuitKind.LOAD_SHUNT:
        powers = {"load": mean_square * circuit.r}
        if circuit.shunt > 0:
            powers["shunt"] = mean_square * circuit.shunt
        return powers
    return {}


def shunt_node_voltage(other: ChannelConfig, other_current: np.ndarray, return_voltage: float) -> np.ndarray:
    """The voltage between the other channel's load and shunt: the return plus the shunt's drop."""
    return return_voltage + np.asarray(other_current, dtype=float) * other.circuit.shunt


def culture_current(config: ChannelConfig, m1k_current: np.ndarray) -> np.ndarray:
    """The current at the culture through the i-v-i board: scale factor x the M1K's current."""
    return config.circuit.scale * np.asarray(m1k_current, dtype=float)
