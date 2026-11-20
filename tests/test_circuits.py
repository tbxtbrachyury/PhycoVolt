"""Do the circuits describe what the operator mounted, with the right physics? (spec 3.3.2)"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phycovolt import circuits as ci
from phycovolt import hardware as hw
from phycovolt import waveforms as wf
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import Role
from phycovolt.waveforms import Shape


def test_returns_offered() -> None:
    assert ci.RETURNS[CircuitKind.RESISTOR] == ("2.5V", "5V")            # no GND for a current source
    assert ci.RETURNS[CircuitKind.LOAD_SHUNT][0] == "GND"
    with pytest.raises(ValueError):
        Circuit(CircuitKind.RESISTOR, "GND", r=10.0)


def test_circuits_for_each_role() -> None:
    assert ci.circuits_for(Role.SOURCE_I) == [CircuitKind.RESISTOR, CircuitKind.IVI_BOARD]
    assert ci.circuits_for(Role.SOURCE_V) == [CircuitKind.LOAD_SHUNT]
    assert ci.circuits_for(Role.HI_Z) == [CircuitKind.FREE, CircuitKind.SHUNT_NODE, CircuitKind.SHUNT]


def test_a_circuit_needs_its_numbers() -> None:
    with pytest.raises(ValueError):
        Circuit(CircuitKind.LOAD_SHUNT, "GND", r=0.0)
    with pytest.raises(ValueError):
        Circuit(CircuitKind.IVI_BOARD, wf.BOARD)                          # no scale factor


def test_role_and_circuit_must_match() -> None:
    with pytest.raises(ValueError):
        ChannelConfig(Role.SOURCE_V, ci.default_circuit(Role.SOURCE_I),
                      wf.default_waveform(Role.SOURCE_V, Shape.DC, "GND"))
    with pytest.raises(ValueError):
        ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I), None)


def test_shunt_node_needs_a_shunt_on_the_other_channel() -> None:
    node = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE))
    no_shunt = ChannelConfig(Role.SOURCE_V, Circuit(CircuitKind.LOAD_SHUNT, "GND", r=57.0),
                             wf.default_waveform(Role.SOURCE_V, Shape.SINE, "GND"))
    with pytest.raises(ValueError):
        ci.check_pair(no_shunt, node)
    ci.check_pair(ChannelConfig(Role.SOURCE_V, ci.default_circuit(Role.SOURCE_V),
                                wf.default_waveform(Role.SOURCE_V, Shape.SINE, "GND")), node)


def test_circuit_from_typed_fields() -> None:
    circuit = ci.circuit_from_text(CircuitKind.RESISTOR, {"r": "4,7k", "return": "5V"})
    assert circuit == Circuit(CircuitKind.RESISTOR, "5V", r=4700.0)
    board = ci.circuit_from_text(CircuitKind.IVI_BOARD, {"scale": "0,01", "input": "2.48 V"})
    assert board.scale == pytest.approx(0.01) and board.board_input == pytest.approx(2.48)
    kept = ci.circuit_from_text(CircuitKind.RESISTOR, {"return": "2.5V"}, base=circuit)
    assert kept.r == 4700.0
    with pytest.raises(ValueError):
        ci.circuit_from_text(CircuitKind.RESISTOR, {"load": "47"})


def test_pin_from_rest_undoes_the_input_divider() -> None:
    open_input = 0.34
    rest = open_input + (2.4457 - open_input) * 1e6 / (1e6 + 50e3)
    assert ci.pin_from_rest(rest, 50e3, open_input) == pytest.approx(2.4457)
    assert ci.pin_from_rest(1.0, math.inf, open_input) == 1.0


def test_expected_values_follow_ohms_law() -> None:
    source_i = ChannelConfig(Role.SOURCE_I, Circuit(CircuitKind.RESISTOR, "2.5V", r=10.0),
                             wf.default_waveform(Role.SOURCE_I, Shape.DC, "2.5V"))
    voltage, current = ci.expected(source_i, np.array([0.01, -0.01]), 2.4457)
    assert voltage == pytest.approx([2.5457, 2.3457]) and current.tolist() == [0.01, -0.01]
    source_v = ChannelConfig(Role.SOURCE_V, ci.default_circuit(Role.SOURCE_V),
                             wf.default_waveform(Role.SOURCE_V, Shape.DC, "GND"))
    voltage, current = ci.expected(source_v, np.array([1.0]), 0.0)
    assert current == pytest.approx([1.0 / 57.0])


def test_powers_in_each_resistor() -> None:
    source_v = ChannelConfig(Role.SOURCE_V, ci.default_circuit(Role.SOURCE_V),
                             wf.default_waveform(Role.SOURCE_V, Shape.DC, "GND"))
    powers = ci.resistor_powers(source_v, np.full(10, 0.05))
    assert powers == pytest.approx({"load": 0.05**2 * 47, "shunt": 0.05**2 * 10})


def test_prediction_includes_the_dac_truncation() -> None:
    config = ChannelConfig(Role.SOURCE_I, Circuit(CircuitKind.RESISTOR, "2.5V", r=10.0),
                           wf.Waveform(Shape.DC, {"level": 0.0, "seconds": 1.0}))
    prediction = ci.predict(config, "A", list(hw.DEFAULT_RECORDS), 2.5)
    assert int(prediction.codes[0]) == hw.DAC_REST_CODE
    assert float(prediction.delivered[0]) == pytest.approx(-3.815e-6, abs=1e-9)


def test_culture_current_through_the_board() -> None:
    board = ChannelConfig(Role.SOURCE_I, Circuit(CircuitKind.IVI_BOARD, wf.BOARD, scale=-0.011),
                          wf.default_waveform(Role.SOURCE_I, Shape.SINE, wf.BOARD))
    assert ci.culture_current(board, np.array([2e-3])) == pytest.approx([-22e-6])
