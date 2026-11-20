"""The real M1K, through pysmu (libsmu 1.0.3). This is the ONLY file that imports pysmu.

Question it answers: how is the instrument driven so that nothing reaches the circuit (or the
culture) that was not asked for?

Main choices: every sequence below follows the operating rules I-01 to I-08 of the specification
(6.1), and each one is written out in a call-order test against a fake pysmu (tests/test_m1k.py).
A new sequence does not go to the instrument without the operator's approval.

- **open**: `Session(ignore_dataflow=False, queue_size=1 000 000)` (I-05; lost samples are errors);
  Python's own Ctrl+C handler back at once, because pysmu's ends the process without switching the
  channels off (I-04, P-02); exactly one device, firmware >= 2.06, 100 kS/s (I-07); then a 100 ms
  rest read, so that libsmu sets its end() timeout before any start/end (I-06, L-10).
- **rest read**: mode A = HI_Z, mode B = HI_Z, flush, get_samples(n). HI_Z is requested every time,
  because libsmu restores the last requested mode at the next read (I-03, L-06).
- **stream**: if a channel will source (SVMI or SIMV), a 10 ms rest read first ("park the DAC" at
  code 26 214: about 0 A in SIMV, 2.0 V in SVMI), so that a burst never starts at the previous
  burst's last code (I-02, L-04); flush (stale samples out); write each sourcing channel's block,
  cyclic (L-08); mode A, mode B, right before start (I-01, L-05); start(chunks x 10 000); read
  10 000 samples per chunk with a 2 s timeout, a missing sample being an error (I-05).
- **end of a stream, by any path** (all chunks read, stop, break, error, Ctrl+C): mode A = HI_Z and
  mode B = HI_Z first, because a mode change acts at once; cancel, if the burst did not finish;
  end; read the overcurrent flag (L-09). Every step is attempted even if an earlier one fails.
- **close**: mode A = HI_Z, mode B = HI_Z; the session is released.

The EEPROM is never written (I-08): pysmu's write_calibration is not called anywhere.
"""

from __future__ import annotations

import importlib
import signal
import threading
from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Iterator

import numpy as np

from phycovolt import hardware as hw
from phycovolt.hardware import ChannelSetup, Chunk, InstrumentError, Record, Role


def open_m1k(pysmu: ModuleType | None = None, import_name: str = "pysmu") -> "M1K":
    """Open the one connected M1K. `pysmu` is the module to use (tests pass a fake one)."""
    if pysmu is None:
        try:
            pysmu = importlib.import_module(import_name)
        except ImportError as error:
            raise InstrumentError("no_pysmu", str(error)) from error
    if threading.current_thread() is not threading.main_thread():
        # pysmu installs a signal handler when the session is created; Python allows that only in
        # the main thread.
        raise InstrumentError("invalid_request", "the M1K must be opened from the main thread")
    try:
        session = pysmu.Session(add_all=True, ignore_dataflow=False, sample_rate=0,
                                queue_size=hw.QUEUE_SAMPLES)
    except pysmu.LibsmuError as error:
        raise InstrumentError("usb", str(error)) from error
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)          # I-04
    devices = session.devices
    if not devices:
        raise InstrumentError("no_board")
    if len(devices) > 1:
        raise InstrumentError("several_boards", f"{len(devices)} devices")
    device = devices[0]
    try:
        firmware = hw.firmware_tuple(str(device.fwver))
    except ValueError as error:
        raise InstrumentError("firmware", f"unreadable firmware version {device.fwver!r}") from error
    if firmware < hw.FIRMWARE_FOR_CALIBRATION:
        raise InstrumentError("firmware", f"firmware {device.fwver}")
    if session.sample_rate != hw.SAMPLE_RATE:
        raise InstrumentError("invalid_request", f"sample rate {session.sample_rate}")
    instrument = M1K(pysmu, session, device)
    instrument._rest(hw.CHUNK_SAMPLES)                                    # I-06
    return instrument


def _chunk(index: int, samples: list) -> Chunk:
    """pysmu's ((v_a, i_a), (v_b, i_b)) per sample, as four arrays."""
    data = np.asarray(samples, dtype=np.float64).reshape(-1, 2, 2)
    return Chunk(index, data[:, 0, 0].copy(), data[:, 0, 1].copy(), data[:, 1, 0].copy(),
                 data[:, 1, 1].copy())


class _M1KStream:
    """The chunks of one burst on the real instrument."""

    def __init__(self, instrument: "M1K", chunks: int) -> None:
        self._instrument = instrument
        self.chunks = chunks
        self.chunks_read = 0
        self.overcurrent = False
        self._stopped = False

    @property
    def complete(self) -> bool:
        return self.chunks_read == self.chunks

    def stop(self) -> None:
        self._stopped = True

    def __iter__(self) -> Iterator[Chunk]:
        while self.chunks_read < self.chunks and not self._stopped:
            samples = self._instrument._read_chunk()
            index = self.chunks_read
            self.chunks_read += 1
            yield _chunk(index, samples)


class M1K:
    """The real instrument (implements hardware.Instrument). Create it with `open_m1k`."""

    def __init__(self, pysmu: ModuleType, session: object, device: object) -> None:
        self._pysmu = pysmu
        self._session = session
        self._device = device
        self._busy = False
        self.serial = str(device.serial)
        self.firmware = str(device.fwver)
        self.hardware = str(device.hwver)
        self.description = f"serial {self.serial} · fw {self.firmware} · hw {self.hardware}"
        self._mode = {Role.HI_Z: pysmu.Mode.HI_Z, Role.SOURCE_V: pysmu.Mode.SVMI,
                      Role.SOURCE_I: pysmu.Mode.SIMV}

    @property
    def reports_overcurrent(self) -> bool:
        """libsmu reads the overcurrent flag only from firmware 2.07 on (L-09)."""
        return hw.firmware_tuple(self.firmware) >= hw.FIRMWARE_FOR_OVERCURRENT

    # --- reading the EEPROM ----------------------------------------------------------------------

    def calibration(self) -> list[Record]:
        self._require_open()
        with self._errors():
            stored = [tuple(float(value) for value in record) for record in self._device.calibration]
        if len(stored) != len(hw.RECORD_ORDER) or any(len(record) != 3 for record in stored):
            raise InstrumentError("usb", f"unexpected calibration shape {stored!r}")
        return [Record(*record) for record in stored]

    # --- reading at rest -------------------------------------------------------------------------

    def rest_read(self, seconds: float) -> Chunk:
        samples = round(seconds * hw.SAMPLE_RATE)
        if not 1 <= samples <= hw.QUEUE_SAMPLES:
            raise InstrumentError("invalid_request", f"a rest read of {seconds} s")
        return self._rest(samples)

    def _rest(self, samples: int) -> Chunk:
        """Both channels explicitly in HI_Z, then one blocking read (I-03; P-03: short reads only)."""
        self._require_open()
        self._busy = True
        try:
            with self._errors():
                self._device.channels["A"].mode = self._mode[Role.HI_Z]
                self._device.channels["B"].mode = self._mode[Role.HI_Z]
                self._session.flush()
                data = self._device.get_samples(samples)
            if len(data) != samples:
                raise InstrumentError("lost_sample", f"{len(data)} of {samples} samples")
            return _chunk(0, data)
        except BaseException:
            self._hi_z_quietly()
            raise
        finally:
            self._busy = False

    # --- streaming -------------------------------------------------------------------------------

    @contextmanager
    def stream(self, a: ChannelSetup, b: ChannelSetup, chunks: int) -> Iterator[_M1KStream]:
        """Source and measure for `chunks` chunks of 100 ms (sequence in the module docstring)."""
        self._require_open()
        if not 1 <= chunks <= hw.MAX_STREAM_CHUNKS:
            raise InstrumentError("invalid_request", f"{chunks} chunks (1 to {hw.MAX_STREAM_CHUNKS})")
        if a.role is not Role.HI_Z or b.role is not Role.HI_Z:
            self._rest(hw.PARK_SAMPLES)                                   # I-02: before every burst that sources
        stream = _M1KStream(self, chunks)
        self._busy = True
        try:
            try:
                with self._errors():
                    self._session.flush()
                    for name, setup in zip(hw.CHANNELS, (a, b)):
                        if setup.role is not Role.HI_Z:
                            self._device.channels[name].write(setup.block.tolist(), cyclic=True)
                    self._device.channels["A"].mode = self._mode[a.role]      # I-01: right
                    self._device.channels["B"].mode = self._mode[b.role]      # before the start
                    self._session.start(chunks * hw.CHUNK_SAMPLES)
            except BaseException:
                self._hi_z_quietly()          # the burst never started: no cancel, no end
                raise
            try:
                yield stream
            except BaseException:
                self._finish(stream, quiet=True)
                raise
            self._finish(stream, quiet=False)
        finally:
            self._busy = False

    def _read_chunk(self) -> list:
        with self._errors():
            samples = self._device.read(hw.CHUNK_SAMPLES, hw.READ_TIMEOUT_MS)
        if len(samples) != hw.CHUNK_SAMPLES:
            raise InstrumentError(
                "timeout", f"{len(samples)} of {hw.CHUNK_SAMPLES} samples in {hw.READ_TIMEOUT_MS} ms")
        return samples

    def _finish(self, stream: _M1KStream, quiet: bool) -> None:
        """End a burst: HI_Z first, then cancel (if unfinished), end, overcurrent. Every step is
        tried; with `quiet` (an error is already on its way) their failures are not raised."""
        stream.stop()                     # the stream object cannot read any more after this
        steps: list[Callable[[], object]] = [
            lambda: self._set_mode("A", Role.HI_Z),
            lambda: self._set_mode("B", Role.HI_Z),
        ]
        if not stream.complete:
            steps.append(self._session.cancel)
        steps.append(self._session.end)

        def read_overcurrent() -> None:
            stream.overcurrent = bool(self._device.overcurrent)

        steps.append(read_overcurrent)
        errors = _attempt_all(steps)
        if any(isinstance(error, KeyboardInterrupt) for error in errors):
            raise KeyboardInterrupt
        if errors and not quiet:
            raise InstrumentError("usb", f"ending the burst: {errors[0]}") from errors[0]

    # --- closing ---------------------------------------------------------------------------------

    def close(self) -> None:
        if self._session is None:
            return
        self._hi_z_quietly()
        self._session = None
        self._device = None

    # --- helpers ---------------------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._session is None:
            raise InstrumentError("invalid_request", "the M1K is closed")
        if self._busy:
            raise InstrumentError("invalid_request", "the M1K is already reading")

    def _set_mode(self, channel: str, role: Role) -> None:
        with self._errors():
            self._device.channels[channel].mode = self._mode[role]

    def _hi_z_quietly(self) -> None:
        _attempt_all([lambda: self._set_mode("A", Role.HI_Z), lambda: self._set_mode("B", Role.HI_Z)])

    @contextmanager
    def _errors(self) -> Iterator[None]:
        """pysmu's exceptions as InstrumentError."""
        try:
            yield
        except self._pysmu.SampleDrop as error:
            raise InstrumentError("lost_sample", str(error)) from error
        except self._pysmu.LibsmuError as error:
            raise InstrumentError("usb", str(error)) from error


def _attempt_all(steps: list[Callable[[], object]]) -> list[BaseException]:
    """Run every step, even when one fails (a second Ctrl+C included); return the failures."""
    errors: list[BaseException] = []
    for step in steps:
        try:
            step()
        except BaseException as error:  # noqa: BLE001 - the remaining steps must still run
            errors.append(error)
    return errors
