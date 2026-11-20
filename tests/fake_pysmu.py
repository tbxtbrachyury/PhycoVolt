"""A stand-in for pysmu 1.0.3 that records every call and touches nothing.

It has the names m1k.py uses, with pysmu's signatures (read in libsmu v1.0.3's
bindings/python/pysmu), and it can inject the failures the call-order tests need: no board, two
boards, old firmware, a lost sample, a short (timed-out) read, a Ctrl+C in the middle of a read.

`make()` returns a fresh module-like object and its log. Every entry of the log is a tuple, for
example ("mode", "A", "SIMV"), ("write", "A", 10000, True), ("start", 30000), ("read", 10000, 2000).
"""

from __future__ import annotations

import signal
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Mode(Enum):
    HI_Z = 0
    SVMI = 1
    SIMV = 2
    HI_Z_SPLIT = 3
    SVMI_SPLIT = 4
    SIMV_SPLIT = 5


class LibsmuError(Exception):
    def __init__(self, msg: str | None = None, errcode: int | None = None) -> None:
        super().__init__(msg)
        self.errcode = errcode


class SessionError(LibsmuError):
    pass


class DeviceError(LibsmuError):
    pass


class DataflowError(LibsmuError):
    pass


class SampleDrop(DataflowError):
    pass


class WriteTimeout(DataflowError):
    pass


# A sample as pysmu returns it: ((v_a, i_a), (v_b, i_b)).
Sample = tuple[tuple[float, float], tuple[float, float]]


def rest_sample(index: int) -> Sample:
    """What a channel at rest on GND roughly reads."""
    return ((0.0001, 0.00001), (0.0002, -0.00001))


@dataclass
class Faults:
    """Failures to inject; a read number counts the device.read calls from 1."""

    devices: int = 1
    firmware: str = "2.17"
    drop_at_read: int | None = None          # SampleDrop raised by that read
    short_at_read: int | None = None         # that read returns fewer samples (a timeout)
    interrupt_at_read: int | None = None     # KeyboardInterrupt raised inside that read
    drop_in_get_samples: bool = False        # SampleDrop raised by get_samples
    usb_error_on_mode: bool = False          # DeviceError when a mode is set
    sample_rate: int | None = None           # the session runs at another rate than 100 000/s
    calibration: list | None = None          # the 8 stored records (default: offset 0, gains 1)


@dataclass
class Fake:
    """One fake pysmu module, with its log and its faults."""

    faults: Faults = field(default_factory=Faults)
    log: list[tuple] = field(default_factory=list)
    sample: Callable[[int], Sample] = rest_sample
    reads: int = 0
    sessions: list["FakeSession"] = field(default_factory=list)

    def module(self) -> types.SimpleNamespace:
        fake = self

        class Session(FakeSession):
            def __init__(self, add_all: bool = True, ignore_dataflow: bool = True,
                         sample_rate: int = 0, queue_size: int | None = None) -> None:
                super().__init__(fake, add_all, ignore_dataflow, sample_rate, queue_size)

        return types.SimpleNamespace(
            Session=Session, Mode=Mode, LibsmuError=LibsmuError, SessionError=SessionError,
            DeviceError=DeviceError, DataflowError=DataflowError, SampleDrop=SampleDrop,
            WriteTimeout=WriteTimeout, __version__="1.0.3",
        )


def make(faults: Faults | None = None, sample: Callable[[int], Sample] = rest_sample) -> Fake:
    return Fake(faults=faults or Faults(), sample=sample)


def _killing_handler(signum: int, frame: object) -> None:   # what pysmu installs (P-02)
    raise AssertionError("pysmu's SIGINT handler ran: m1k.py did not restore Python's handler")


class FakeChannel:
    def __init__(self, fake: Fake, name: str) -> None:
        self._fake = fake
        self.name = name
        self._mode = Mode.HI_Z
        self.data: list[float] = []

    @property
    def mode(self) -> Mode:
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        if mode not in Mode:
            raise ValueError(f"invalid mode: {mode}")
        self._fake.log.append(("mode", self.name, mode.name))
        if self._fake.faults.usb_error_on_mode:
            raise DeviceError(f"failed setting mode {mode}: ", 5)
        self._mode = mode

    def write(self, data: list[float], cyclic: bool = False) -> None:
        self._fake.log.append(("write", self.name, len(data), cyclic))
        self.data = list(data)

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"m1k.py must not call Channel.{name} (P-01: no pysmu waveforms)")


class FakeDevice:
    def __init__(self, fake: Fake, index: int) -> None:
        self._fake = fake
        self.serial = f"FAKE{index:027d}"
        self.fwver = fake.faults.firmware
        self.hwver = "D"
        self.channels = OrderedDict([("A", FakeChannel(fake, "A")), ("B", FakeChannel(fake, "B"))])
        stored = fake.faults.calibration or [[0.0, 1.0, 1.0] for _ in range(8)]
        self.calibration = [list(record) for record in stored]
        self._overcurrent = 0

    @property
    def overcurrent(self) -> int:
        self._fake.log.append(("overcurrent",))
        return self._overcurrent

    def get_samples(self, num_samples: int) -> list[Sample]:
        self._fake.log.append(("get_samples", num_samples))
        if self._fake.faults.drop_in_get_samples:
            raise SampleDrop("data sample dropped")
        return [self._fake.sample(k) for k in range(num_samples)]

    def read(self, num_samples: int, timeout: int = 0, skipsamples: bool = False) -> list[Sample]:
        self._fake.log.append(("read", num_samples, timeout))
        self._fake.reads += 1
        faults = self._fake.faults
        if faults.drop_at_read == self._fake.reads:
            raise SampleDrop("data sample dropped")
        if faults.interrupt_at_read == self._fake.reads:
            raise KeyboardInterrupt
        count = num_samples // 2 if faults.short_at_read == self._fake.reads else num_samples
        start = (self._fake.reads - 1) * num_samples
        return [self._fake.sample(start + k) for k in range(count)]

    def write_calibration(self, file: object) -> None:
        raise AssertionError("I-08: the program never writes the EEPROM")

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"m1k.py must not call Device.{name}")


class FakeSession:
    def __init__(self, fake: Fake, add_all: bool, ignore_dataflow: bool, sample_rate: int,
                 queue_size: int | None) -> None:
        self._fake = fake
        fake.log.append(("Session", {"add_all": add_all, "ignore_dataflow": ignore_dataflow,
                                     "sample_rate": sample_rate, "queue_size": queue_size}))
        self.queue_size = 100_000 if queue_size is None else queue_size
        self.ignore_dataflow = ignore_dataflow
        self.sample_rate = fake.faults.sample_rate or sample_rate or 100_000
        self.devices = tuple(FakeDevice(fake, k) for k in range(fake.faults.devices)) if add_all else ()
        fake.sessions.append(self)
        signal.signal(signal.SIGINT, _killing_handler)     # pysmu does this in Session.__init__

    def start(self, samples: int) -> None:
        self._fake.log.append(("start", samples))

    def end(self) -> None:
        self._fake.log.append(("end",))

    def cancel(self) -> None:
        self._fake.log.append(("cancel",))

    def flush(self) -> None:
        self._fake.log.append(("flush",))

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"m1k.py must not call Session.{name}")
