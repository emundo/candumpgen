import abc
import binascii
import copy
import logging
import random
import time
from typing import cast, Any, Dict, List, NamedTuple, Optional, Sequence, Set, Union

import can
import cantools
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import truncnorm

from .maybe_open import Openable, maybe_open

LOGGER = logging.getLogger(__name__)


class CANDumpLine(NamedTuple):
    timestamp: float
    interface: str
    identifier: int
    body: bytes

    def __str__(self) -> str:
        return "({:.6f}) {} {}#{}\n".format(
            self.timestamp,
            self.interface,
            ("000" + hex(self.identifier)[2:].upper())[-3:],
            binascii.hexlify(self.body).decode("ASCII").upper()
        )


class Point(NamedTuple):
    x: float
    y: float


def generate_candump(
    candump_file: Openable,
    dbc_file: Openable,
    simulated_interface: str = "can0",
    simulation_time_ms: int = 1000000
) -> None:
    """
    Args:
        candump_file: The name/path of the candump file to generate or an opened writable file-like object
            representing the file.
        dbc_file: The name/path of the DBC file to generate a realistic candump off of, or an opened readable
            file-like object representing the file.
        simulated_interface: The name of the interface to simulate, e.g. "can0".
        simulation_time_ms: The number of milliseconds to simulate.
    """

    with maybe_open(dbc_file, "r") as f:
        db: cantools.db.Database = cantools.database.load(f)

    simulation_start = time.time()
    simulated_messages = [ SimulatedMessage(
        message,
        simulation_time_ms,
        simulated_interface,
        simulation_start
    ) for message in db.messages ]

    with maybe_open(candump_file, "w") as f:
        for simulated_ms in range(simulation_time_ms):

            def value_after(x: SimulatedMessage, simulated_ms: int = simulated_ms) -> Optional[CANDumpLine]:
                return x.value_after(simulated_ms)

            candump_lines = map(value_after, simulated_messages)
            for candump_line in filter(lambda x: x is not None, candump_lines):
                f.write(str(candump_line))


def cangen(dbc_file: Openable, interface: str, simulation_time_ms: int = 1000000) -> None:
    """
    Args:
        dbc_file: The name/path of the DBC file to generate a realistic candump off of, or an opened readable
            file-like object representing the file.
        interface: The name of the interface to send to, e.g. "vcan0".
        simulation_time_ms: The number of milliseconds to simulate.
    """

    with maybe_open(dbc_file, "r") as f:
        db: cantools.db.Database = cantools.database.load(f)

    simulation_start = time.time()
    simulated_messages = [ SimulatedMessage(
        message,
        simulation_time_ms,
        interface,
        simulation_start
    ) for message in db.messages ]

    can_bus = can.interface.Bus(interface, bustype="socketcan")
    for simulated_ms in range(simulation_time_ms):

        def value_after(x: SimulatedMessage, simulated_ms: int = simulated_ms) -> Optional[CANDumpLine]:
            return x.value_after(simulated_ms)

        candump_lines = map(value_after, simulated_messages)
        for candump_line in filter(lambda x: x is not None, candump_lines):
            assert candump_line is not None  # This is always the thanks to above call to filter
            can_bus.send(can.Message(arbitration_id=candump_line.identifier, data=candump_line.body))


class SimulatedMessage:
    def __init__(
        self,
        message: cantools.db.Message,
        simulation_time_ms: int,
        simulated_interface: str,
        simulation_start: float
    ) -> None:
        """
        Args:
            message: The CAN message definition to simulate.
            simulation_time_ms: The number of milliseconds to run the simulation.
            simulated_interface: The name of the interface to simulate, e.g. "can0".
            simulation_start The (unix) timestamp of the simulation start, used to generate realistic
                timestamps in the candump log file.
        """

        # The average delay between packets of this simulated message, in milliseconds
        self.__delay_ms = random.randrange(1000) + 10
        self.__global_signals: Set[SimulatedSignal] = set()
        self.__multiplexer_signal: Optional[SimulatedMultiplexerSignal] = None
        self.__multiplexed_signals: Dict[int, Set[SimulatedSignal]] = {}
        self.__message = message
        self.__simulated_interface = simulated_interface
        self.__simulation_start = simulation_start
        self.__next_packet_elapsed_time_ms = 0

        LOGGER.debug("Generating a packet every %s ms", self.__delay_ms)

        # Find multiplexer signals
        multiplexer_signals = list(filter(lambda x: x.is_multiplexer, message.signals))

        # Only a single multiplexer signal is supported
        if len(multiplexer_signals) > 1:
            raise ValueError("Only a single multiplexer signal per CAN message is supported.")

        if len(multiplexer_signals) == 1:
            multiplexer_signal = multiplexer_signals[0]

            # Get all values the multiplexer signal can become
            multiplexer_values: Set[int] = set.union(*[
                set(signal.multiplexer_ids or [])
                for signal in message.signals
                if signal.multiplexer_signal == multiplexer_signal.name
            ])

            LOGGER.debug("Multiplexer signal with values %s", multiplexer_values)

            self.__multiplexer_signal = SimulatedMultiplexerSignal(
                multiplexer_values,
                multiplexer_signal,
                simulation_time_ms
            )

        for signal in message.signals:
            # The presence of choices marks this signal as an enum signal. The signal creation is wrapped in a
            # lambda as a poor-mans laziness replacement.
            def simulated_signal(signal: cantools.db.Signal = signal) -> SimulatedSignal:
                if signal.choices is None:  # pylint: disable=no-else-return
                    return SimulatedValueSignal(signal, simulation_time_ms)
                else:
                    return SimulatedEnumSignal(signal, simulation_time_ms)

            # A potential multiplexer signal is handled above
            if not signal.is_multiplexer:
                if signal.multiplexer_ids is None:
                    self.__global_signals.add(simulated_signal())
                else:
                    if len(signal.multiplexer_ids) != 1:
                        raise Exception("Signal multiplexed by multiple values, which is not supported.")

                    multiplexer_value = signal.multiplexer_ids[0]

                    self.__multiplexed_signals[multiplexer_value] = self.__multiplexed_signals.get(
                        multiplexer_value,
                        set()
                    )

                    self.__multiplexed_signals[multiplexer_value].add(simulated_signal())

        self.__prepare_next()

    def __prepare_next(self) -> None:
        """
        Prepares for the next packet output.
        """

        # TODO: Jitter maybe?
        self.__next_packet_elapsed_time_ms += self.__delay_ms

    def value_after(self, elapsed_time_ms: int) -> Optional[CANDumpLine]:
        """
        Args:
            elapsed_time_ms: The (simulated) time that has elapsed since the simulation has started, in
                milliseconds.

        Returns:
            If applicable for this simulation timestamp, a candump line to add to the generated log file.
        """

        if elapsed_time_ms >= self.__next_packet_elapsed_time_ms:
            # Query all signals for their values after this elapsed time
            signal_values = {}

            # Global signals
            for signal in self.__global_signals:
                signal_values[signal.name] = signal.value_after(elapsed_time_ms)

            # The multiplexer and multiplexed signals
            if self.__multiplexer_signal is not None:
                multiplexer_value = self.__multiplexer_signal.value_after(elapsed_time_ms)

                signal_values[self.__multiplexer_signal.name] = multiplexer_value

                for signal in self.__multiplexed_signals[multiplexer_value]:
                    signal_values[signal.name] = signal.value_after(elapsed_time_ms)

            timestamp = self.__simulation_start + elapsed_time_ms / 1000.
            interface = self.__simulated_interface
            identifier = self.__message.frame_id
            body = self.__message.encode(signal_values)

            # Append the simulated log line to the output list
            candump_line = CANDumpLine(
                timestamp=timestamp,
                interface=interface,
                identifier=identifier,
                body=body
            )

            self.__prepare_next()

            return candump_line

        return None


class SimulatedSignal(abc.ABC):
    def __init__(self, signal: cantools.db.Signal, simulation_time_ms: int) -> None:
        """
        Args:
            signal: The signal to be simulated by this instance.
            simulation_time_ms: The number of milliseconds to run the simulation.
        """

        self._signal = signal
        self._simulation_time_ms = simulation_time_ms

    @abc.abstractmethod
    def value_after(self, elapsed_time_ms: int) -> Union[float, int, str]:
        """
        Args:
            elapsed_time_ms: The (simulated) time that has elapsed since the simulation has started, in
                milliseconds.

        Returns:
            The simulated value for that timestamp.
        """

    @property
    def name(self) -> str:
        return cast(str, self._signal.name)


class SimulatedMultiplexerSignal(SimulatedSignal):
    def __init__(self, values: Set[int], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Generate a random sequence of multiplexer values which makes for realistic signal values. The length
        # of the sequence and the relative weight of each enum value in the sequence are randomized.

        population = list(values)
        sequence_length = random.randrange(1, int(2 * np.log(self._simulation_time_ms // 100)))
        weights = np.random.choice(range(100), size=len(population))
        weights = weights / sum(weights)
        self.__sequence = np.random.choice(population, size=sequence_length, p=weights)
        self.__index = 0

        LOGGER.debug("Multiplexer field sequence: %s", self.__sequence)

    def value_after(self, elapsed_time_ms: int) -> int:
        # Each time the value is read, the next entry in the sequence is returned
        self.__index += 1
        return int(self.__sequence[self.__index % len(self.__sequence)])


class SimulatedEnumSignal(SimulatedSignal):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Generate a random sequence of enum values which makes for realistic signal values. The length of the
        # sequence and the relative weight of each enum value in the sequence are randomized.

        population = list(self._signal.choices.values())
        sequence_length = random.randrange(1, int(2 * np.log(self._simulation_time_ms // 100)))
        weights = np.random.choice(range(100), size=len(population))
        weights = weights / sum(weights)

        # This isn't truly a sequence of ints, but it can be treated as such
        self.__sequence: Sequence[int] = np.random.choice(population, size=sequence_length, p=weights)

    def value_after(self, elapsed_time_ms: int) -> int:
        # Each index is served for 100 (simulated) millisecond, the sequence is repeated periodically.
        return self.__sequence[(elapsed_time_ms // 100) % len(self.__sequence)]


class SimulatedValueSignal(SimulatedSignal):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Generate a few random points that represent values at points in time and smoothly interpolate
        # between them which makes for realistic signal values. The following parameters are randomized:
        # - The average x-distance between generated points
        # - The variance of the x-distance between adjacent generated points
        # - The variance of the y-distance between adjacent generated points
        # A set of points is generated according to those parameters, which are then interpolated using a
        # "Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)", which makes for a smooth function that
        # doesn't overshoot the interpolated points.

        # Generate the point distribution standard deviation in x and y direction
        xstddev = random.random()
        ystddev = random.random()

        # If an initial value is not defined, generate a random one
        initial = self._signal.initial
        if initial is None:
            initial = self.__adjust_y(self.__generate_random_point(xstddev, ystddev).y)

        # Generate at most one point per 5 simulated seconds and at least one per simulated minute
        degree = random.randrange(
            self._simulation_time_ms // 60000,
            self._simulation_time_ms // 5000
        )

        LOGGER.debug(
            "value signal %s; standard deviation (x/y): %.2f / %.2f; degree: %s",
            self._signal.name,
            xstddev,
            ystddev,
            degree
        )

        # Generate the (random) anchor points
        points = [ Point(x=0, y=initial) ]
        for index in range(degree):
            point = self.__generate_random_point(xstddev, ystddev)
            points.append(Point(
                x=int(((index + point.x + 0.5) * self._simulation_time_ms) / degree + 0.5),
                y=self.__adjust_y(point.y)
            ))

        # Add a fixed point at the very last millisecond of the simulation.
        points.append(Point(
            x=max(self._simulation_time_ms, points[-1].x) + 1,
            y=self.__adjust_y(self.__generate_random_point(xstddev, ystddev).y)
        ))

        # Interpolate the function
        self.__function = PchipInterpolator([ p.x for p in points ], [ p.y for p in points ])

    def value_after(self, elapsed_time_ms: int) -> float:
        return float(self.__function(elapsed_time_ms))

    def __adjust_y(self, y: float) -> float:
        """
        Args:
            y: A floating point value between 0 and 1 to adjust to the value ranges of the signal.

        Returns:
            The y coordinate, adjusted to offset, minimum, maximum and step size required by the signal.
        """

        scale = self._signal.scale
        offset = self._signal.offset
        length = self._signal.length

        if self._signal.is_float:
            if length not in [ 32, 64 ]:  # TODO: Test floats
                raise Exception("Float has unknown bit width.")

            if self._signal.length == 32:
                raw_minimum = np.finfo(np.float32).min
                raw_maximum = np.finfo(np.float32).min

            if self._signal.length == 64:
                raw_minimum = np.finfo(np.float64).min
                raw_maximum = np.finfo(np.float64).min
        else:
            if self._signal.is_signed:
                raw_minimum = 0 - 2 ** (length - 1)
                raw_maximum = 2 ** (length - 1) - 1
            else:
                raw_minimum = 0
                raw_maximum = 2 ** length - 1

        minimum: float = self._signal.minimum
        if minimum is None:
            # It is important that the division is performed before the subtraction to avoid an underflow.
            minimum = raw_minimum * scale + offset

        maximum: float = self._signal.maximum
        if maximum is None:
            maximum = raw_maximum * scale + offset

        return minimum + y * (maximum - minimum)

    @staticmethod
    def __generate_random_point(xstddev: float, ystddev: float) -> Point:
        """
        Args:
            xstddev: The standard deviation in x direction.
            ystddev: The standard deviation in y direction.

        Returns:
            A random point with x and y coordinates between 0 and 1.
        """

        xmean = 0.5  # TODO: Pass too?
        ymean = 0.5  # TODO: Pass too?
        xlower = 0.05
        xupper = 0.95
        ylower = 0.
        yupper = 1.

        x_dist = truncnorm(
            (xlower - xmean) / xstddev,
            (xupper - xmean) / xstddev,
            loc=xmean,
            scale=xstddev
        )

        y_dist = truncnorm(
            (ylower - ymean) / ystddev,
            (yupper - ymean) / ystddev,
            loc=ymean,
            scale=ystddev
        )

        return Point(x=x_dist.rvs(), y=y_dist.rvs())


MAXIMUM_IDENTIFIER_DISTANCE = 100.
MAXIMUM_SIGNAL_DISTANCE = 10.
BIT_MISMATCH_DISTANCE = .25
FLOAT_MISMATCH_DISTANCE = 2.
MULTIPLEXER_MISMATCH_DISTANCE = 5.
MULTIPLEXED_STATUS_MISMATCH_DISTANCE = 2.5
MULTIPLEXED_VALUE_MISMATCH_DISTANCE = 7.5


def dbc_dist(dbc_a: Openable, dbc_b: Openable) -> float:
    """
    Calculate the "distance" between two DBC files, used for performance measuring. Compares only those
    properties which physically influence the data on the wire, like signal bit position and size. Purely
    semantic properties like comments, signal names or units are not compared.

    Args:
        dbc_a, dbc_b: The DBC files to compare. Either a path-like object pointing to the files, or an opened
            readable file-like object.

    Returns:
        The "distance".
    """

    dist = 0.

    with maybe_open(dbc_a, "r") as f:
        db_a: cantools.db.Database = cantools.database.load(f)
    with maybe_open(dbc_b, "r") as f:
        db_b: cantools.db.Database = cantools.database.load(f)

    # Use the following to dump the database in a nice ASCII-art format.
    # cantools.subparsers.dump._dump_can_database(db_a)
    # cantools.subparsers.dump._dump_can_database(db_b)

    frame_ids_a: Set[int] = set(message.frame_id for message in db_a.messages)
    frame_ids_b: Set[int] = set(message.frame_id for message in db_b.messages)

    common_frame_ids = frame_ids_a & frame_ids_b
    differing_frame_ids = (frame_ids_a | frame_ids_b) - common_frame_ids

    # Frames that have no counterpart are assigned a static (maximum) distance.
    dist += len(differing_frame_ids) * MAXIMUM_IDENTIFIER_DISTANCE
    LOGGER.info(
        "%s frames without counterpart; punishment (per frame/total): %s/%s",
        len(differing_frame_ids),
        MAXIMUM_IDENTIFIER_DISTANCE,
        len(differing_frame_ids) * MAXIMUM_IDENTIFIER_DISTANCE
    )

    for frame_id in common_frame_ids:
        message_a: cantools.db.Message = db_a.get_message_by_frame_id(frame_id)
        message_b: cantools.db.Message = db_b.get_message_by_frame_id(frame_id)

        if message_a.length != message_b.length:
            # Frames of different (byte) sizes are assigned a static (maximum) distance.
            dist += MAXIMUM_IDENTIFIER_DISTANCE
            LOGGER.info(
                "Byte sizes of frame %s differ; punishment: %s",
                frame_id,
                MAXIMUM_IDENTIFIER_DISTANCE
            )
        else:
            shortest_signal_assignment_dist = _find_shortest_dist_signal_assignment(
                message_a.signals,
                message_b.signals,
                message_a.length
            )
            dist += shortest_signal_assignment_dist
            LOGGER.info(
                "Shortest signal assignment distance for frame %s: %s",
                frame_id,
                shortest_signal_assignment_dist
            )

    return dist


def _get_affected_bytes(signal: cantools.db.Signal) -> Set[int]:
    """
    Args:
        signal: The signal.

    Returns:
        The indizes of all bytes in the CAN frame affected by this signal.
    """

    first_affected_byte = signal.start // 8
    last_affected_byte = (signal.start + signal.length - 1) // 8

    if signal.byte_order == "big_endian":  # TODO
        # Little endian signals grow (sawtooth) into more-significant bytes, while big endian signals grow
        # into less-significant bytes.
        last_affected_byte = 2 * first_affected_byte - last_affected_byte

        first_affected_byte, last_affected_byte = last_affected_byte, first_affected_byte

    return set(range(first_affected_byte, last_affected_byte + 1))


def signal_dist(signal_a: cantools.db.Signal, signal_b: cantools.db.Signal, message_size: int) -> float:
    """
    Calculate the "distance" between two CAN/DBC signals, used for performance measuring. Compares only those
    properties which physically influence the data on the wire, like signal bit position and size. Purely
    semantic properties like comments, signal names or units are not compared.

    Args:
        signal_a, signal_b: The signals to compare.
        message_size: The size of the CAN/DBC message, in bytes.

    Returns:
        The "distance".
    """

    # Signals are only compared if their endianness matches or the endianness does not make a difference (i.e.
    # one of the signals doesn't cross byte borders). If that's not the case, the maximum distance is
    # returned.
    if signal_a.byte_order != signal_b.byte_order:
        affected_bytes_a = _get_affected_bytes(signal_a)
        affected_bytes_b = _get_affected_bytes(signal_b)

        # The endianness doesn't match; does at least one of the signals not cross byte borders?
        if min(len(affected_bytes_a), len(affected_bytes_b)) <= 1:
            # One of the signals doesn't cross byte borders! Adjust the endianness of that signal to the
            # endianness of the other signal to allow comparison of the starting and ending bit positions.

            # Swap the signal which doesn't cross byte borders into signal a to make the following code
            # easier:
            if len(affected_bytes_b) <= 1:
                signal_a, signal_b = signal_b, signal_a

            # Adjust the endianness of signal a to that of signal b.
            if signal_b.byte_order == "big_endian":
                # Convert from little to big endian:
                signal_a_start = signal_a.start + signal_a.length - 1
            else:
                # Convert from big to little endian:
                signal_a_start = signal_a.start - signal_a.length + 1

            # Build the updated signal.
            signal_a = copy.deepcopy(signal_a)
            signal_a.start = signal_a_start
            signal_a.byte_order = signal_b.byte_order
        else:
            return MAXIMUM_SIGNAL_DISTANCE

    dist = 0.

    # Get "normalized" start and end positions of both signals.
    signal_a_start = signal_a.start
    signal_b_start = signal_b.start

    signal_a_end = signal_a_start + signal_a.length
    signal_b_end = signal_b_start + signal_b.length

    if signal_a.byte_order == "big_endian":
        # At this point both signals are guaranteed to have the same endianness. In DBC's style of counting
        # bits, little endian signals naturally count up even across byte borders, while big endian signals
        # "jump" at byte borders. Normalize big endian signals to little endian, so that the distance between
        # the start and end bits of the signals can be easily calculated by subtracting them.

        signal_a_start = (message_size - 1 - (signal_a_start // 8)) * 8 + signal_a_start % 8
        signal_b_start = (message_size - 1 - (signal_b_start // 8)) * 8 + signal_b_start % 8

    # Get the bit position difference of signal start and end and add a distance based on these differences.
    start_difference = abs(signal_a_start - signal_b_start)
    end_difference = abs(signal_a_end - signal_b_end)

    # TODO: Exponential punishment?
    dist += start_difference * BIT_MISMATCH_DISTANCE
    dist += end_difference * BIT_MISMATCH_DISTANCE

    # Add a static value to the distance if one signal is an IEEE 754 float and the other one isn't.
    if signal_a.is_float != signal_b.is_float:
        dist += FLOAT_MISMATCH_DISTANCE

    # Add a static value to the distance if one signal is marked as a multiplexer signal and the
    # other one isn't.
    if signal_a.is_multiplexer != signal_b.is_multiplexer:
        dist += MULTIPLEXER_MISMATCH_DISTANCE

    # This distance measure only supports one multiplexer per CAN/DBC message definition
    multiplexer_id_a = None if signal_a.multiplexer_ids is None else signal_a.multiplexer_ids[0]
    multiplexer_id_b = None if signal_b.multiplexer_ids is None else signal_b.multiplexer_ids[0]

    # Add a static value to the distance if one signal is marked as being multiplexed and the other one isn't.
    if (multiplexer_id_a is None) != (multiplexer_id_b is None):
        dist += MULTIPLEXED_STATUS_MISMATCH_DISTANCE

    # Add an even bigger static value to the distance if both signals are marked as being multiplexed but the
    # multiplexer values don't match.
    if multiplexer_id_a != multiplexer_id_b:
        dist += MULTIPLEXED_VALUE_MISMATCH_DISTANCE

    return dist


def _find_shortest_dist_signal_assignment(
    signals_a: List[cantools.db.Signal],
    signals_b: List[cantools.db.Signal],
    message_size: int
) -> float:
    """
    Assign the signals of message a to the signals of message b, in the manner which results in the
    shortest cumulative "distance" between the signals.

    Args:
        signals_a, signals_b: The lists of signals to assign.
        message_size: The size of the CAN/DBC message, in bytes.

    Returns:
        The cumulative signal "distance" when using the optimal (= shortest distance) assignment.
    """

    min_len = min(len(signals_a), len(signals_b))
    max_len = max(len(signals_a), len(signals_b))

    if min_len == 0:
        # Signals which have no counterpart are assigned a static (maximum) distance.
        return max_len * MAXIMUM_SIGNAL_DISTANCE

    distances = set()
    for signal_a in signals_a:
        remaining_signals_a = list(signals_a)
        remaining_signals_a.remove(signal_a)

        for signal_b in signals_b:
            remaining_signals_b = list(signals_b)
            remaining_signals_b.remove(signal_b)

            signal_distance = signal_dist(signal_a, signal_b, message_size)
            remaining_distance = _find_shortest_dist_signal_assignment(
                remaining_signals_a,
                remaining_signals_b,
                message_size
            )

            distances.add(signal_distance + remaining_distance)

    return min(distances)
