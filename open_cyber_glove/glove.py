import serial
import struct
import zlib
import time
import numpy as np
from typing import Optional, Any, Tuple, List
from dataclasses import dataclass
import threading
import queue
from tqdm import tqdm
import logging

# for advanced calibration
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_cyber_glove.visualizer import HandVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GloveSensorData:
    """
    Data structure containing all sensor readings from the cyber glove.
    
    Attributes:
        tensile_data: Raw tensile sensor values from 19 sensors across the glove
        acc_data: 3-axis accelerometer data (x, y, z) in m/s²
        gyro_data: 3-axis gyroscope data (x, y, z) in rad/s
        mag_data: 3-axis magnetometer data (x, y, z) in μT
        temperature: Temperature reading from the glove in Celsius
        timestamp: Microsecond timestamp of the data packet
    """
    tensile_data: Tuple[int, ...]
    acc_data: Tuple[float, ...]
    gyro_data: Tuple[float, ...]
    mag_data: Tuple[float, ...]
    temperature: float
    timestamp: int

class Glove:
    """
    Abstract base class for cyber glove device management.
    
    Provides comprehensive functionality for connecting to, reading from, and calibrating
    a single data glove (left or right hand). Handles serial communication, data parsing,
    sensor calibration, and real-time data streaming with thread-safe operations.
    
    The class manages a background reader thread that continuously polls the serial port
    and maintains a thread-safe queue of the most recent sensor data packets.
    """
    # Protocol constants
    PACKET_SIZE = 132
    CRC_DATA_SIZE = 120
    TENSILE_DATA_OFFSET = 0
    TENSILE_DATA_SIZE = 76  # 19 int32 * 4 bytes
    ACC_DATA_OFFSET = 76
    ACC_DATA_SIZE = 12      # 3 float * 4 bytes
    GYRO_DATA_OFFSET = 88
    GYRO_DATA_SIZE = 12     # 3 float * 4 bytes
    MAG_DATA_OFFSET = 100
    MAG_DATA_SIZE = 12      # 3 float * 4 bytes
    TEMP_DATA_OFFSET = 112
    TEMP_DATA_SIZE = 4      # 1 float * 4 bytes
    TIMESTAMP_OFFSET = 116
    TIMESTAMP_SIZE = 4      # 1 uint32 * 4 bytes
    NUM_TENSILE_SENSORS = 19
    MODEL_OUTPUT_SIZE = 22
    NUM_IMU_AXES = 3
    SENSOR_MAX_VALUE = 8192 * 2
    DEFAULT_BAUDRATE = 1000000
    READ_TIMEOUT = 5.0  # seconds to wait for data before assuming the stream stalled
    SENSOR_ORDER = [3, 1, 0, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 2, 7, 11, 15]

    def __init__(self, hand_type: str):
        """
        Initialize a new glove instance for the specified hand.
        
        Args:
            hand_type: Specifies which hand this glove represents ('left' or 'right')
            
        Note:
            Initializes calibration arrays, data queue, and threading components.
            The data queue can hold up to 10 seconds of data at 120 Hz sampling rate.
        """
        self.hand_type = hand_type
        self.serial_port: Optional[serial.Serial] = None
        self.min_val = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        self.max_val = [0] * self.NUM_TENSILE_SENSORS
        self.avg_val = [0.0] * self.NUM_TENSILE_SENSORS
        self.custom_tuning = np.ones(self.MODEL_OUTPUT_SIZE, dtype=np.float32)
        self.is_calibrated = False
        self._data_queue = queue.Queue(maxsize=1200)  # 10 seconds of data at 120 Hz
        self._reader_thread = None
        self._reader_running = threading.Event()
        self._queue_lock = threading.Lock()
        self._buffer = bytearray()

    def connect(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        """
        Establish serial connection to the glove device.
        
        Args:
            port: Serial port identifier (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication baud rate (default: 1,000,000 bps)
            
        Note:
            Opens a serial connection with 1-second timeout for read operations.
        """
        self.serial_port = serial.Serial(port, baudrate, timeout=1)

    def start_reader(self):
        """
        Start the background data reading thread.
        
        Creates and starts a daemon thread that continuously reads data packets
        from the serial port and stores them in the thread-safe data queue.
        If a reader thread is already running, this method does nothing.
        """
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._reader_running.set()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop_reader(self):
        """
        Stop the background data reading thread.
        
        Signals the reader thread to stop and waits for it to complete.
        Cleans up the thread reference after termination.
        """
        self._reader_running.clear()
        if self._reader_thread is not None:
            self._reader_thread.join()
            self._reader_thread = None
        # Release the serial device so the port can be reopened by a later
        # connect()/start(); otherwise it stays locked for the process lifetime
        # and a second start() raises "device or resource busy".
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception as e:
                logger.error(f"Error closing serial port: {e}")
            self.serial_port = None

    def _reader_loop(self):
        """
        Main loop for the background data reading thread.
        
        Continuously monitors the serial port for incoming data, buffers it,
        and searches for valid data packets. A valid packet is identified by
        its CRC checksum. Once a valid packet is found, it's added to a
        thread-safe queue. This approach handles potential data stream
        misalignment. If the queue is full, the oldest packet is dropped.
        
        Note:
            This method runs in a separate thread and handles exceptions gracefully
            by logging errors and continuing operation.
        """
        while self._reader_running.is_set():
            try:
                # Read all available data from serial and add to buffer
                if self.serial_port and self.serial_port.in_waiting > 0:
                    data_in = self.serial_port.read(self.serial_port.in_waiting)
                    self._buffer.extend(data_in)

                # Process buffer to find packets
                while len(self._buffer) >= self.PACKET_SIZE:
                    found_packet = False
                    for i in range(len(self._buffer) - self.PACKET_SIZE + 1):
                        possible_packet = self._buffer[i:i + self.PACKET_SIZE]
                        if self._is_valid_data(possible_packet):
                            try:
                                # Additional validation for tensile data range
                                tensile_data = struct.unpack(f'<{self.NUM_TENSILE_SENSORS}i', possible_packet[self.TENSILE_DATA_OFFSET:self.TENSILE_DATA_OFFSET + self.TENSILE_DATA_SIZE])
                                if not all(0 <= val <= self.SENSOR_MAX_VALUE for val in tensile_data):
                                    continue  # Skip packet with invalid sensor values

                                with self._queue_lock:
                                    if self._data_queue.full():
                                        self._data_queue.get_nowait()  # Drop oldest
                                    self._data_queue.put_nowait(bytes(possible_packet))
                                
                                # Remove processed packet and any preceding bytes from buffer
                                self._buffer = self._buffer[i + self.PACKET_SIZE:]
                                found_packet = True
                                break  # Restart search from the beginning of the modified buffer
                            except struct.error:
                                continue # Could not unpack, so not a valid packet.

                    if not found_packet:
                        # No valid packet found, discard bytes that cannot form a full packet
                        # Keep the last part of the buffer that might be an incomplete packet
                        self._buffer = self._buffer[-(self.PACKET_SIZE - 1):]
                        break # Break from the inner while loop to wait for more data

                # If no data is available, sleep briefly to avoid busy-waiting
                if not (self.serial_port and self.serial_port.in_waiting > 0):
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in reader loop: {e}")
                time.sleep(0.01)

    def get_raw_data(self, timeout: Optional[float] = None) -> bytes:
        """
        Retrieve the most recent raw data packet from the queue.

        Args:
            timeout: Maximum seconds to wait for a packet. None waits indefinitely.

        Returns:
            The most recent complete data packet as bytes

        Raises:
            RuntimeError: If the serial port is not connected, the reader thread is
                not running, or no data arrives within ``timeout`` seconds.

        Note:
            This method blocks until at least one data packet is available (or the
            timeout expires). It drains all packets from the queue and returns only
            the most recent one.
        """
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        if self._reader_thread is None or not self._reader_thread.is_alive():
            raise RuntimeError(
                "Reader thread is not running. Call start_reader() "
                "(or OpenCyberGlove.start()) before reading data."
            )
        deadline = (time.time() + timeout) if timeout is not None else None
        # Wait for at least one data packet
        while self._data_queue.empty():
            if deadline is not None and time.time() >= deadline:
                raise RuntimeError(
                    f"No data received within {timeout:.1f}s "
                    f"(glove stalled or disconnected?)."
                )
            time.sleep(0.001)
        # Drain all but the last
        with self._queue_lock:
            last = None
            while not self._data_queue.empty():
                last = self._data_queue.get()
        return last

    def parse_raw_data(self, raw: bytes) -> GloveSensorData:
        """
        Convert raw binary data packet into structured sensor data.
        
        Args:
            raw: Raw binary data packet from the glove
            
        Returns:
            GloveSensorData object containing parsed sensor readings
            
        Raises:
            ValueError: If the raw data cannot be parsed due to format issues
            
        Note:
            Parses tensile sensors, IMU data (accelerometer, gyroscope, magnetometer),
            temperature, and timestamp according to the defined packet structure.
        """
        try:
            tensile_data = struct.unpack(f'<{self.NUM_TENSILE_SENSORS}i', raw[self.TENSILE_DATA_OFFSET:self.TENSILE_DATA_OFFSET + self.TENSILE_DATA_SIZE])
            acc_data = struct.unpack(f'<{self.NUM_IMU_AXES}f', raw[self.ACC_DATA_OFFSET:self.ACC_DATA_OFFSET + self.ACC_DATA_SIZE])
            gyro_data = struct.unpack(f'<{self.NUM_IMU_AXES}f', raw[self.GYRO_DATA_OFFSET:self.GYRO_DATA_OFFSET + self.GYRO_DATA_SIZE])
            mag_data = struct.unpack(f'<{self.NUM_IMU_AXES}f', raw[self.MAG_DATA_OFFSET:self.MAG_DATA_OFFSET + self.MAG_DATA_SIZE])
            temperature = struct.unpack('<f', raw[self.TEMP_DATA_OFFSET:self.TEMP_DATA_OFFSET + self.TEMP_DATA_SIZE])[0]
            timestamp = struct.unpack('<I', raw[self.TIMESTAMP_OFFSET:self.TIMESTAMP_OFFSET + self.TIMESTAMP_SIZE])[0]
            return GloveSensorData(
                tensile_data=tensile_data,
                acc_data=acc_data,
                gyro_data=gyro_data,
                mag_data=mag_data,
                temperature=temperature,
                timestamp=timestamp
            )
        except struct.error as e:
            raise ValueError(f"Failed to parse raw data: {e}")
        
    def get_data(self, timeout: Optional[float] = None) -> GloveSensorData:
        """Get the most recent parsed sensor data from the glove."""
        raw_data = self.get_raw_data(timeout=timeout)
        return self.parse_raw_data(raw_data)

    def get_all_raw_data(self) -> List[bytes]:
        """
        Drain and return every buffered raw packet in arrival order.

        Unlike :meth:`get_raw_data`, which discards all but the most recent packet,
        this method returns the complete backlog so no frames are lost. This is the
        primitive used by frame-complete recording.

        Returns:
            A list of raw packets (oldest first). Empty if no data is buffered.

        Raises:
            RuntimeError: If the serial port is not connected.
        """
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        with self._queue_lock:
            raws = []
            while not self._data_queue.empty():
                raws.append(self._data_queue.get())
        return raws

    def get_all_data(self) -> List[GloveSensorData]:
        """
        Drain and parse every buffered packet in arrival order (frame-complete).

        Returns:
            A list of parsed :class:`GloveSensorData` (oldest first). Empty if no
            data is buffered. Does not block.
        """
        return [self.parse_raw_data(raw) for raw in self.get_all_raw_data()]

    def flush(self) -> None:
        """
        Discard all packets currently buffered in the queue.

        Call this right before a fresh recording so the first captured frames
        correspond to when recording actually starts — not data that accumulated
        earlier (e.g. while the operator was reading a prompt). Safe to call while
        the reader thread is running.
        """
        with self._queue_lock:
            while not self._data_queue.empty():
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break

    def record(
        self,
        samples_count: Optional[int] = 500,
        duration: Optional[float] = None,
        prompt: Optional[str] = "Please perform the action you want to record.",
        wait_for_enter: bool = True,
        stall_timeout: Optional[float] = 5.0,
    ) -> List[GloveSensorData]:
        """
        Frame-complete recording from this glove.

        Unlike :meth:`data_collection` / :meth:`get_data` (which return only the most
        recent frame per call and therefore drop intermediate packets), this method
        drains every buffered packet in arrival order, so no frames are lost as long
        as the queue is serviced faster than it fills (a ~1200-frame / 10 s buffer at
        120 Hz).

        Recording stops when ``samples_count`` frames have been collected or
        ``duration`` seconds have elapsed, whichever comes first. At least one of the
        two limits must be provided.

        Args:
            samples_count: Target number of frames to record (None to record purely
                by duration).
            duration: Maximum recording duration in seconds (None to record purely by
                frame count).
            prompt: Instruction shown to the operator before recording starts.
            wait_for_enter: If True, wait for the operator to press Enter before
                recording (set False for programmatic/non-interactive recording).
            stall_timeout: If no new frames arrive for this many seconds, stop early
                and keep what was collected (guards against an infinite wait when the
                stream stalls/disconnects). None disables the watchdog.

        Returns:
            List of parsed :class:`GloveSensorData`, oldest first.

        Raises:
            RuntimeError: If the serial port is not connected.
            ValueError: If neither ``samples_count`` nor ``duration`` is provided.
        """
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        if samples_count is None and duration is None:
            raise ValueError("Provide at least one of samples_count or duration.")
        if prompt is not None and wait_for_enter:
            print(f"[{self.hand_type}] Press Enter to start recording: {prompt}")
            input()

        # Drop frames buffered before recording starts (e.g. while the operator
        # was reading the prompt) so frame 0 is a post-start sample.
        self.flush()

        frames: List[GloveSensorData] = []
        start = time.time()
        last_data = start
        with tqdm(total=samples_count, desc=f"[{self.hand_type}] recording") as pbar:
            while True:
                if samples_count is not None and len(frames) >= samples_count:
                    break
                if duration is not None and (time.time() - start) >= duration:
                    break
                batch = self.get_all_data()
                if batch:
                    if samples_count is not None:
                        batch = batch[: samples_count - len(frames)]
                    frames.extend(batch)
                    pbar.update(len(batch))
                    last_data = time.time()
                else:
                    if stall_timeout is not None and (time.time() - last_data) >= stall_timeout:
                        print(f"\n[{self.hand_type}] WARNING: no data for "
                              f"{stall_timeout:.0f}s — stopping early (stream stalled "
                              f"or disconnected). Keeping {len(frames)} frames.")
                        break
                    time.sleep(0.001)
        return frames

    def calibrate(self, samples_min_max: int = 1000, samples_avg: int = 1000) -> None:
        """Calibrate the glove (min/max and static average)."""
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        # Min/max calibration
        self.min_val = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        self.max_val = [0] * self.NUM_TENSILE_SENSORS
        print(f"[{self.hand_type}] Calibration Pose 1: Make a fist and open your hand, multiple times. Press Enter to continue...")
        input()
        for i in tqdm(range(samples_min_max), desc=f"[{self.hand_type}] min/max calibration"):
            data = self.get_data(timeout=self.READ_TIMEOUT)
            for j in range(self.NUM_TENSILE_SENSORS):
                v = data.tensile_data[j]
                if v < self.min_val[j]:
                    self.min_val[j] = v
                if v > self.max_val[j]:
                    self.max_val[j] = v
        print()
        # Static average calibration
        print(f"[{self.hand_type}] Calibration Pose 2: Hold your hand static, four fingers forward and thumb out 45°. Press Enter to continue...")
        input()
        sums = [0] * self.NUM_TENSILE_SENSORS
        last = None
        collected = 0
        with tqdm(total=samples_avg, desc=f"[{self.hand_type}] static avg calibration") as pbar:
            while collected < samples_avg:
                data = self.get_data(timeout=self.READ_TIMEOUT)
                current = data.tensile_data
                if last is not None and not self._sensors_still(current, last, threshold=10):
                    continue
                for j in range(self.NUM_TENSILE_SENSORS):
                    sums[j] += current[j]
                last = current
                collected += 1
                pbar.update(1)
        print()
        self.avg_val = np.array([s / samples_avg for s in sums])
        self.is_calibrated = True

    def data_collection(self, samples_count: int = 500, prompt: str = "Please perform the action you want to collect data for.") -> List[GloveSensorData]:
        """Collect data for glove."""
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        # Collect data
        data = []
        print(f"[{self.hand_type}] Please follow the instructions, then press Enter: {prompt}")
        input()
        for _ in tqdm(range(samples_count), desc=f"[{self.hand_type}] data collection"):
            data.append(self.get_data(timeout=self.READ_TIMEOUT))
        return data

    def advanced_calibrate(self, samples_count: int = 200, model: Optional[Any] = None, vis: Optional[HandVisualizer] = None) -> None:
        """Advanced calibration for glove. Optimization for pinch."""
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        print(f"\033[92m[Advanced Calibration] Starting advanced calibration for hand type: {self.hand_type}\033[0m")
        # Collect data
        data_batch = []
        distance_index_set = [[4, 8], [4, 12], [4, 16]]
        prompts = ["Pinch with index and thumb", "Pinch with middle and thumb", "Pinch with ring and thumb"]
        for i in range(len(distance_index_set)):
            data = self.data_collection(samples_count, prompt=prompts[i])
            angles = torch.from_numpy(self.batch_inference(data, method="model", model=model))
            data_batch.append(angles)
        opt_tuning_param = nn.Parameter(torch.ones(self.MODEL_OUTPUT_SIZE, dtype=torch.float32))
        optimizer = torch.optim.Adam([opt_tuning_param], lr=0.01)
        lambda_reg = 0.5
        num_steps = 300  # depending on convergence

        for j in tqdm(range(num_steps)):
            optimizer.zero_grad()
            loss = 0
            for i in range(len(distance_index_set)):
                angles = data_batch[i]
                distance_index = distance_index_set[i]
                joints = vis.get_joints(angles * opt_tuning_param, self.hand_type, backend=torch, dtype=torch.float32, device='cpu')
                dist = torch.norm(joints[:, distance_index[0]] - joints[:, distance_index[1]], dim=1)
                main_loss = dist.mean()
                loss += main_loss
            reg_loss = F.mse_loss(opt_tuning_param, torch.ones_like(opt_tuning_param))
            loss += lambda_reg * reg_loss
            loss.backward()
            optimizer.step()
            # Clamp first 16 entries to 0.9-1.1, rest to 0.5-1.5
            opt_tuning_param.data[:-6].clamp_(0.9, 1.1)
            opt_tuning_param.data[-6:].clamp_(0.8, 1.2)
        self.custom_tuning = opt_tuning_param.data.cpu().numpy()

        print(f"\033[92m[Advanced Calibration] Finished advanced calibration for hand type: {self.hand_type}\033[0m")

    @staticmethod
    def _sensors_still(current, last, threshold=10) -> bool:
        """
        Check if sensor values are relatively stable between two consecutive readings.
        
        Args:
            current: Current sensor readings as a list of integers
            last: Previous sensor readings as a list of integers  
            threshold: Maximum allowed difference between corresponding sensors (default: 10)
            
        Returns:
            bool: True if all sensor differences are below threshold, False otherwise
        """
        return all(abs(c - l) < threshold for c, l in zip(current, last))

    def _is_valid_data(self, data: bytes) -> bool:
        if len(data) != self.PACKET_SIZE:
            return False
        received_crc = struct.unpack('<I', data[-4:])[0]
        computed_crc = zlib.crc32(data[:self.CRC_DATA_SIZE]) & 0xFFFFFFFF
        return received_crc == computed_crc

    def inference(self, data: GloveSensorData, method: str = "model", model: Optional[Any] = None) -> np.ndarray:
        """
        Infer joint angles from sensor data using specified method.
        
        Args:
            data: GloveSensorData object containing tensile sensor readings
            method: Inference method to use ("linear" for linear mapping, "model" for ML model)
            
        Returns:
            np.ndarray: Array of joint angles in radians for each finger joint
        """
        if method == "linear":
            # TODO: add linear mapping implementation here
            raise NotImplementedError
        elif method == "model":
            if model is None:
                raise ValueError("Model is required for model-based inference")
            delta_input = (np.array(data.tensile_data) - np.array(self.avg_val)).astype(np.float32)
            outputs = model.run(None, {'input': delta_input[self.SENSOR_ORDER].reshape(1, -1)})
            return outputs[0][0] * self.custom_tuning
        else:
            raise NotImplementedError
        
    def batch_inference(self, data: List[GloveSensorData], method: str = "model", model: Optional[Any] = None) -> np.ndarray:
        """
        Batch inference for glove.
        """
        if method == "linear":
            raise NotImplementedError
        elif method == "model":
            if model is None:
                raise ValueError("Model is required for model-based inference")
            batch_tensile_data = np.array([d.tensile_data for d in data])
            batch_tensile_data = (batch_tensile_data - self.avg_val).astype(np.float32)
            outputs = model.run(None, {'input': batch_tensile_data[:, self.SENSOR_ORDER]})
            return outputs[0]
        else:
            raise NotImplementedError


def glove_data_to_arrays(data: List[GloveSensorData]) -> dict:
    """
    Pack a sequence of :class:`GloveSensorData` frames into a dict of numpy arrays.

    This is the storage-friendly representation used when persisting recordings: it
    is decoupled from the ``GloveSensorData`` class definition (so saved files do not
    break if the dataclass changes) and is trivially loadable with plain numpy.

    Args:
        data: List of parsed sensor frames (oldest first).

    Returns:
        Dict with keys ``num_frames``, ``tensile`` (N, 19) int32, ``acc`` / ``gyro`` /
        ``mag`` (N, 3) float32, ``temperature`` (N,) float32 and ``timestamp`` (N,)
        uint32. Empty arrays with the correct shape are returned for an empty input.
    """
    n = len(data)
    if n == 0:
        return {
            'num_frames': 0,
            'tensile': np.empty((0, Glove.NUM_TENSILE_SENSORS), dtype=np.int32),
            'acc': np.empty((0, Glove.NUM_IMU_AXES), dtype=np.float32),
            'gyro': np.empty((0, Glove.NUM_IMU_AXES), dtype=np.float32),
            'mag': np.empty((0, Glove.NUM_IMU_AXES), dtype=np.float32),
            'temperature': np.empty((0,), dtype=np.float32),
            'timestamp': np.empty((0,), dtype=np.uint32),
        }
    return {
        'num_frames': n,
        'tensile': np.array([d.tensile_data for d in data], dtype=np.int32),
        'acc': np.array([d.acc_data for d in data], dtype=np.float32),
        'gyro': np.array([d.gyro_data for d in data], dtype=np.float32),
        'mag': np.array([d.mag_data for d in data], dtype=np.float32),
        'temperature': np.array([d.temperature for d in data], dtype=np.float32),
        'timestamp': np.array([d.timestamp for d in data], dtype=np.uint32),
    }


def arrays_to_glove_data(hand_payload: dict) -> List[GloveSensorData]:
    """
    Reconstruct a list of :class:`GloveSensorData` frames from the array dict
    produced by :func:`glove_data_to_arrays` (the inverse operation).

    This lets a saved recording be fed back through the inference pipeline
    (e.g. :meth:`Glove.batch_inference`) for offline replay / visualization.

    Args:
        hand_payload: A per-hand dict from a loaded recording (keys ``num_frames``,
            ``tensile``, ``acc``, ``gyro``, ``mag``, ``temperature``, ``timestamp``).

    Returns:
        List of :class:`GloveSensorData`, oldest first.
    """
    n = int(hand_payload.get('num_frames', 0))
    tensile = hand_payload['tensile']
    acc = hand_payload['acc']
    gyro = hand_payload['gyro']
    mag = hand_payload['mag']
    temperature = hand_payload['temperature']
    timestamp = hand_payload['timestamp']
    frames = []
    for i in range(n):
        frames.append(GloveSensorData(
            tensile_data=tuple(int(v) for v in tensile[i]),
            acc_data=tuple(float(v) for v in acc[i]),
            gyro_data=tuple(float(v) for v in gyro[i]),
            mag_data=tuple(float(v) for v in mag[i]),
            temperature=float(temperature[i]),
            timestamp=int(timestamp[i]),
        ))
    return frames