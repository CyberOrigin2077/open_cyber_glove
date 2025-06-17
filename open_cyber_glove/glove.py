import serial
import struct
import zlib
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import threading
import queue
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GloveSensorData:
    tensile_data: Tuple[int, ...]
    acc_data: Tuple[float, ...]
    gyro_data: Tuple[float, ...]
    mag_data: Tuple[float, ...]
    temperature: float
    timestamp: int

class Glove(ABC):
    """
    SDK class for a single data glove device (left or right).
    Handles connection, data reading, parsing, and calibration.
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
    NUM_IMU_AXES = 3
    SENSOR_MAX_VALUE = 8192 * 2
    DEFAULT_BAUDRATE = 1000000
    SENSOR_ORDER = [3, 1, 0, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 2, 7, 11, 15]

    def __init__(self, hand_type: str):
        """
        Args:
            hand: 'left' or 'right'
        """
        self.hand_type = hand_type
        self.serial_port: Optional[serial.Serial] = None
        self.min_val = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        self.max_val = [0] * self.NUM_TENSILE_SENSORS
        self.avg_val = [0.0] * self.NUM_TENSILE_SENSORS
        self.is_calibrated = False
        self._data_queue = queue.Queue(maxsize=1200)  # 10 seconds of data at 120 Hz
        self._reader_thread = None
        self._reader_running = threading.Event()
        self._queue_lock = threading.Lock()

    def connect(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        """Connect to the glove via serial port."""
        self.serial_port = serial.Serial(port, baudrate, timeout=1)

    def start_reader(self):
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._reader_running.set()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop_reader(self):
        self._reader_running.clear()
        if self._reader_thread is not None:
            self._reader_thread.join()
            self._reader_thread = None

    def _reader_loop(self):
        while self._reader_running.is_set():
            try:
                if self.serial_port and self.serial_port.in_waiting >= self.PACKET_SIZE:
                    data = self.serial_port.read(self.PACKET_SIZE)
                    if self._is_valid_data(data):
                        with self._queue_lock:
                            if self._data_queue.full():
                                self._data_queue.get_nowait()
                            self._data_queue.put_nowait(data)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in reader loop: {e}")
                time.sleep(0.01)

    def get_raw_data(self) -> bytes:
        """Get the most recent raw data packet from the queue."""
        if self.serial_port is None:
            raise RuntimeError("Serial port not connected.")
        # Wait for at least one data packet
        while self._data_queue.empty():
            time.sleep(0.001)
        # Drain all but the last
        with self._queue_lock:
            last = None
            while not self._data_queue.empty():
                last = self._data_queue.get()
        return last

    def parse_raw_data(self, raw: bytes) -> GloveSensorData:
        """Parse a raw data packet into structured sensor data."""
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
            raw = self.get_raw_data()
            data = self.parse_raw_data(raw)
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
                raw = self.get_raw_data()
                data = self.parse_raw_data(raw)
                current = data.tensile_data
                if last is not None and not self._sensors_still(current, last, threshold=10):
                    continue
                for j in range(self.NUM_TENSILE_SENSORS):
                    sums[j] += current[j]
                last = current
                collected += 1
                pbar.update(1)
        print()
        self.avg_val = [s / samples_avg for s in sums]
        self.is_calibrated = True

    @staticmethod
    def _sensors_still(current, last, threshold=10) -> bool:
        return all(abs(c - l) < threshold for c, l in zip(current, last))

    def _is_valid_data(self, data: bytes) -> bool:
        if len(data) != self.PACKET_SIZE:
            return False
        received_crc = struct.unpack('<I', data[-4:])[0]
        computed_crc = zlib.crc32(data[:self.CRC_DATA_SIZE]) & 0xFFFFFFFF
        return received_crc == computed_crc

    # @abstractmethod
    # def inference(self, data: GloveSensorData) -> np.ndarray:
    #     """Abstract method for inference. To be implemented by subclasses."""
    #     pass 