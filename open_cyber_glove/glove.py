import serial
import struct
import zlib
import time
import numpy as np
from typing import Optional, Any, Tuple
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
    NUM_IMU_AXES = 3
    SENSOR_MAX_VALUE = 8192 * 2
    DEFAULT_BAUDRATE = 1000000
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

    def get_raw_data(self) -> bytes:
        """
        Retrieve the most recent raw data packet from the queue.
        
        Returns:
            The most recent complete data packet as bytes
            
        Raises:
            RuntimeError: If the serial port is not connected
            
        Note:
            This method blocks until at least one data packet is available.
            It drains all packets from the queue and returns only the most recent one.
        """
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
        
    def get_data(self) -> GloveSensorData:
        """Get the most recent parsed sensor data from the glove."""
        raw_data = self.get_raw_data()
        return self.parse_raw_data(raw_data)

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
            data = self.get_data()
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
                data = self.get_data()
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
            delta_input = (data.tensile_data - self.avg_val).astype(np.float32)
            outputs = model.run(None, {'input': delta_input[self.SENSOR_ORDER].reshape(1, -1)})
            return outputs[0][0]
        else:
            raise NotImplementedError