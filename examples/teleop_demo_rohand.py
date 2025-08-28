import asyncio
import signal
import argparse
from typing import List, Optional
from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
from serial.tools import list_ports
from open_cyber_glove.sdk import OpenCyberGlove

# modbus registers for ROHand
ROH_FINGER_SPEED0 = (1125) # Borrow from https://github.com/oymotion/roh_demos/blob/main/glove_ctrled_rohand/roh_registers_v1.py
ROH_FINGER_POS_TARGET0 = (1135)

NODE_ID = 2
NUM_FINGERS = 6

# Threshold for detecting significant target position change (integer for position control, float for angle control)
TOLERANCE = round(65536 / 32)
# If position change is below this value, linearly adjust finger movement speed
SPEED_CONTROL_THRESHOLD = 8192


def clamp(n: float, smallest: float, largest: float) -> float:
    """Clamp a value between smallest and largest."""
    return max(smallest, min(n, largest))


def interpolate(n: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    """Linearly interpolate n from [from_min, from_max] to [to_min, to_max]."""
    if from_max == from_min:
        raise ValueError("from_max and from_min cannot be equal for interpolation.")
    return (n - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


class OpenCyberGloveRoHandDemo:
    def __init__(
        self,
        glove_port: str = "/dev/ttyUSB0",
        hand_device: str = "/dev/ttyUSB1",
    ):
        signal.signal(signal.SIGINT, lambda sig, frame: self._signal_handler())
        self.terminated = False
        print(f"Connecting to glove on port {glove_port}")
        self.hand_device = hand_device
        self.glove = OpenCyberGlove(right_port=glove_port)
        self.glove.start()
        self.glove.calibrate()

    def _signal_handler(self):
        print("Received SIGINT (Ctrl-C), exiting...")
        self.terminated = True

    @staticmethod
    def find_comport(port_name: str) -> Optional[str]:
        """
        Automatically find an available serial port by description.

        Args:
            port_name: Substring to match in the port description (e.g., "CH340").

        Returns:
            The device path of the matching port, or None if not found.
        """
        ports = list_ports.comports()
        for port in ports:
            if port_name in port.description:
                return port.device
        return None

    def map_raw_to_angle(
        self,
        raw_data: List[float],
        min_vals: List[float],
        max_vals: List[float],
        coeffs: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Map raw glove sensor data to target angles for each finger.

        Args:
            raw_data: Raw sensor data.
            min_vals: Minimum calibration values for each sensor.
            max_vals: Maximum calibration values for each sensor.
            coeffs: Optional scaling coefficients for each finger.

        Returns:
            List of target angles (as integers) for each finger.
        """
        # Sensor groupings for each finger and thumb rotation
        sensor_groups = [
            [1, 0],        # Thumb flexion: sensors 2,4
            [4, 5],        # Index finger: sensors 5,6
            [8, 9],        # Middle finger: sensors 9,10
            [12, 13],      # Ring finger: sensors 13,14
            [16, 17],      # Pinky: sensors 17,18
            [3],           # Thumb rotation: sensor 3
        ]

        # Weights for each sensor in the group
        sensor_weights = [
            [0, 1],    # Thumb flexion
            [0, 1],    # Index finger
            [0, 1],    # Middle finger
            [0, 1],    # Ring finger
            [0, 1],    # Pinky
            [1.0],     # Thumb rotation
        ]

        if coeffs is None:
            coeffs = [1.0] * 6

        angles = []
        for i, group in enumerate(sensor_groups):
            vals = []
            weights = sensor_weights[i]
            for j, idx in enumerate(group):
                denom = max_vals[idx] - min_vals[idx]
                if denom == 0:
                    val = 0.0
                else:
                    val = (raw_data[idx] - min_vals[idx]) / denom
                    val = 1 - val  # The more flexed, the higher the value
                    val = clamp(val, 0, 1)
                vals.append(val * weights[j])

            sum_weights = sum(weights)
            sum_val = sum(vals) / sum_weights if sum_weights != 0 else 0.0
            sum_val = clamp(sum_val * coeffs[i], 0, 1)
            angles.append(int(65535 - sum_val * 65535))
        return angles

    @staticmethod
    def write_registers(client: ModbusSerialClient, address: int, values: List[int]) -> bool:
        """
        Write data to a Modbus device.

        Args:
            client: Modbus client instance.
            address: Register address.
            values: Data to be written.

        Returns:
            True if successful, False otherwise.
        """
        try:
            resp = client.write_registers(address, values, NODE_ID)
            if resp.isError():
                print("client.write_registers() returned", resp)
                return False
            return True
        except ModbusException as e:
            print(f"ModbusException: {e}")
            return False

    @staticmethod
    def read_registers(client: ModbusSerialClient, address: int, count: int) -> Optional[List[int]]:
        """
        Read data from a Modbus device.

        Args:
            client: Modbus client instance.
            address: Register address.
            count: Number of registers to read.

        Returns:
            List of register values if successful, None otherwise.
        """
        try:
            resp = client.read_holding_registers(address, count, NODE_ID)
            if resp.isError():
                return None
            return resp.registers
        except ModbusException as e:
            print(f"ModbusException: {e}")
            return None

    async def main(self):
        """
        Main loop for reading glove data and controlling the robotic hand.
        """
        client = ModbusSerialClient(
            self.hand_device,
            framer=FramerType.RTU,
            baudrate=115200,
        )
        if not client.connect():
            print("Failed to connect to Modbus device")
            raise ConnectionError("Could not connect to Modbus device")

        try:
            while not self.terminated:
                glove_data = self.glove.get_data("right")
                coeffs = [1, 1, 1, 1, 1, 1]
                pos = self.map_raw_to_angle(
                    glove_data.tensile_data,
                    self.glove.right_glove.min_val,
                    self.glove.right_glove.max_val,
                    coeffs=coeffs,
                )

                # Optionally, read current position and adjust speed dynamically
                # resp = self.read_registers(client, ROH_FINGER_POS0, NUM_FINGERS)
                # curr_pos = resp if resp is not None else [0] * NUM_FINGERS
                # speed = [
                #     clamp(
                #         round(interpolate(abs(curr_pos[i] - pos[i]), 0, SPEED_CONTROL_THRESHOLD, 0, 65535)),
                #         0,
                #         65535,
                #     )
                #     for i in range(NUM_FINGERS)
                # ]

                # Set speed to maximum for all fingers
                if not self.write_registers(client, ROH_FINGER_SPEED0, [65535] * NUM_FINGERS):
                    print("Failed to set speed")

                # Set target positions for the robotic hand
                if not self.write_registers(client, ROH_FINGER_POS_TARGET0, pos):
                    print("Failed to set position")

        finally:
            self.glove.stop()
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Control ROHand with OpenCyberGlove")
    parser.add_argument(
        "--glove_serial_port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for the OpenCyberGlove (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--hand_serial_port",
        type=str,
        default="/dev/ttyUSB1",
        help="Serial port for the ROHand device (default: /dev/ttyUSB1)",
    )
    args = parser.parse_args()
    print("Starting OpenCyberGlove with ROHand demo...")
    app = OpenCyberGloveRoHandDemo(glove_port=args.glove_serial_port, hand_device=args.hand_serial_port)
    print("Initialization complete")
    asyncio.run(app.main())


if __name__ == "__main__":
    main()