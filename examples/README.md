# Example Usage

This directory contains example scripts demonstrating how to use the OpenCyberGlove SDK and its integration with the InspireHand robotic hand.

## Example Index

1. [hello_world.py](#1-helloworldpy)  
   Basic usage of OpenCyberGlove SDK: initialization, calibration, diagnosis, and real-time hand pose visualization.

2. [teleop_demo.py](#2-teleopdemopy)  
   Teleoperation: map glove sensor data to control an InspireHand robotic hand in real time.


## 1. `hello_world.py`

This script demonstrates basic usage of the OpenCyberGlove SDK, including initialization, calibration, diagnosis, and real-time hand pose visualization.

### Usage

```bash
python hello_world.py --left_port <LEFT_GLOVE_SERIAL_PORT> --right_port <RIGHT_GLOVE_SERIAL_PORT> --calib_path <CALIBRATION_MODEL_PATH> --model_path <MODEL_PATH>
```

#### Arguments

- `--left_port`: Serial port for the left glove (e.g., `/dev/ttyUSB0`). Optional.
- `--right_port`: Serial port for the right glove (e.g., `/dev/ttyUSB1`). Optional.
- `--calib_path`: Path to the calibration model (default: `model/hand_model.pkl`).
- `--model_path`: Path to the inference model (default: `model/best.pth`).

#### Example

```bash
python hello_world.py --left_port /dev/ttyUSB0 --model_path model/20250703_110909.onnx
```

The script will:
- Initialize the glove(s)
- Calibrate and diagnose the device(s)
- Start a real-time visualization of hand poses at 120 Hz
- Stop gracefully on `Ctrl+C`

---

## 2. `teleop_demo.py`

This script demonstrates teleoperation: mapping glove sensor data to control an InspireHand robotic hand in real time.

### Usage

```bash
python teleop_demo.py --hand_serial_port <INSPIRE_HAND_PORT> --glove_serial_port <GLOVE_PORT>
```

#### Arguments

- `--hand_serial_port`: Serial port for InspireHand (default: `/dev/ttyUSB1`)
- `--glove_serial_port`: Serial port for OpenCyberGlove (default: `/dev/ttyUSB2`)

#### Example

```bash
python teleop_demo.py --hand_serial_port /dev/ttyUSB1 --glove_serial_port /dev/ttyUSB2
```

The script will:
- Connect to the InspireHand and OpenCyberGlove devices
- Calibrate the glove
- Continuously read glove sensor data, map it to finger angles, and send commands to the InspireHand for real-time teleoperation

**Note:**  
- Make sure the required dependencies (`open_cyber_glove`, `inspire_hand`) are installed and the serial ports are correctly set for your hardware. Install `inspire_hand` from [inspire_hands](https://github.com/Sentdex/inspire_hands). Due to the implementation inside [inspire_hands](https://github.com/Sentdex/inspire_hands/blob/main/inspire_hand/modbus.py#L129), it can be lagging. For more accurate retargeting, please refer to [open_cyber_glove_retarget_ros2](https://github.com/CyberOrigin2077/open_cyber_glove_retarget_ros2).
- For more details on calibration and setup, refer to the main project documentation.
