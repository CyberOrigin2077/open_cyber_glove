# OpenCyberGlove SDK

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/jAujSRAK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white"/></a>
  <a href="https://open-cyber-glove.notion.site/OpenCyberGlove-Intro-21d7a9fbe9288032b0c6c5fab62d21b1" target="_blank"><img alt="Notion" src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"/></a>
    <a href="https://CyberOrigin2077.github.io/open_cyber_glove/" target="_blank"><img alt="Website" src="https://img.shields.io/badge/Google%20Chrome-4285F4?style=for-the-badge&logo=GoogleChrome&logoColor=white"/></a>
   <br>
  <a href="https://github.com/CyberOrigin2077/open_cyber_glove" target="_blank"><img alt="GitHub stars" src="https://img.shields.io/github/stars/CyberOrigin2077/open_cyber_glove?style=social"/></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img alt="License" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"/></a>
</div>


An open-source Python SDK for interfacing with data gloves, supporting real-time sensor data acquisition, calibration, and extensible inference. Designed for robotics, VR/AR, and HCI applications.

## Features
- Connect to one or two data gloves via serial port (1Mbps baudrate)
- Real-time sensor data acquisition at 120Hz
- Comprehensive sensor data including:
  - 19 tensile sensors
  - Temperature sensor
  - Timestamp
- Extensible inference interface for custom models

For more product info, refer to [here](https://open-cyber-glove.notion.site/OpenCyberGlove-Intro-21d7a9fbe9288032b0c6c5fab62d21b1).

## Installation

### From Source
```bash
git clone https://github.com/CyberOrigin2077/open_cyber_glove.git
cd open_cyber_glove
conda create --name ocg python=3.9
conda activate ocg
pip install -e .
```

### Dependencies
The package requires:
- Python 3.7+
- pyserial
- numpy
- matplotlib
- tqdm
- open3d

## Usage

Here are a few examples of how to use the `OpenCyberGlove` SDK.

### Basic Usage: Single Glove

This example shows how to connect to a single glove, calibrate it, and read raw sensor data.

```python
from open_cyber_glove.sdk import OpenCyberGlove

# Initialize SDK for a single left glove.
# Replace '/dev/ttyUSB0' with your glove's serial port.
sdk = OpenCyberGlove(left_port='/dev/ttyUSB0')

# Start the background data reader.
sdk.start()

# Calibrate the glove. This is an interactive process.
print("Starting calibration...")
sdk.calibrate()
print("Calibration finished.")

# Get the latest sensor data packet.
sensor_data = sdk.get_data('left')
print(f"Timestamp: {sensor_data.timestamp}")
print(f"Tensile data: {sensor_data.tensile_data}")
print(f"Accelerometer: {sensor_data.acc_data}")

# Stop the data reader.
sdk.stop()
```

### Real-Time 3D Hand Visualization

The `examples/hello_world.py` script provides a complete demonstration of the SDK's capabilities, including real-time hand pose inference and 3D visualization. It reads data from one or two gloves, feeds it into a pre-trained ONNX model to infer joint angles, and then visualizes the hand's movement in a 3D environment using Open3D.

To run the example, use the following command structure, providing the serial ports for your gloves and paths to the model files. For model files, you need to download from [here](https://drive.google.com/drive/folders/1LW2z2wBDEpXqcMQsoiB_N38c-V5Sq9XT?usp=drive_link) and put them under `model` directory.

**Example command for dual gloves:**
```bash
python3 -m examples.hello_world --right_port ${RIGHT_PORT} --left_port ${LEFT_PORT} --calib_path ${HAND_MODEL} --model_path ${MODEL_PATH}
```

**You can also run it with a single glove:**
```bash
python3 -m examples.hello_world --right_port ${RIGHT_PORT} --calib_path ${HAND_MODEL} --model_path ${MODEL_PATH}
```

The script will first guide you through the interactive calibration process for each connected glove. After calibration, a 3D visualization window will appear, showing the real-time movement of the hand(s).

**Note**: The quality of the visualization **significantly** depends on the model and proper calibration.

## Data Structure
The `GloveSensorData` class provides structured access to all sensor data:
- `tensile_data`: Tuple of 19 integers (0-16384)
- `acc_data`: Tuple of 3 floats (To be added)
- `gyro_data`: Tuple of 3 floats (To be added)
- `mag_data`: Tuple of 3 floats (To be added)
- `temperature`: Float
- `timestamp`: Integer

## License
BSD 3-Clause License