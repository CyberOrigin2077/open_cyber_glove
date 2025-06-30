# OpenCyberGlove SDK

An open-source Python SDK for interfacing with data gloves, supporting real-time sensor data acquisition, calibration, and extensible inference. Designed for research, robotics, VR/AR, and HCI applications.

## Features
- Connect to one or two data gloves via serial port (1Mbps baudrate)
- Real-time sensor data acquisition at 120Hz
- Comprehensive sensor data including:
  - 19 tensile sensors
  - 3-axis accelerometer
  - 3-axis gyroscope
  - 3-axis magnetometer
  - Temperature sensor
  - Timestamp
- Automatic CRC validation for data integrity
- Interactive calibration with progress bars:
  - Min/max calibration for dynamic range
  - Static average calibration for baseline
- Real-time data visualization with matplotlib
- Threaded data acquisition with queue-based buffering
- Extensible inference interface for custom models

## Installation

### From Source
```bash
git clone https://github.com/CyberOrigin2077/open_cyber_glove.git
cd open_cyber_glove
pip install -r requirements.txt
```
For developers, you can install the package in editable mode, which helps in applying changes to the code without reinstalling.
```bash
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

### Hand Pose Inference

This example shows how to load a pre-trained model and perform inference to get hand joint angles.

```python
from open_cyber_glove.sdk import OpenCyberGlove

# Initialize SDK with port and path to the ONNX model.
sdk = OpenCyberGlove(left_port='/dev/ttyUSB0', model_path='model/20250616_164927_haichen_ensemble.onnx')

sdk.start()
sdk.calibrate()

# Get inferred joint angles.
joint_angles = sdk.get_angles('left', method='model')
print("Inferred joint angles (radians):", joint_angles)

sdk.stop()
```
**Note**: The quality of the inference heavily depends on the model and proper calibration.

### Dual Glove & Live Visualization

This example demonstrates managing two gloves simultaneously and visualizing their raw sensor data in real-time.

```python
from open_cyber_glove.sdk import OpenCyberGlove

# Initialize with ports for both left and right gloves.
# Replace with your actual serial ports.
sdk = OpenCyberGlove(left_port='/dev/ttyUSB0', right_port='/dev/ttyUSB1')

# Start data acquisition for both gloves.
sdk.start()

# Calibrate both gloves sequentially.
sdk.calibrate()

# Run real-time diagnosis with visualization.
# This will open a plot showing live sensor data.
# Close the plot window to stop.
sdk.diagnose()

# Stop all gloves.
sdk.stop()
```

## Calibration
The calibration process is interactive and consists of two phases:

1. **Min/Max Calibration**
   - Make a fist and open your hand repeatedly
   - Progress bar shows completion status
   - Captures dynamic range for each sensor

2. **Static Average Calibration**
   - Hold your hand static in a specific pose
   - Four fingers forward, thumb out 45°
   - Progress bar shows completion status
   - Calculates baseline values for each sensor

## Data Structure
The `GloveSensorData` class provides structured access to all sensor data:
- `tensile_data`: Tuple of 19 integers (0-16384)
- `acc_data`: Tuple of 3 floats (accelerometer)
- `gyro_data`: Tuple of 3 floats (gyroscope)
- `mag_data`: Tuple of 3 floats (magnetometer)
- `temperature`: Float
- `timestamp`: Integer

## License
BSD 3-Clause License

Copyright (c) 2024, OpenCyberGlove Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.