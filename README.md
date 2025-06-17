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
pip install -e .
```

### Dependencies
The package requires:
- Python 3.7+
- pyserial
- numpy
- matplotlib
- tqdm

## Usage

### Single Glove Example
```python
from open_cyber_glove.glove import Glove

class MyGlove(Glove):
    def inference(self, data):
        # Implement your inference logic here
        return None

# Initialize and connect
glove = MyGlove('left')
glove.connect('/dev/ttyUSB0')

# Start background data acquisition
glove.start_reader()

# Calibrate the glove
glove.calibrate()  # Follow on-screen instructions

# Get and parse data
raw = glove.get_raw_data()
parsed = glove.parse_raw_data(raw)
print(parsed.tensile_data)  # Access specific sensor data
print(parsed.acc_data)      # Accelerometer data
print(parsed.gyro_data)     # Gyroscope data
print(parsed.mag_data)      # Magnetometer data
print(parsed.temperature)   # Temperature
print(parsed.timestamp)     # Timestamp

# Stop data acquisition
glove.stop_reader()
```

### Dual Glove Example
```python
from open_cyber_glove.sdk import OpenCyberGlove

# Initialize with one or both ports
sdk = OpenCyberGlove('/dev/ttyUSB0', '/dev/ttyUSB1')

# Start data acquisition for all gloves
sdk.start()

# Calibrate all gloves
sdk.calibrate()

# Run real-time diagnosis with visualization
sdk.diagnose()  # Interactive plot showing sensor data

# Stop all gloves
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

## Error Handling
The SDK includes comprehensive error handling:
- CRC validation for data integrity
- Queue-based buffering for data acquisition
- Proper resource cleanup
- Logging for debugging and monitoring

## License
MIT 