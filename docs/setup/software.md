# Software Guide

## Setup in Python

### Installation
```bash
git clone https://github.com/CyberOrigin2077/open_cyber_glove.git
cd open_cyber_glove
pip install -e .
```

### Basic Usage
#### Connect
Find your device address by `ls /dev/ttyUSB*`, and then run following lines.
```python
from open_cyber_glove import OpenCyberGlove
left_hand_port = "/dev/ttyUSB0"
gloves = OpenCyberGlove(left_port=left_hand_port)
gloves.start() # start getting data
```

#### Calibration
Before using the gloves, there are two calibration procedures for each glove. The prompt pops up during calibration.

- Calibration A: Move your fingers between making a fist and wide openning multiple times to collect min/max possible sensor values.
- Calibration B: Hold your hand flat and static, four fingers forward and thumb out 45°.
<!-- TODO: add two calibration pose images -->
```python
gloves.calibrate()
```

#### Data Access
<!-- TODO: add data getter and parser script -->
```python
data = gloves.get(hand="left") # use hand="right" to get from the other hand
```

#### Diagnose

The diagnostic prodecure can be run as following, to visualize the sensor values in live.
```python
gloves.diagnose()
```
<!-- TODO: add an image here to show the GUI -->

#### Joint Angle Inference

Two methods are provided for joint angle inferencing:
1. Mapping sensor value to joint angle linearly, with human hand angles as prior
2. A small deep learning model to do the mapping

```python
joint_angles = gloves.inference(data, method="linear") # use method="model" to switch
```

### Data Visualization
A base hand skeleton model is provided for converting joint angles to joint locations. When using visualization, the joints are transformed from the base model. 
<!-- TODO: update visualizer in open cyber glove SDK -->
```python
from opencyberglove import GloveVisualizer

# Initialize visualizer
visualizer = GloveVisualizer()

# Start real-time visualization
visualizer.start(glove)
```

### API Reference

#### GloveController
- `connect(port=None)`: Connect to glove
- `disconnect()`: Disconnect from glove
- `start_streaming()`: Start data streaming
- `stop_streaming()`: Stop data streaming
- `get_sensor_data()`: Get raw sensor data
- `get_joint_angles()`: Get processed joint angles
- `calibrate()`: Start calibration process
- `save_calibration(filepath)`: Save calibration data
- `load_calibration(filepath)`: Load calibration data

#### GloveVisualizer
- `start(glove_controller)`: Start visualization
- `stop()`: Stop visualization
- `set_mode(mode)`: Set visualization mode
- `save_recording(filepath)`: Save visualization recording

## Example Applications
For detailed examples and use cases, please see our [Examples Guide](../examples.md). The guide includes:
- Teleoperation examples
- Data collection tutorials
- Machine learning integration
- Visualization examples
- Integration with other platforms (ROS2, Unity, etc.)

## FK

## Getting Help
- [API Documentation](docs/api.md) (TODO)
- [Example Gallery](../examples.md)
- [Community Forum](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/discussions)
- [GitHub Issues](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/issues)

## Contributing
We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details.
