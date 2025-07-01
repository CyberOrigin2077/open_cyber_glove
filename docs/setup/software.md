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

## Forward Kinematics

Forward Kinematics (FK) is used to reconstruct the 3D joint positions and orientations of the hand based on given joint angles and calibration data. This functionality is implemented by the `forward_kinematics_from_angles` function.

### Basic Principle
Its core principle is **sequential coordinate system transformation**: Starting from the root joint (wrist), the pose of each child joint relative to its parent joint is calculated sequentially along the kinematic chain, until the positions and orientations of all fingertips are determined.

### Function Details

#### Input Parameters

- `gt_angles` (`dict`): A dictionary containing the target rotation angles (flexion, abduction) for each joint, typically as increments relative to the calibration pose.
- `calib_dict` (`dict`): The kinematics calibration dictionary, which defines the static geometric information of the hand model. Key fields include:
    - `joint_pos`: 3D coordinates of the joints in the calibration pose.
    - `all_coordinates`: The local coordinate system of each joint in the calibration pose (`4x4` matrix).
    - `link_lengths`: The bone lengths between adjacent joints.
    - `kinematic_tree`: The kinematic chain that defines the parent-child relationships between joints.

#### Return Values

- `fk_joints` (`numpy.ndarray`): The reconstructed 3D coordinates of the 21 hand joints, with a shape of `(21, 3)`.
- `fk_rot` (`numpy.ndarray`): The local coordinate system (rotation matrix) of each joint after rotation, defining its final orientation, with a shape of `(21, 3, 3)`.

### Calculation Process
The following is the calculation process for forward kinematics (taking a single finger, such as the index finger, as an example):

1.  **Parent-Child Joint Determination**: The calculation proceeds iteratively from the root joint (wrist) along the kinematic chain. At each level, the pose of the currently calculated joint (child joint) is derived from the pose of its preceding joint (parent joint).

2.  **Local Coordinate System Transformation (Pose Calculation)**:
    - First, obtain the parent joint's pose (`prev_pos`, `prev_cs`) and the target rotation angles (flexion, abduction) of the current joint.
    - Next, generate the rotation matrices `R_flex` and `R_abd` accordingly and apply them to the parent joint's local coordinate system to calculate the new orientation of the child joint, `rotated_cs`:
      ```python
      rotated_cs = R_abd @ R_flex @ prev_cs[:3, :3]
      ```

3.  **Child Joint Position Calculation**:
    - The position of the child joint, `curr_pos`, is the sum of the parent joint's position and the bone vector.
    - This vector is determined by the bone length, `link_length`, and its direction in the new coordinate system (the Y-axis unit vector, `rotated_cs[:, 1]`):
      ```python
      curr_pos = prev_pos + link_length * rotated_cs[:, 1]
      ```

4.  **State Update and Propagation**:
    The calculated pose of the child joint (`curr_pos`, `rotated_cs`) serves as the parent joint's information for the next level of calculation, propagating along the kinematic chain to the fingertip.

## Getting Help
- [API Documentation](docs/api.md) (TODO)
- [Example Gallery](../examples.md)
- [Community Forum](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/discussions)
- [GitHub Issues](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/issues)

## Contributing
We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details.
