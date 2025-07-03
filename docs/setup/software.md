# Software Guide

## OpenCyberGlove in Python

### Installation
```bash
git clone https://github.com/CyberOrigin2077/open_cyber_glove.git
cd open_cyber_glove
pip install -e .
```

### Usage

Refer to [https://github.com/CyberOrigin2077/open_cyber_glove](https://github.com/CyberOrigin2077/open_cyber_glove)

### Note
#### Forward Kinematics

Forward Kinematics (FK) is used to reconstruct the 3D joint positions and orientations of the hand based on given joint angles and calibration data. This functionality is implemented by the `forward_kinematics_from_angles` function. Due to the number of flex sensors in the glove, we implemented a simpified version FK to get the fingertip position.

#####  Basic Principle
Its core principle is **sequential coordinate system transformation**: Starting from the root joint (wrist), the pose of each child joint relative to its parent joint is calculated sequentially along the kinematic chain, until the positions and orientations of all fingertips are determined.

##### Calculation Process
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

## Contributing
We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details.
