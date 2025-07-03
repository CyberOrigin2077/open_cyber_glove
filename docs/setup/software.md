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

##### Hand Model and Coordinate System
Our hand model defines 21 degrees of freedom:
- **Thumb**: CMC (2 DOF), MCP (2 DOF), IP (1 DOF)
- **Other Four Fingers**: MCP (2 DOF), PIP (1 DOF), DIP (1 DOF) each

Joint coordinate systems: 
- Wrist and four MCP joints maintain fixed relative positions. Calibration establishes these systems, with left/right hands handled by mirroring abduction/adduction angles.
- X-axis for flexion/extension, Z-axis for abduction/adduction, Y-axis points to child joint. 
- The FK process can be customized with user-defined coordinate decomposition for specialized applications.

#### Joint Angle Estimation Model

The model provided is trained by data collected from many people (with medium size hands), mainly focusing on generalizability. We plan to release the training pipeline later for our users to implement their own model.

## More Examples

Release soon

## Contributing
We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details.
