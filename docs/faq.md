# Frequently Asked Questions

## General Questions

### What is OpenCyberGlove?
OpenCyberGlove is an open-source data glove designed for high-fidelity finger tracking. It features 19 stretch sensors and can operate at frequencies up to 120Hz, making it suitable for research, development, and applications in embodied AI.

### How accurate is the finger tracking?
The accuracy depends on proper calibration and sensor placement. With optimal setup, the system can achieve sub-degree accuracy for finger joint angles.

### What operating systems are supported?
The software is primarily developed and tested on Linux (Ubuntu 20.04/22.04), but it should work on any system that supports Python 3.8+ and ROS2.

## Hardware Questions

### What sensors are used?
The glove uses 19 stretch sensors strategically placed to track finger and hand movements. Each sensor is carefully calibrated for optimal performance.

### How long does the battery last?
The battery life depends on the sampling rate and usage patterns. Typically, you can expect 4-6 hours of continuous operation at 120Hz.

### Is the glove waterproof?
No, the current version is not waterproof. Care should be taken to avoid exposure to water or moisture.

## Software Questions

### What programming languages are supported?
The core software is written in Python, with ROS2 integration. The API is designed to be language-agnostic, allowing integration with various programming languages.

### How do I calibrate the glove?
Calibration is done through our calibration software, which guides you through the process. The system automatically detects when sensors are stable to ensure accurate calibration data collection.

### Can I use multiple gloves simultaneously?
Yes, the system supports multiple gloves through our multi-threaded processing architecture. Each glove operates independently to ensure optimal performance.

## Development Questions

### How can I contribute to the project?
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.

### Is there a test suite?
Yes, we maintain a comprehensive test suite. New features should include appropriate tests to ensure reliability.

### How do I report bugs?
Please use our [GitHub Issues](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/issues) page to report bugs. Include as much detail as possible about the issue, including steps to reproduce and expected behavior.

## Commercial Use

### Can I use OpenCyberGlove for commercial purposes?
Yes, the project is licensed under the BSD 3-Clause License, which allows for commercial use. However, we ask that you acknowledge the project in your work.

### Do you offer commercial support?
Currently, we do not offer commercial support. However, we maintain an active community and documentation to help users.

## Community

### Where can I get help?
- GitHub Issues: For bug reports and feature requests
- Documentation: For detailed guides and tutorials
- Community Forums: For general discussion and help
- Discord: For real-time community support

### How can I stay updated?
- Star and watch our GitHub repository
- Join our Discord community
- Follow our social media channels
