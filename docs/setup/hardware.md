# Hardware Specifications

## Sensor and Material Specifications

### Glove Structure
- **Material**: Premium Lycra blend with conductive thread integration
- **Size Options**: S, M, L, XL
- **Weight**: ~150g per glove
- **Durability**: Designed for 1000+ hours of continuous use

### Sensor Specifications
- **Total Sensors**: 19 flex sensors per glove
- **Sensor Type**: Conductive flex sensors
- **Sensor Placement**:
  - 3 sensors per finger (MCP, PIP, DIP joints)
  - 4 sensors for finger-between
- **Processor**: Data from sensors are read from a controller board mounted on the back of the glove

![sensor location](../imgs/sensor_specs.png)

## Communication
- **Method**: serial port through USB Type-C
- **Baudrate**: 1M
- **Frequency**: Up to 120Hz
- **Data**:
  - 19 flex sensor data
  - 9-axis accelerometer, gyro and magnetometer data
  - temperature
  - timestamp
  - reserved and data validation bits

### Definition
```cpp
typedef struct
{
    /* tensile sensor data */
    int         tensile_data[19];    // corresponding to strech sensor data, ranging from 0-16380
    /* imu sensor data */            // Acc, Gyro, Mag data, currently not used
    float       acc_data[3];
    float       gyro_data[3];
    float       mag_data[3];
    float       temperature;
    /* time stamp, unit: ms */
    uint32_t    timestamp;            // in millisecond (ms)
    /* unused */
    uint8_t     reserve[8];           // reserved, currently not used
    uint32_t    crc32;                // crc32 check, for bytes in frond of timestamp
} hand_data_t;
```

## About flex sensors
<!-- put the image from doc here, make a better graph. (TODO @zimo) -->


## Related Documentation
<!-- - [Firmware Update Guide](docs/setup/firmware.md) (TODO) -->
<!-- - [Maintenance Guide](docs/setup/maintenance.md) (TODO) -->
- [Troubleshooting Guide](docs/setup/troubleshooting.md) (TODO)

## Technical Support
For hardware-related issues, please:
1. Check the [FAQ](../faq.md)
2. Visit our [GitHub Issues](https://github.com/CyberOrigin2077/cyber_glove_ros2_py/issues)
3. Contact our technical support team
