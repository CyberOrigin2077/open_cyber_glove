# Example Usage

This directory contains example scripts demonstrating how to use the OpenCyberGlove SDK and its integration with the InspireHand robotic hand.

## Example Index

1. [hello_world.py](#1-helloworldpy)  
   Basic usage of OpenCyberGlove SDK: initialization, calibration, diagnosis, and real-time hand pose visualization.

2. [teleop_demo_inspire.py](#2-teleopdemoinspirepy)  
   Teleoperation: map glove sensor data to control an InspireHand robotic hand in real time.

3. [teleop_demo_rohand.py](#3-teleopdemorohandpy)  
   Teleoperation: map glove sensor data to control a ROHand robotic hand in real time.

4. [advanced_calibration.py](#4-advancedcalibrationpy)

5. [data_collection.py](#5-datacollectionpy)
   Record calibrated glove sensor data (frame-complete) and save it to a `.pkl` file for later use (e.g. model training), with a post-recording quality report.

6. [replay_recording.py](#6-replayrecordingpy)
   Replay a recorded `.pkl` offline as a 3D stick-figure hand (same visualizer as `hello_world.py`).

## 1. [`hello_world.py`](#1-helloworldpy) 

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

## 2. [`teleop_demo_inspire.py`](#2-teleopdemoinspirepy)

This script demonstrates teleoperation: mapping glove sensor data to control an InspireHand robotic hand in real time.

### Usage

```bash
python teleop_demo_inspire.py --hand_serial_port <INSPIRE_HAND_PORT> --glove_serial_port <GLOVE_PORT>
```

#### Arguments

- `--hand_serial_port`: Serial port for InspireHand (default: `/dev/ttyUSB1`)
- `--glove_serial_port`: Serial port for OpenCyberGlove (default: `/dev/ttyUSB2`)

#### Example

```bash
python teleop_demo_inspire.py --hand_serial_port /dev/ttyUSB1 --glove_serial_port /dev/ttyUSB2
```

The script will:
- Connect to the InspireHand and OpenCyberGlove devices
- Calibrate the glove
- Continuously read glove sensor data, map it to finger angles, and send commands to the InspireHand for real-time teleoperation

**Note:**  
- Make sure the required dependencies (`open_cyber_glove`, `inspire_hand`) are installed and the serial ports are correctly set for your hardware. Install `inspire_hand` from [inspire_hands](https://github.com/Sentdex/inspire_hands). Due to the implementation inside [inspire_hands](https://github.com/Sentdex/inspire_hands/blob/main/inspire_hand/modbus.py#L129), it can be lagging. For more accurate retargeting, please refer to [open_cyber_glove_retarget_ros2](https://github.com/CyberOrigin2077/open_cyber_glove_retarget_ros2).
- For more details on calibration and setup, refer to the main project documentation.

## 3. [`teleop_demo_rohand.py`](#3-teleopdemorohandpy)

This script demonstrates teleoperation: mapping glove sensor data to control a ROHand from OYMotion in real time.

### Usage

```bash
python teleop_demo_rohand.py --hand_serial_port <ROHAND_PORT> --glove_serial_port <GLOVE_PORT>
```

#### Arguments

- `--hand_serial_port`: Serial port for ROHand (default: `/dev/ttyUSB1`)
- `--glove_serial_port`: Serial port for OpenCyberGlove (default: `/dev/ttyUSB0`)

#### Example

```bash
python teleop_demo_rohand.py --hand_serial_port /dev/ttyUSB1 --glove_serial_port /dev/ttyUSB0
```

The script will:
- Connect to the ROHand and OpenCyberGlove devices
- Calibrate the glove
- Continuously read glove sensor data, map it to finger angles, and send commands to the ROHand for real-time teleoperation

**Note:**  
- Make sure the required dependencies (`open_cyber_glove`, `pymodbus`, `pyserial`) are installed and the serial ports are correctly set for your hardware.
- For more details on calibration and setup, refer to the main project documentation.

## 4. [`advanced_calibration.py`](#4-advancedcalibrationpy)

This script offers an experimental approach to enhance the accuracy of the pinching gesture. If you notice issues with pinching precision, consider running this script to determine whether advanced calibration improves performance. Please follow the interactive instructions provided in the command line.

### Usage

```bash
python advanced_calibration.py --right_port <RIGHT_GLOVE_SERIAL_PORT> --calib_path <CALIBRATION_MODEL_PATH> --model_path <MODEL_PATH>
```

#### Arguments

- `--right_port`: Serial port for the right glove (e.g., `/dev/ttyUSB1`). Optional.
- `--calib_path`: Path to the calibration model (default: `model/hand_model.pkl`).
- `--model_path`: Path to the inference model (default: `model/best.pth`).

#### Example

```bash
python advanced_calibration.py --right_port /dev/ttyUSB0 --calib_path model/hand_model.pkl --model_path model/20250703_110909.onnx
```

## 5. [`data_collection.py`](#5-datacollectionpy)

This script records sensor data from one or two gloves after calibration and saves it to a `.pkl` file. Recording is **frame-complete**: every buffered packet is drained in arrival order (no dropped or duplicated frames), unlike the real-time `get_data()` path which only returns the latest frame.

### Usage

```bash
python data_collection.py --left_port <LEFT_GLOVE_SERIAL_PORT> --right_port <RIGHT_GLOVE_SERIAL_PORT> --output <OUTPUT_PKL> --samples <NUM_FRAMES>
```

After you press Enter to start, recording runs for **10 minutes by default** (or until `--samples` frames are reached, if given). You can stop early at any time with **Ctrl-C** — whatever was collected so far is still saved. If the data stream stalls (e.g. the glove is unplugged) for several seconds, recording also stops automatically and keeps what was collected. The queue is flushed at the moment you press Enter, so the recording starts cleanly from your action (no buffered pre-action frames). When it finishes, a quality report prints per-sensor active ranges, the effective sample rate, and an estimate of dropped frames.

#### Arguments

- `--left_port`: Serial port for the left glove (e.g. `/dev/ttyUSB0`). Optional.
- `--right_port`: Serial port for the right glove (e.g. `/dev/ttyUSB1`). Optional.
- `--output`: Output `.pkl` path (default: `recordings/session.pkl`). Parent directories are created automatically.
- `--duration`: Max recording duration in seconds (default: `600`, i.e. 10 minutes).
- `--samples`: Frames to record per glove. Default `None` = record purely by `--duration`. If both are given, recording stops at whichever limit is reached first.
- `--prompt`: Instruction shown to the operator before recording starts.

#### Example (left glove, default 10-minute recording)

```bash
python data_collection.py --left_port /dev/ttyUSB0 --output recordings/fist.pkl
```

#### Example (right glove, fixed 1000 frames)

```bash
python data_collection.py --right_port /dev/ttyUSB0 --output recordings/fist.pkl --samples 1000
```

### Saved format

The `.pkl` holds a plain dict (numpy arrays + metadata; it does **not** pickle `GloveSensorData` objects, so files stay loadable even if the class changes):

```python
{
  'fps': 120,
  'created_at': "YYYY-MM-DD HH:MM:SS",
  'hands': {
    'left': {                       # and/or 'right'
      'num_frames': N,
      'tensile':     ndarray (N, 19) int32,
      'acc':         ndarray (N, 3)  float32,
      'gyro':        ndarray (N, 3)  float32,
      'mag':         ndarray (N, 3)  float32,
      'temperature': ndarray (N,)    float32,
      'timestamp':   ndarray (N,)    uint32,
      'calibration': {'min_val', 'max_val', 'avg_val', 'custom_tuning', 'is_calibrated', 'sensor_order'},
    },
  },
  'metadata': {...},
}
```

Load a recording with:

```python
from open_cyber_glove.sdk import load_recording
rec = load_recording("recordings/fist.pkl")
tensile = rec['hands']['left']['tensile']   # (N, 19)
```

## 6. [`replay_recording.py`](#6-replayrecordingpy)

This script replays a recorded `.pkl` **offline** as the same 3D stick-figure hand shown by `hello_world.py`. No glove hardware is needed. The recording already stores everything inference requires (raw `tensile` plus the per-hand `avg_val` / `custom_tuning` / `sensor_order` calibration), so the recorded frames are fed back through the ONNX joint-angle model and animated. Both `left` and `right` recordings are supported.

### Usage

```bash
python replay_recording.py --input <RECORDING_PKL> --calib_path <HAND_MODEL> --model_path <ONNX_MODEL> [--fps <HZ>] [--loop]
```

#### Arguments

- `--input`: Path to a recording `.pkl` saved by `data_collection.py` (required).
- `--calib_path`: Path to the hand model for the visualizer (default: `model/hand_model.pkl`).
- `--model_path`: Path to the ONNX joint-angle model (default: `model/best.pth`).
- `--fps`: Playback rate (default: the recording's stored `fps`).
- `--loop`: Loop playback until the window is closed or `Ctrl-C`.

#### Example

```bash
python replay_recording.py --input recordings/fist.pkl --calib_path model/hand_model.pkl --model_path model/20250703_110909.onnx --loop
```

**Note:** Replay quality depends on the same model + calibration as live visualization. You need the model files (download per the main README) and a recording that was made after calibration.
