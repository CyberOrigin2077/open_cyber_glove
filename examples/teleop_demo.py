from inspire_hand import InspireHand
from open_cyber_glove.sdk import OpenCyberGlove
from inspire_hand.hand import Register
import time

def threshold_map(val, angle_min=0, angle_max=1000, threshold=0.8):
    if val >= threshold:
        return angle_max
    elif val <= (1 - threshold):
        return angle_min
    else:
        # Linear interpolation to the middle interval
        return int((val - (1 - threshold)) / (2 * threshold - 1) * (angle_max - angle_min) + angle_min)

def map_raw_to_angle(raw_data, min_vals, max_vals, angle_min=0, angle_max=1000, threshold=0.6, coeffs=None):
    sensor_groups = [
        [16, 17],      # Pinky: sensors 17,18
        [12, 13],      # Ring finger: sensors 13,14
        [8, 9],        # Middle finger: sensors 9,10
        [4, 5],        # Index finger: sensors 5,6
        [1, 0],        # Thumb flexion: sensors 2,4
        [3]            # Thumb rotation: sensors 3,4
    ]
    sensor_weights = [
        [0, 1],    # Pinky
        [0, 1],    # Ring finger
        [0, 1],    # Middle finger
        [0, 1],    # Index finger
        [0, 1],    # Thumb flexion
        [1.0]      # Thumb rotation
    ]
    if coeffs is None:
        coeffs = [1.0] * 6
    angles = []
    for i, group in enumerate(sensor_groups):
        vals = []
        weights = sensor_weights[i]
        for j, idx in enumerate(group):
            denom = max_vals[idx] - min_vals[idx]
            if denom == 0:
                val = 0.0
            else:
                val = (raw_data[idx] - min_vals[idx]) / denom
                val = 1 - val
                val = max(0, min(1, val))
            vals.append(val * weights[j])
        sum_val = sum(vals)
        sum_val = sum_val / sum(weights)
        sum_val = max(0, min(1, sum_val * coeffs[i]))
        angle = threshold_map(sum_val, angle_min, angle_max, threshold)
        angles.append(angle)
        print()
    return angles

def main(hand_serial_port="/dev/ttyUSB0", glove_serial_port="/dev/ttyUSB2"):
    hand = InspireHand(port=hand_serial_port, baudrate=115200)
    hand.open()  
    hand.open_all_fingers()
    time.sleep(2)

    glove = OpenCyberGlove(left_port=glove_serial_port)
    glove.start()
    glove.calibrate()

    print("Start teleoperation. Move your glove!")
    while True:
        data = glove.get_data(hand_type='left')
        raw_data = data.tensile_data  
        min_vals = glove.left_glove.min_val
        max_vals = glove.left_glove.max_val
        coeffs = [0.8, 1.0, 1.0, 0.8, 0.8, 0.55]
        angles = map_raw_to_angle(raw_data, min_vals, max_vals, coeffs=coeffs)
        for i, angle in enumerate(angles):
            if not 0 <= angle <= 1000:
                raise ValueError(f"Angle for finger {i} out of range: {angle}")
        hand.modbus.write_multiple_registers(Register.ANGLE_SET, angles)

if __name__ == "__main__":
    main()