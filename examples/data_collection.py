from open_cyber_glove.sdk import OpenCyberGlove, load_recording, recording_quality_report
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record calibrated glove sensor data and save it to a .pkl file."
    )
    parser.add_argument('--left_port', type=str, default=None,
                        help="Serial port for the left glove (Linux: /dev/ttyUSB0, "
                             "macOS: /dev/cu.usbserial-XXX). Optional.")
    parser.add_argument('--right_port', type=str, default=None,
                        help="Serial port for the right glove (Linux: /dev/ttyUSB1, "
                             "macOS: /dev/cu.usbserial-XXX). Optional.")
    parser.add_argument('--output', type=str, default='recordings/session.pkl',
                        help="Output .pkl path (default: recordings/session.pkl).")
    parser.add_argument('--duration', type=float, default=600.0,
                        help="Max recording duration in seconds (default: 600 = 10 min). "
                             "Recording can be stopped early with Ctrl-C.")
    parser.add_argument('--samples', type=int, default=None,
                        help="Frames to record per glove. Default None = record by --duration.")
    parser.add_argument('--prompt', type=str,
                        default="Perform the action you want to record.",
                        help="Instruction shown before recording starts.")
    args = parser.parse_args()

    sdk = OpenCyberGlove(left_port=args.left_port, right_port=args.right_port)
    sdk.start()
    sdk.calibrate()

    try:
        path = sdk.collect_and_save(
            args.output,
            samples_count=args.samples,
            duration=args.duration,
            prompt=args.prompt,
        )
        print(f"Saved recording to {path}")

        # Quality check of what was written (sensor ranges + dropped-frame estimate).
        recording_quality_report(load_recording(path))
    finally:
        sdk.stop()
