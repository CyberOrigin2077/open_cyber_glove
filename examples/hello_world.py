from open_cyber_glove import OpenCyberGlove
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_port', type=str, default=None)
    parser.add_argument('--right_port', type=str, default=None)
    args = parser.parse_args()

    sdk = OpenCyberGlove(left_port=args.left_port, right_port=args.right_port)
    sdk.start()
    sdk.calibrate()
    sdk.diagnose()
    sdk.stop()