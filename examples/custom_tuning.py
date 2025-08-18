from open_cyber_glove.sdk import OpenCyberGlove
import argparse
from open_cyber_glove.visualizer import HandVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_port', type=str, default=None)
    parser.add_argument('--right_port', type=str, default=None)
    parser.add_argument('--calib_path', type=str, default='model/hand_model.pkl')
    parser.add_argument('--model_path', type=str, default='model/best.pth')
    args = parser.parse_args()

    sdk = OpenCyberGlove(left_port=args.left_port, 
                         right_port=args.right_port,
                         model_path=args.model_path)
    
    # Check if SDK is properly initialized
    sdk.start()
    sdk.calibrate()
    visualizer = HandVisualizer(args.calib_path)
if sdk.right_glove:
    visualizer.enable_tuning_panel(sdk.right_glove, sdk.model, hand_type='right')
else:
    print("Right glove not connected. Tuning panel will not be shown.")

