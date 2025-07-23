from open_cyber_glove.sdk import OpenCyberGlove
import argparse
from open_cyber_glove.visualizer import HandVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--right_port', type=str, default=None)
    parser.add_argument('--calib_path', type=str, default='model/hand_model.pkl')
    parser.add_argument('--model_path', type=str, default='model/best.pth')
    args = parser.parse_args()

    sdk = OpenCyberGlove(right_port=args.right_port,
                         model_path=args.model_path)
    
    hand_visualizer = HandVisualizer(args.calib_path)

    # Check if SDK is properly initialized
    sdk.start()
    sdk.calibrate()
    sdk.experimental_advanced_calibrate(visualizer=hand_visualizer)

    while True:
        if sdk.right_glove is not None:
            angles = sdk.get_angles(hand_type='right', method='model')                
            hand_visualizer.update(angles, hand_type='right')


