from open_cyber_glove.sdk import OpenCyberGlove
from open_cyber_glove.visualizer import HandVisualizer
import argparse
import time
import numpy as np

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
    sdk.diagnose()

    visualizer = HandVisualizer(model_path=args.calib_path)    
    
    # Real-time update loop
    try:
        print("Starting real-time update loop...")
        while True:
            # Update poses with timeout protection
            if sdk.left_glove is not None:
                angles = sdk.get_angles(hand_type='left', method='model')                
                visualizer.update(angles, hand_type='left')
            if sdk.right_glove is not None:
                angles = sdk.get_angles(hand_type='right', method='model')                
                visualizer.update(angles, hand_type='right')
            
            time.sleep(1/120)  # 120 Hz

    except KeyboardInterrupt:
        print("Stopping visualization...")
    finally:
        sdk.stop()
        visualizer.close()
