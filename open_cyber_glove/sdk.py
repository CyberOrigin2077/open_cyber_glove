import threading
from typing import Optional
from .glove import Glove
import matplotlib.pyplot as plt
import numpy as np
import time

class OpenCyberGlove:
    """
    SDK class to manage one or two gloves (left and/or right) in parallel.
    """
    def __init__(self, left_port: Optional[str] = None, right_port: Optional[str] = None, glove_cls=Glove):
        if not left_port and not right_port:
            raise ValueError("At least one of left_port or right_port must be provided.")
        self.left_glove: Optional[Glove] = glove_cls('left') if left_port else None
        self.right_glove: Optional[Glove] = glove_cls('right') if right_port else None
        self.left_port = left_port
        self.right_port = right_port
        self._running = False

    def start(self) -> None:
        """Start available gloves' background data readers."""
        if self.left_glove and self.left_port:
            self.left_glove.connect(self.left_port)
            self.left_glove.start_reader()
        if self.right_glove and self.right_port:
            self.right_glove.connect(self.right_port)
            self.right_glove.start_reader()
        self._running = True

    def stop(self) -> None:
        """Stop all running gloves' background data readers."""
        self._running = False
        if self.left_glove:
            self.left_glove.stop_reader()
        if self.right_glove:
            self.right_glove.stop_reader()

    def visualize(self) -> None:
        """Placeholder for visualization method."""
        pass 

    def calibrate(self) -> None:
        """Calibrate all available gloves."""
        if self.left_glove:
            self.left_glove.calibrate()
        if self.right_glove:
            self.right_glove.calibrate()

    def diagnose(self) -> None:
        """Diagnose all available gloves with an interactive plot."""
        plt.style.use('dark_background')
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        num_sensors = Glove.NUM_TENSILE_SENSORS
        x = np.arange(num_sensors)
        left_line = None
        right_line = None
        if self.left_glove:
            left_line, = ax.plot(x, np.zeros(num_sensors), 'r-', label='Left Glove', linewidth=2)
        if self.right_glove:
            right_line, = ax.plot(x, np.zeros(num_sensors), 'b-', label='Right Glove', linewidth=2)
        ax.set_ylim(0, Glove.SENSOR_MAX_VALUE)
        ax.set_xlabel('Sensor Index')
        ax.set_ylabel('Tensile Value')
        ax.set_title('Glove Raw Tensile Data (Live)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        try:
            while plt.fignum_exists(fig.number):
                left_data = None
                right_data = None
                if self.left_glove:
                    try:
                        left_raw = self.left_glove.get_raw_data()
                        left_data = self.left_glove.parse_raw_data(left_raw)
                    except Exception as e:
                        print(f"Left glove error: {e}")
                if self.right_glove:
                    try:
                        right_raw = self.right_glove.get_raw_data()
                        right_data = self.right_glove.parse_raw_data(right_raw)
                    except Exception as e:
                        print(f"Right glove error: {e}")
                if left_data is not None and left_line is not None:
                    left_line.set_ydata(left_data.tensile_data)
                if right_data is not None and right_line is not None:
                    right_line.set_ydata(right_data.tensile_data)
                ax.set_title('Glove Raw Tensile Data (Live)')
                fig.canvas.draw()
                fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("Diagnosis stopped by user.")
        finally:
            plt.ioff()
            plt.close(fig)
            
