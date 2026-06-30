import os
import time
import pickle
from typing import Optional, Dict, List
from .glove import Glove, GloveSensorData, glove_data_to_arrays
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

RECORDING_FORMAT_VERSION = 1
DEFAULT_SAMPLING_RATE_HZ = 120


def load_recording(path: str) -> dict:
    """
    Load a recording previously saved with :meth:`OpenCyberGlove.save_data`.

    Args:
        path: Path to the ``.pkl`` recording file.

    Returns:
        The structured payload dict (see :meth:`OpenCyberGlove.save_data`).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recording file not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def recording_quality_report(
    recording: dict,
    fps: int = DEFAULT_SAMPLING_RATE_HZ,
    flat_range_threshold: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Assess the quality of a recording (loaded payload or :meth:`OpenCyberGlove.save_data` output).

    For each hand it reports:
      - Timing: span, effective sample rate, and an estimate of dropped frames,
        derived from the device timestamps (handling uint32 wrap-around).
      - Per-sensor tensile activity: min/max/range, and which sensors stayed
        near-constant (range < ``flat_range_threshold``) — a sign of a sensor that
        was not exercised or is disconnected.

    Args:
        recording: Loaded recording dict (must contain a ``hands`` mapping).
        fps: Nominal sampling rate, used as a reference in the printout.
        flat_range_threshold: Tensile range below which a sensor is flagged "flat".
        verbose: If True, print a human-readable summary.

    Returns:
        Dict mapping hand name -> a report dict.
    """
    report = {}
    for hand, payload in recording.get('hands', {}).items():
        n = int(payload.get('num_frames', 0))
        rep = {'num_frames': n}

        ts = payload.get('timestamp')
        if n >= 2 and ts is not None and len(ts) >= 2:
            dt = np.diff(np.asarray(ts, dtype=np.int64))
            dt[dt < 0] += 2 ** 32  # uint32 microsecond timestamp wrap-around
            median_dt = float(np.median(dt))
            rep['span_s'] = float(dt.sum()) / 1e6
            if median_dt > 0:
                steps = np.maximum(np.round(dt / median_dt).astype(int) - 1, 0)
                rep['effective_fps'] = 1e6 / median_dt
                rep['estimated_dropped_frames'] = int(steps.sum())
                rep['num_gaps'] = int((steps > 0).sum())
            else:
                rep['effective_fps'] = None
                rep['estimated_dropped_frames'] = 0
                rep['num_gaps'] = 0

        tensile = payload.get('tensile')
        if tensile is not None and n > 0:
            tensile = np.asarray(tensile)
            tmin = tensile.min(axis=0)
            tmax = tensile.max(axis=0)
            trange = tmax - tmin
            rep['tensile_min'] = tmin.tolist()
            rep['tensile_max'] = tmax.tolist()
            rep['tensile_range'] = trange.tolist()
            rep['flat_sensors'] = np.where(trange < flat_range_threshold)[0].tolist()

        report[hand] = rep

        if verbose:
            print(f"\n=== Quality report [{hand}] ===")
            print(f"  frames: {n}")
            if 'span_s' in rep:
                efps = rep['effective_fps']
                rate = ('%.1f Hz' % efps) if efps else 'n/a'
                print(f"  span: {rep['span_s']:.2f}s   effective rate: {rate} "
                      f"(nominal {fps} Hz)")
                print(f"  estimated dropped frames: {rep['estimated_dropped_frames']} "
                      f"across {rep['num_gaps']} gap(s)")
            if 'tensile_range' in rep:
                rng = rep['tensile_range']
                print(f"  tensile active range per sensor: "
                      f"min={min(rng)}, max={max(rng)}")
                if rep['flat_sensors']:
                    print(f"  WARNING flat/near-constant sensors "
                          f"(range < {flat_range_threshold}): {rep['flat_sensors']} "
                          f"-> check wiring or that the sensor was exercised")
                else:
                    print(f"  all {len(rng)} sensors show movement")
    return report

class OpenCyberGlove:
    """
    SDK class to manage one or two gloves (left and/or right) in parallel.
    """
    def __init__(self, 
                 left_port: Optional[str] = None, 
                 right_port: Optional[str] = None,
                 model_path: Optional[str] = None,
                 glove_cls=Glove,
                 ):
        if not left_port and not right_port:
            raise ValueError("At least one of left_port or right_port must be provided.")
        self.left_glove: Optional[Glove] = glove_cls('left') if left_port else None
        self.right_glove: Optional[Glove] = glove_cls('right') if right_port else None
        self.left_port = left_port
        self.right_port = right_port
        self._running = False

        self.model = None
        if model_path:
            self.model = ort.InferenceSession(model_path)

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

    def get_data(self, hand_type: str) -> GloveSensorData:
        """
        Get sensor data from the specified glove.
        
        Args:
            hand_type (str): Type of hand ('left' or 'right')
            
        Returns:
            GloveSensorData: Sensor data from the specified glove
            
        Raises:
            ValueError: If hand_type is invalid
            RuntimeError: If the specified glove is not available
        """
        if hand_type == 'left':
            if not self.left_glove:
                raise RuntimeError("Left glove not available")
            return self.left_glove.get_data()
        elif hand_type == 'right':
            if not self.right_glove:
                raise RuntimeError("Right glove not available")
            return self.right_glove.get_data()
        else:
            raise ValueError(f"Invalid hand type: {hand_type}")
    
    def get_angles(self, hand_type: str, method: str = 'model') -> np.ndarray:
        """
        Get joint angles from the specified glove.
        
        Args:
            hand_type (str): Type of hand ('left' or 'right')
            method (str): Method to use for inference ('model' or 'linear')

        Returns:
            np.ndarray: Array of joint angles in radians for each finger joint
            
        Raises:
            ValueError: If hand_type is invalid
            RuntimeError: If the specified glove is not available
        """
        data = self.get_data(hand_type)
        if hand_type == 'left':
            if not self.left_glove:
                raise RuntimeError("Left glove not available")
            return self.left_glove.inference(data, method, model=self.model)
        elif hand_type == 'right':
            if not self.right_glove:
                raise RuntimeError("Right glove not available")
            return self.right_glove.inference(data, method, model=self.model)
        else:
            raise ValueError(f"Invalid hand type: {hand_type}")
        
    def visualize(self) -> None:
        """Placeholder for visualization method."""
        pass 

    def calibrate(self) -> None:
        """Calibrate all available gloves."""
        if self.left_glove:
            self.left_glove.calibrate()
        if self.right_glove:
            self.right_glove.calibrate()

    def experimental_advanced_calibrate(self, visualizer: 'HandVisualizer') -> None:
        """
        Experimental advanced calibration method.

        Note:
            This function is experimental. If you are experiencing issues with pinching accuracy,
            using this function may help you achieve a more accurate pinching distance.

            Usage:
                Run this function from your application or script to perform advanced calibration.
                You will be prompted in the command line to perform specific hand poses (e.g., "Pinch index finger and thumb").
                Please follow the on-screen instructions and press Enter when ready for each pose.
                The calibration process will collect data and optimize the glove's pinch accuracy.

            Example:
                sdk.experimental_advanced_calibrate(visualizer)
        """
        if self.left_glove:
            self.left_glove.advanced_calibrate(samples_count=500, model=self.model, vis=visualizer)
        if self.right_glove:
            self.right_glove.advanced_calibrate(samples_count=500, model=self.model, vis=visualizer)

    def _select_gloves(self, hand_type: Optional[str] = None) -> Dict[str, Glove]:
        """
        Resolve which connected gloves to operate on.

        Args:
            hand_type: 'left', 'right', or None for all connected gloves.

        Returns:
            Ordered dict mapping hand name -> Glove for the requested, available gloves.

        Raises:
            ValueError: If hand_type is invalid.
            RuntimeError: If the requested glove is not available, or none are.
        """
        available = {}
        if self.left_glove:
            available['left'] = self.left_glove
        if self.right_glove:
            available['right'] = self.right_glove

        if hand_type is None:
            if not available:
                raise RuntimeError("No glove available to record from.")
            return available
        if hand_type not in ('left', 'right'):
            raise ValueError(f"Invalid hand type: {hand_type}")
        if hand_type not in available:
            raise RuntimeError(f"{hand_type.capitalize()} glove not available")
        return {hand_type: available[hand_type]}

    def collect_data(
        self,
        samples_count: Optional[int] = 500,
        duration: Optional[float] = None,
        prompt: str = "Please perform the action you want to record.",
        hand_type: Optional[str] = None,
        wait_for_enter: bool = True,
        stall_timeout: Optional[float] = 5.0,
    ) -> Dict[str, List[GloveSensorData]]:
        """
        Frame-complete recording from one or both connected gloves.

        All requested gloves are drained within a single loop. Recording stops when
        every glove has reached ``samples_count`` frames, or when ``duration`` seconds
        have elapsed, whichever comes first. At least one limit must be provided.

        Note on dual-hand alignment: each glove has its own reader thread and device
        clock, so the two streams are co-recorded but NOT guaranteed to be frame-index
        aligned. Use the per-frame ``timestamp`` arrays to align left/right post-hoc.

        Args:
            samples_count: Target frames per glove (None to record purely by duration).
            duration: Maximum recording duration in seconds (None for count-only).
            prompt: Instruction shown to the operator before recording starts.
            hand_type: 'left', 'right', or None for all connected gloves.
            wait_for_enter: If True, wait for Enter before recording starts.
            stall_timeout: If no new frames arrive from any glove for this many
                seconds, stop early and keep what was collected (guards against an
                infinite wait when the stream stalls/disconnects). None disables it.

        Returns:
            Dict mapping hand name -> list of recorded :class:`GloveSensorData`.

        Raises:
            ValueError: If neither ``samples_count`` nor ``duration`` is provided.
        """
        if samples_count is None and duration is None:
            raise ValueError("Provide at least one of samples_count or duration.")
        gloves = self._select_gloves(hand_type)
        recordings: Dict[str, List[GloveSensorData]] = {name: [] for name in gloves}

        if wait_for_enter:
            hands_label = " & ".join(gloves)
            print(f"[{hands_label}] Press Enter to start recording: {prompt}")
            input()

        # Drop any frames buffered before recording starts (e.g. while the operator
        # was reading the prompt) so each hand's frame 0 is a post-start sample.
        for glove in gloves.values():
            glove.flush()

        # When recording purely by duration, show a time-based progress bar;
        # otherwise show a frame-count bar. Ctrl-C stops early and keeps what was
        # collected (useful for long recordings).
        use_time_bar = samples_count is None and duration is not None
        total = duration if use_time_bar else samples_count
        start = time.time()
        last_data = start
        try:
            with tqdm(total=total, desc="recording",
                      unit='s' if use_time_bar else 'it') as pbar:
                while True:
                    elapsed = time.time() - start
                    if duration is not None and elapsed >= duration:
                        break
                    if samples_count is not None and all(
                        len(recordings[name]) >= samples_count for name in gloves
                    ):
                        break
                    got_new = False
                    for name, glove in gloves.items():
                        if samples_count is not None and len(recordings[name]) >= samples_count:
                            continue
                        batch = glove.get_all_data()
                        if batch:
                            if samples_count is not None:
                                batch = batch[: samples_count - len(recordings[name])]
                            recordings[name].extend(batch)
                            got_new = True
                    if got_new:
                        last_data = time.time()
                    min_frames = min(len(recordings[name]) for name in gloves)
                    if use_time_bar:
                        pbar.n = round(min(elapsed, duration), 1)
                        pbar.set_postfix(frames=min_frames)
                        pbar.refresh()
                    elif samples_count is not None:
                        pbar.n = min_frames
                        pbar.refresh()
                    if not got_new:
                        if stall_timeout is not None and (time.time() - last_data) >= stall_timeout:
                            print(f"\nWARNING: no data for {stall_timeout:.0f}s — stopping "
                                  f"early (stream stalled or disconnected). Keeping "
                                  f"frames collected so far.")
                            break
                        time.sleep(0.001)
                # Ensure a duration-based bar lands exactly at 100% on a clean finish.
                if use_time_bar:
                    pbar.n = total
                    pbar.refresh()
        except KeyboardInterrupt:
            print("\nRecording interrupted by user; keeping frames collected so far.")

        for name in gloves:
            print(f"[{name}] recorded {len(recordings[name])} frames")
        return recordings

    def save_data(
        self,
        recordings: Dict[str, List[GloveSensorData]],
        path: str,
        prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Persist recordings to a structured pickle file.

        The saved payload is a plain dict of numpy arrays plus per-hand calibration
        parameters and session metadata. It does NOT pickle ``GloveSensorData``
        instances, so the file stays loadable even if that class changes.

        Args:
            recordings: Output of :meth:`collect_data` (hand name -> frame list).
            path: Destination ``.pkl`` path (parent directories are created).
            prompt: Optional prompt/label describing the recorded action.
            metadata: Optional extra metadata to store under the ``metadata`` key.

        Returns:
            The path the file was written to.
        """
        hands = {}
        for name, frames in recordings.items():
            glove = self.left_glove if name == 'left' else self.right_glove
            hand_payload = glove_data_to_arrays(frames)
            # Only persist calibration when the glove was actually calibrated;
            # otherwise the values are constructor placeholders (min=16384, max=0,
            # avg=0) that would silently corrupt downstream inference/replay.
            if glove is not None and glove.is_calibrated:
                hand_payload['calibration'] = {
                    'min_val': np.asarray(glove.min_val),
                    'max_val': np.asarray(glove.max_val),
                    'avg_val': np.asarray(glove.avg_val),
                    'custom_tuning': np.asarray(glove.custom_tuning),
                    'is_calibrated': True,
                    'sensor_order': list(glove.SENSOR_ORDER),
                }
            else:
                hand_payload['calibration'] = None
            hands[name] = hand_payload

        payload = {
            'format_version': RECORDING_FORMAT_VERSION,
            'fps': DEFAULT_SAMPLING_RATE_HZ,
            'prompt': prompt,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hands': hands,
            'metadata': metadata or {},
        }

        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        return path

    def collect_and_save(
        self,
        path: str,
        samples_count: Optional[int] = 500,
        duration: Optional[float] = None,
        prompt: str = "Please perform the action you want to record.",
        hand_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        wait_for_enter: bool = True,
        stall_timeout: Optional[float] = 5.0,
    ) -> str:
        """Convenience wrapper: record from the glove(s) then save to ``path``."""
        recordings = self.collect_data(
            samples_count=samples_count,
            duration=duration,
            prompt=prompt,
            hand_type=hand_type,
            wait_for_enter=wait_for_enter,
            stall_timeout=stall_timeout,
        )
        return self.save_data(recordings, path, prompt=prompt, metadata=metadata)

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
                        left_data = self.left_glove.get_data()
                    except Exception as e:
                        print(f"Left glove error: {e}")
                if self.right_glove:
                    try:
                        right_data = self.right_glove.get_data()
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

