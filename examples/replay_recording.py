"""Replay a recorded .pkl as a 3D stick-figure hand.

Loads a recording saved by ``data_collection.py``, runs the recorded tensile data
back through the ONNX joint-angle model (using the calibration stored in the
recording), and animates the resulting hand skeleton. No glove hardware is required.

Two display backends:
  - ``matplotlib`` (default): a pure-matplotlib 3D skeleton. Works everywhere,
    including macOS, where Open3D's bundled GLFW can fail to open a window on newer
    OS versions ("Cocoa: Failed to find service port for display").
  - ``open3d``: the richer Open3D mesh renderer used by ``hello_world.py``.
"""
from open_cyber_glove.sdk import load_recording
from open_cyber_glove.glove import Glove, arrays_to_glove_data
from open_cyber_glove.visualizer import HandVisualizer
import argparse
import time
import numpy as np
import onnxruntime as ort


def prepare_angles(rec, model_path):
    """Run each recorded hand's tensile data through the ONNX model -> joint angles."""
    model = ort.InferenceSession(model_path)
    angles_by_hand = {}
    for hand, payload in rec['hands'].items():
        if payload.get('num_frames', 0) == 0:
            continue
        glove = Glove(hand)
        cal = payload.get('calibration') or {}
        if not cal.get('is_calibrated'):
            print(f"[{hand}] WARNING: recording has no calibration; replay angles "
                  f"will be inaccurate (model expects delta-from-rest input).")
        if 'avg_val' in cal:
            glove.avg_val = np.asarray(cal['avg_val'])
        if 'custom_tuning' in cal:
            glove.custom_tuning = np.asarray(cal['custom_tuning'])
        frames = arrays_to_glove_data(payload)
        # batch_inference omits custom_tuning; apply it to match the live path.
        angles = glove.batch_inference(frames, method='model', model=model)
        angles = angles * glove.custom_tuning
        angles_by_hand[hand] = angles
        print(f"[{hand}] prepared {len(angles)} frames for playback")
    return angles_by_hand


def compute_joints(angles_by_hand, calib_path):
    """Forward-kinematics every frame -> (num_frames, 21, 3) joint positions per hand.

    Uses a headless HandVisualizer (no Open3D window) purely for its kinematics.
    """
    fk = HandVisualizer(calib_path, create_window=False)
    return {
        hand: np.stack([fk.get_joints(angles[i], hand_type=hand)
                        for i in range(len(angles))])
        for hand, angles in angles_by_hand.items()
    }


def play_open3d(angles_by_hand, calib_path, fps, loop):
    """Animate with the Open3D mesh renderer (richer, but needs a working GL window)."""
    visualizer = HandVisualizer(calib_path)
    num_frames = max(len(a) for a in angles_by_hand.values())
    print(f"Playing back {num_frames} frames at {fps} Hz (open3d). Ctrl-C to stop.")
    try:
        while True:
            for i in range(num_frames):
                for hand, angles in angles_by_hand.items():
                    if i < len(angles):
                        visualizer.update(angles[i], hand_type=hand)
                time.sleep(1.0 / fps)
            if not loop:
                break
    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        visualizer.close()


# Per-finger palette (child joint of each bone decides its color).
# Deeper tones chosen for legibility on a white background.
FINGER_COLORS = {'thumb': '#e63946', 'index': '#f48c06', 'middle': '#2a9d8f',
                 'ring': '#1d6fe0', 'pinky': '#9b3fd1'}
_FINGER_RANGES = {'thumb': (1, 4), 'index': (5, 8), 'middle': (9, 12),
                  'ring': (13, 16), 'pinky': (17, 20)}


def _bone_color(a, b):
    j = max(a, b)
    for finger, (lo, hi) in _FINGER_RANGES.items():
        if lo <= j <= hi:
            return FINGER_COLORS[finger]
    return '#cfd3da'


def play_matplotlib(angles_by_hand, calib_path, fps, loop, _frame_hook=None, _save_path=None):
    """Animate the hand skeleton with a polished matplotlib 3D view (no Open3D window).

    _frame_hook / _save_path are test hooks (render given frames headlessly / save a PNG).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

    conns = HandVisualizer.HAND_CONNECTIONS
    joints_by_hand = compute_joints(angles_by_hand, calib_path)

    all_pts = np.concatenate([j.reshape(-1, 3) for j in joints_by_hand.values()])
    center = all_pts.mean(0)   # centroid keeps the hand centered in the view
    radius = max(np.abs(all_pts - center).max() * 1.05, 0.05)
    num_frames = max(len(a) for a in angles_by_hand.values())

    bg = 'white'
    plt.ion()
    fig = plt.figure(figsize=(6.5, 6.5))
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.6)   # zoom fills the frame without clipping poses
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-72)

    # Build per-hand artists once, then update their data each frame (fast + smooth).
    artists = {}
    for hand, J in joints_by_hand.items():
        p = J[0]
        lines = []
        for a, b in conns:
            ln, = ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]],
                          color=_bone_color(a, b), linewidth=5.0,
                          solid_capstyle='round', alpha=0.95, zorder=2)
            lines.append(ln)
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=52, c='#333333',
                        edgecolors='white', linewidths=0.6, depthshade=True, zorder=3)
        artists[hand] = (sc, lines)

    def set_frame(i):
        for hand, J in joints_by_hand.items():
            if i >= len(J):
                continue
            p = J[i]
            sc, lines = artists[hand]
            sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            for ln, (a, b) in zip(lines, conns):
                ln.set_data_3d([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]])

    if _frame_hook is not None:          # test path: render frames, optionally save, no loop
        for i in _frame_hook:
            set_frame(i)
        fig.canvas.draw()
        if _save_path:
            fig.savefig(_save_path, facecolor=fig.get_facecolor(), dpi=110)
        plt.close(fig)
        return

    print(f"Playing back {num_frames} frames at {fps} Hz (matplotlib). "
          f"Drag to rotate. Close the window or Ctrl-C to stop.")
    try:
        while True:
            for i in range(num_frames):
                if not plt.fignum_exists(fig.number):
                    return
                set_frame(i)
                plt.pause(1.0 / fps)   # draws + processes GUI events + sleeps
            if not loop:
                break
    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        plt.ioff()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded .pkl as a 3D stick-figure hand."
    )
    parser.add_argument('--input', type=str, required=True,
                        help="Path to a recording .pkl saved by data_collection.py.")
    parser.add_argument('--calib_path', type=str, default='model/hand_model.pkl',
                        help="Path to the hand model used for kinematics.")
    parser.add_argument('--model_path', type=str, default='model/model.onnx',
                        help="Path to the ONNX joint-angle model.")
    parser.add_argument('--backend', choices=['matplotlib', 'open3d'], default='matplotlib',
                        help="Display backend. 'matplotlib' (default) works on macOS; "
                             "'open3d' uses the richer mesh renderer (needs a working "
                             "GL window).")
    parser.add_argument('--fps', type=float, default=None,
                        help="Playback FPS (default: the recording's stored fps).")
    parser.add_argument('--loop', action='store_true',
                        help="Loop playback until the window is closed / Ctrl-C.")
    args = parser.parse_args()

    rec = load_recording(args.input)
    fps = args.fps or rec.get('fps', 120)
    angles_by_hand = prepare_angles(rec, args.model_path)
    if not angles_by_hand:
        print("Nothing to play back (no frames in recording).")
        raise SystemExit(0)

    if args.backend == 'open3d':
        play_open3d(angles_by_hand, args.calib_path, fps, args.loop)
    else:
        play_matplotlib(angles_by_hand, args.calib_path, fps, args.loop)


if __name__ == "__main__":
    main()
