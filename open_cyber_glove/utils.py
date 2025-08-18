import pickle
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
NUM_JOINTS = 21
DEFAULT_GT_ORDER = [
    'index_mcp_flexion', 'index_mcp_abduction', 'index_pip_flexion', 'index_dip_flexion',
    'middle_mcp_flexion', 'middle_mcp_abduction', 'middle_pip_flexion', 'middle_dip_flexion',
    'ring_mcp_flexion', 'ring_mcp_abduction', 'ring_pip_flexion', 'ring_dip_flexion',
    'pinky_mcp_flexion', 'pinky_mcp_abduction', 'pinky_pip_flexion', 'pinky_dip_flexion',
    'thumb_mcp_flexion', 'thumb_mcp_abduction', 'thumb_pip_flexion', 'thumb_dip_flexion',
    'thumb_wrist_flexion', 'thumb_wrist_abduction'
]

def get_nested_value(data: dict, keys: List[str], default: Optional[Any]=None) -> Any:
    """Safely get nested dictionary value."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def build_joint_map(joint_names: List[List[str]]) -> Dict[str, int]:
    """Build joint index mapping for mocap format."""
    joint_map = {}
    idx = 1
    for f_idx, finger in enumerate(FINGER_NAMES):
        for j_name in joint_names[f_idx][1:]:
            joint_map[f"{finger}_{j_name}"] = idx
            idx += 1
    return joint_map

def get_joint_angles(
    pred_angles: dict,
    finger: str,
    joint: str,
    hand_type: str = 'right',
    backend=None,
    dtype=None,
    device=None
) -> Tuple[Any, Any]:
    """
    Get flexion and abduction angles for a joint, backend-agnostic.
    backend: np or torch module. dtype/device for torch.
    Returns: (flexion, abduction) as backend scalars/tensors.
    """
    joint_angles = get_nested_value(pred_angles, [finger, joint], {})
    if backend is not None and hasattr(backend, 'tensor'):
        # torch
        flexion = joint_angles.get('flexion', backend.tensor(0.0, dtype=dtype, device=device))
        abduction = joint_angles.get('abduction', backend.tensor(0.0, dtype=dtype, device=device))
    else:
        # numpy or python float
        flexion = joint_angles.get('flexion', 0.0)
        abduction = joint_angles.get('abduction', 0.0)
    if hand_type == 'left':
        abduction = -abduction
    if 'abduction' not in joint_angles:
        flexion = -flexion
    return flexion, abduction

def rotation_matrix(axis, theta, backend, dtype=None, device=None):
    """
    Unified rotation matrix for both single and batch input.
    - axis: (3,) or (B, 3)
    - theta: scalar or (B,)
    Returns: (3, 3) or (B, 3, 3)
    """
    if backend.__name__ == 'numpy':
        from scipy.spatial.transform import Rotation
        if axis.ndim == 1:
            return Rotation.from_rotvec(axis * theta).as_matrix()
        else:
            # Batch mode: axis (B, 3), theta (B,)
            return Rotation.from_rotvec(axis * theta[:, np.newaxis]).as_matrix()
    else:
        # Batch
        axis = axis / axis.norm(dim=1, keepdim=True)
        K = backend.zeros((axis.shape[0], 3, 3), dtype=dtype, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        I = backend.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(axis.shape[0], 3, 3)
        theta = theta.view(-1, 1, 1)
        sin_theta = backend.sin(theta)
        cos_theta = backend.cos(theta)
        return I + sin_theta * K + (1 - cos_theta) * backend.bmm(K, K)

def calculate_rotation_matrices(finger: str, prev_x: Any, prev_z: Any, 
                              flexion: float, abduction: float, joint_angles: dict, 
                              hand_type: str = 'right', backend: Any = np, dtype: Any = None, device: Any = None) -> Tuple[Any, Any]:
    """Calculate rotation matrices for flexion and abduction."""
    if finger != 'thumb':
        R_flex = rotation_matrix(prev_x, flexion, backend, dtype, device)
        R_abd = rotation_matrix(prev_z, abduction, backend, dtype, device)
    else:
        if 'abduction' not in joint_angles:
            if hand_type == 'left':
                flexion = -flexion
            R_flex = rotation_matrix(prev_z, flexion, backend, dtype, device)
            R_abd = backend.eye(3)
            if R_flex.ndim != R_abd.ndim:
                R_abd = R_abd.unsqueeze(0).expand(R_flex.shape[0], -1, -1)
        else:
            R_flex = rotation_matrix(prev_x, flexion, backend, dtype, device)
            R_abd = rotation_matrix(prev_z, abduction, backend, dtype, device)
    
    return R_flex, R_abd

def recover_joint_position(cs, link_length, flexion, abduction, backend):
    """
    Recover joint position from coordinate system and angles.
    Handles both batch and non-batch for torch and numpy.
    """
    if backend.__name__ == 'numpy':
        R = cs[:3, :3]
        t = cs[:3, 3]
        dir_local = backend.array([
            -backend.sin(abduction),
            backend.cos(abduction) * backend.cos(flexion),
            backend.cos(abduction) * backend.sin(flexion)
        ])
        return t + R @ (link_length * dir_local)
    else:
        R = cs[:3, :3]
        t = cs[:3, 3]
        dir_local = backend.stack([
            -backend.sin(abduction),
            backend.cos(abduction) * backend.cos(flexion),
            backend.cos(abduction) * backend.sin(flexion)
        ])
        return t.unsqueeze(-1) + backend.matmul(R, link_length * dir_local)

def process_finger_joints(
    hand_model: dict,
    pred_angles: dict,
    finger: str,
    joint_map: dict,
    fk_joints: Any,
    fk_rot: Any,
    hand_type: str,
    backend,
    dtype=None,
    device=None
) -> None:
    """
    Process joints for a single finger (excluding MCP).
    backend: np or torch
    """
    names = hand_model['joint_names'][FINGER_NAMES.index(finger)]
    mcp_cs = backend.array(hand_model['all_coordinates'][FINGER_NAMES.index(finger)][1]) if backend.__name__ == 'numpy' else backend.tensor(hand_model['all_coordinates'][FINGER_NAMES.index(finger)][1], dtype=dtype, device=device)
    prev_cs = mcp_cs.clone() if hasattr(mcp_cs, 'clone') else mcp_cs.copy()
    for j in range(2, len(names)):
        prev_joint_name = names[j-1]
        curr_joint_name = names[j]
        prev_key = f"{finger}_{prev_joint_name}"
        curr_key = f"{finger}_{curr_joint_name}"
        prev_idx = joint_map[prev_key]
        curr_idx = joint_map[curr_key]
        prev_pos = fk_joints[prev_idx] if backend.__name__ == 'numpy' else fk_joints[prev_idx]
        prev_x = prev_cs[:3, 0]
        prev_y = prev_cs[:3, 1]
        prev_z = prev_cs[:3, 2]
        joint_angles = get_nested_value(pred_angles, [finger, prev_joint_name], {})
        flexion, abduction = get_joint_angles(pred_angles, finger, prev_joint_name, hand_type, backend=backend, dtype=dtype, device=device)
        link_key = f"{prev_key}_to_{curr_key}"
        link_length = hand_model['link_lengths'][link_key]

        R_flex, R_abd = calculate_rotation_matrices(
            finger, prev_x, prev_z, flexion, abduction, joint_angles, hand_type, backend=backend, dtype=dtype, device=device
        )

        rotated_cs = backend.stack([prev_x, prev_y, prev_z], axis=1) if backend.__name__ == 'numpy' else backend.stack([prev_x, prev_y, prev_z], dim=1)
        rotated_cs = backend.matmul(R_abd, backend.matmul(R_flex, rotated_cs)) if backend.__name__ != 'numpy' else R_abd @ R_flex @ rotated_cs
        rot_y = rotated_cs[:, 1] if backend.__name__ == 'numpy' else rotated_cs[:, 1]
        curr_pos = prev_pos + link_length * rot_y
        fk_joints[curr_idx] = curr_pos
        # Update coordinate system for next iteration
        new_cs = prev_cs.copy() if hasattr(prev_cs, 'copy') else prev_cs.clone()
        new_cs[:3, :3] = rotated_cs
        new_cs[:3, 3] = curr_pos
        prev_cs = new_cs
        fk_rot[curr_idx, ...] = rotated_cs


def is_batch_pred_angles(pred_angles: dict) -> bool:
    """
    Return True if any value in pred_angles is a 1D tensor/array with len > 1.
    """
    for finger_dict in pred_angles.values():
        for joint_dict in finger_dict.values():
            for v in joint_dict.values():
                if hasattr(v, 'dim') and v.dim() == 1 and v.shape[0] > 1:
                    return True
                if hasattr(v, 'ndim') and v.ndim == 1 and v.shape[0] > 1:
                    return True
    return False


def forward_kinematics(
    hand_model: dict,
    pred_angles: dict,
    hand_type: str = 'right',
    backend=None,
    dtype=None,
    device=None,
) -> Tuple[Any, Any]:
    """
    Compute forward kinematics for a hand model.
    Supports both single-sample and batch input for pred_angles (always a dict):
      - Single mode: elements are scalars (float or 0-dim tensor)
      - Batch mode: elements are tensors of shape (B,)

    Args:
        hand_model: Hand model parameters (dict)
        pred_angles: Nested dict of predicted angles (see above)
        hand_type: 'right' or 'left'
        backend: np or torch module
        dtype: torch dtype (if using torch)
        device: torch device (if using torch)

    Returns:
        fk_joints: (NUM_JOINTS, 3) or (B, NUM_JOINTS, 3)
        fk_rot: (NUM_JOINTS, 3, 3) or (B, NUM_JOINTS, 3, 3)
    """
    is_batch = is_batch_pred_angles(pred_angles) # TODO(@jessey): refactor this later

    # --- Single-sample (non-batch) mode ---
    if not is_batch:
        if backend is None:
            raise ValueError('backend (np or torch) must be specified')
        # Allocate output arrays
        if backend.__name__ == 'numpy':
            fk_joints = backend.zeros((NUM_JOINTS, 3))
            fk_rot = backend.eye(3)[None, ...].repeat(NUM_JOINTS, axis=0)
            hand_model_joint_pos = backend.array(hand_model['joint_pos'])
        else:
            fk_joints = backend.zeros((NUM_JOINTS, 3), dtype=dtype, device=device)
            fk_rot = backend.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(NUM_JOINTS, 1, 1)
            hand_model_joint_pos = backend.tensor(hand_model['joint_pos'], dtype=dtype, device=device)
        joint_names = hand_model['joint_names']
        fk_joints[0] = hand_model_joint_pos[0]
        joint_map = build_joint_map(joint_names)
        # Process each finger
        for f_idx, finger in enumerate(FINGER_NAMES):
            if finger == 'thumb':
                # Special case: thumb MCP
                mcp_name = joint_names[0][1]
                mcp_key = f"{finger}_{mcp_name}"
                mcp_idx = joint_map[mcp_key]
                wrist_cs = backend.array(hand_model['all_coordinates'][0][0]) if backend.__name__ == 'numpy' else backend.tensor(hand_model['all_coordinates'][0][0], dtype=dtype, device=device)
                angles = get_nested_value(pred_angles, ['thumb', 'wrist'], {})
                flexion = angles.get('flexion', 0.0 if backend.__name__ == 'numpy' else backend.tensor(0.0, dtype=dtype, device=device))
                abduction = angles.get('abduction', 0.0 if backend.__name__ == 'numpy' else backend.tensor(0.0, dtype=dtype, device=device))
                calib_flexion = get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'flexion'])
                calib_abduction = get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'abduction'])
                link_length = hand_model['link_lengths']['wrist_to_thumb_mcp']
                if hand_type == 'left':
                    abduction = -abduction
                    calib_abduction = -calib_abduction
                thumb_mcp_pos = recover_joint_position(
                    wrist_cs, link_length, flexion + calib_flexion, abduction + calib_abduction, backend
                )
                fk_joints[mcp_idx] = thumb_mcp_pos
            else:
                mcp_name = joint_names[f_idx][1]
                mcp_key = f"{finger}_{mcp_name}"
                mcp_idx = joint_map[mcp_key]
                fk_joints[mcp_idx] = hand_model_joint_pos[mcp_idx]
            # Process remaining joints for this finger
            process_finger_joints(hand_model, pred_angles, finger, joint_map, fk_joints, fk_rot, hand_type, backend, dtype, device)
        return fk_joints, fk_rot

    # --- Batch mode (vectorized torch only) ---
    # Infer batch size B from any tensor in pred_angles
    B = None
    for finger_dict in pred_angles.values():
        for joint_dict in finger_dict.values():
            for v in joint_dict.values():
                B = v.shape[0]
                break
            break
        break
    # Allocate output arrays
    joint_names = hand_model['joint_names']
    hand_model_joint_pos = torch.tensor(hand_model['joint_pos'], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)  # (B, NUM_JOINTS, 3)
    fk_joints = torch.zeros((B, NUM_JOINTS, 3), dtype=dtype, device=device)
    fk_rot = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(B, NUM_JOINTS, 3, 3).clone()
    fk_joints[:, 0, :] = hand_model_joint_pos[:, 0, :]
    joint_map = build_joint_map(joint_names)
    # Process MCP for each finger
    for f_idx, finger in enumerate(FINGER_NAMES):
        if finger == 'thumb':
            mcp_name = joint_names[0][1]
            mcp_key = f"{finger}_{mcp_name}"
            mcp_idx = joint_map[mcp_key]
            wrist_cs = torch.tensor(hand_model['all_coordinates'][0][0], dtype=dtype, device=device)
            angles = get_nested_value(pred_angles, ['thumb', 'wrist'], {})
            flexion = angles.get('flexion', torch.zeros(B, dtype=dtype, device=device))
            abduction = angles.get('abduction', torch.zeros(B, dtype=dtype, device=device))
            calib_flexion = torch.tensor(get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'flexion']), dtype=dtype, device=device)
            calib_abduction = torch.tensor(get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'abduction']), dtype=dtype, device=device)
            link_length = torch.tensor(hand_model['link_lengths']["wrist_to_thumb_mcp"], dtype=dtype, device=device)
            if hand_type == 'left':
                abduction = -abduction
                calib_abduction = -calib_abduction
            # Vectorized thumb MCP position
            thumb_mcp_pos = recover_joint_position(
                wrist_cs, link_length, flexion + calib_flexion, abduction + calib_abduction, backend
            )
            fk_joints[:, mcp_idx, :] = thumb_mcp_pos.T
        else:
            mcp_name = joint_names[f_idx][1]
            mcp_key = f"{finger}_{mcp_name}"
            mcp_idx = joint_map[mcp_key]
            fk_joints[:, mcp_idx, :] = hand_model_joint_pos[:, mcp_idx, :]
    # Process all finger joints in batch
    for f_idx, finger in enumerate(FINGER_NAMES):
        names = joint_names[f_idx]
        mcp_cs = torch.tensor(hand_model['all_coordinates'][f_idx][1], dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        prev_cs = mcp_cs.clone()
        for j in range(2, len(names)):
            prev_joint_name = names[j-1]
            curr_joint_name = names[j]
            prev_key = f"{finger}_{prev_joint_name}"
            curr_key = f"{finger}_{curr_joint_name}"
            prev_idx = joint_map[prev_key]
            curr_idx = joint_map[curr_key]
            prev_pos = fk_joints[:, prev_idx, :]
            prev_x = prev_cs[:, :3, 0]
            prev_y = prev_cs[:, :3, 1]
            prev_z = prev_cs[:, :3, 2]
            joint_angles = get_nested_value(pred_angles, [finger, prev_joint_name], {})
            flexion, abduction = get_joint_angles(pred_angles, finger, prev_joint_name, hand_type, backend=backend, dtype=dtype, device=device)
            link_key = f"{prev_key}_to_{curr_key}"
            link_length = torch.tensor(hand_model['link_lengths'][link_key], dtype=dtype, device=device).expand(B)
            R_flex, R_abd = calculate_rotation_matrices(
                finger, prev_x, prev_z, flexion, abduction, joint_angles, hand_type, backend=backend, dtype=dtype, device=device
            )
            rotated_cs = torch.stack([prev_x, prev_y, prev_z], dim=2)  # (B, 3, 3)
            rotated_cs = torch.bmm(R_abd, torch.bmm(R_flex, rotated_cs))
            rot_y = rotated_cs[:, :, 1]
            curr_pos = prev_pos + link_length.unsqueeze(1) * rot_y
            fk_joints[:, curr_idx, :] = curr_pos
            new_cs = prev_cs.clone()
            new_cs[:, :3, :3] = rotated_cs
            new_cs[:, :3, 3] = curr_pos
            prev_cs = new_cs
            fk_rot[:, curr_idx, ...] = rotated_cs
    return fk_joints, fk_rot


def load_hand_model(hand_model_path: str) -> dict:
    """Load the hand model from the given path."""
    try:
        with open(hand_model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Hand model file not found: {hand_model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading hand model: {e}")
