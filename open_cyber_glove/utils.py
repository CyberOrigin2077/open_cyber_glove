import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional

# Constants
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
NUM_JOINTS = 21

def load_hand_model(hand_model_path: str) -> dict:
    """
    Load the hand model from the given path.
    """
    try:
        with open(hand_model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Hand model file not found: {hand_model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading hand model: {e}")

def recover_joint_position(cs: np.ndarray, link_length: float, flexion: float, abduction: float) -> np.ndarray:
    """Recover joint position from coordinate system and angles."""
    R = cs[:3, :3]
    t = cs[:3, 3]
    dir_local = np.array([
        -np.sin(abduction),
        np.cos(abduction) * np.cos(flexion),
        np.cos(abduction) * np.sin(flexion)
    ])
    return t + R @ (link_length * dir_local)

def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Create rotation matrix using scipy's Rotation class."""
    return Rotation.from_rotvec(axis * theta).as_matrix()

def get_nested_value(data: dict, keys: List[str], default=None):
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

def get_joint_angles(pred_angles: dict, finger: str, joint: str, hand_type: str = 'right') -> Tuple[float, float]:
    """Get flexion and abduction angles for a joint."""
    joint_angles = get_nested_value(pred_angles, [finger, joint], {})
    flexion = joint_angles.get('flexion', 0.0)
    abduction = joint_angles.get('abduction', 0.0)
    
    # Handle left hand mirroring
    if hand_type == 'left':
        abduction = -abduction
    
    # Handle 1-DOF joints
    if 'abduction' not in joint_angles:
        flexion = -flexion
    
    return flexion, abduction

def calculate_rotation_matrices(finger: str, prev_x: np.ndarray, prev_z: np.ndarray, 
                              flexion: float, abduction: float, joint_angles: dict, 
                              hand_type: str = 'right') -> Tuple[np.ndarray, np.ndarray]:
    """Calculate rotation matrices for flexion and abduction."""
    if finger != 'thumb':
        R_flex = rotation_matrix(prev_x, flexion)
        R_abd = rotation_matrix(prev_z, abduction)
    else:
        if 'abduction' not in joint_angles:
            if hand_type == 'left':
                flexion = -flexion
            R_flex = rotation_matrix(prev_z, flexion)
            R_abd = np.eye(3)
        else:
            R_flex = rotation_matrix(prev_x, flexion)
            R_abd = rotation_matrix(prev_z, abduction)
    
    return R_flex, R_abd

def process_thumb_mcp(hand_model: dict, pred_angles: dict, joint_map: dict, 
                     fk_joints: np.ndarray, hand_type: str) -> None:
    """Process thumb MCP joint position."""
    finger = 'thumb'
    mcp_name = hand_model['joint_names'][0][1]  # thumb MCP name
    mcp_key = f"{finger}_{mcp_name}"
    mcp_idx = joint_map[mcp_key]
    
    # Get wrist coordinate system and angles
    wrist_cs = hand_model['all_coordinates'][0][0].copy()
    joint_angles = get_nested_value(pred_angles, ['thumb', 'wrist'], {})
    flexion = pred_angles.get('flexion', 0.0)
    abduction = pred_angles.get('abduction', 0.0)
    
    # Get calibration angles
    calib_flexion = get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'flexion'])
    calib_abduction = get_nested_value(hand_model, ['angles', 'thumb', 'wrist', 'abduction'])
    
    if calib_flexion is None or calib_abduction is None:
        raise ValueError("Missing calibration angles for thumb wrist joint")
    
    # Get link length and handle left hand
    link_length = hand_model['link_lengths'].get("wrist_to_thumb_mcp", 0.0)
    if hand_type == 'left':
        abduction = -abduction
        calib_abduction = -calib_abduction
    
    # Calculate thumb MCP position
    thumb_mcp_pos = recover_joint_position(
        wrist_cs, link_length, flexion + calib_flexion, abduction + calib_abduction
    )
    fk_joints[mcp_idx] = thumb_mcp_pos

def process_finger_joints(hand_model: dict, pred_angles: dict, finger: str, 
                         joint_map: dict, fk_joints: np.ndarray, fk_rot: np.ndarray,
                         hand_type: str) -> None:
    """Process joints for a single finger (excluding MCP)."""
    names = hand_model['joint_names'][FINGER_NAMES.index(finger)]
    mcp_cs = np.array(hand_model['all_coordinates'][FINGER_NAMES.index(finger)][1].copy())
    prev_cs = mcp_cs
    
    # Process subsequent joints (PIP, DIP, TIP)
    for j in range(2, len(names)):
        prev_joint_name = names[j-1]
        curr_joint_name = names[j]
        
        prev_key = f"{finger}_{prev_joint_name}"
        curr_key = f"{finger}_{curr_joint_name}"
        prev_idx = joint_map[prev_key]
        curr_idx = joint_map[curr_key]
        
        # Get previous joint position and coordinate system
        prev_pos = fk_joints[prev_idx]
        prev_x = prev_cs[:3, 0]  # X-axis for flexion
        prev_y = prev_cs[:3, 1]  # Y-axis points to next joint
        prev_z = prev_cs[:3, 2]  # Z-axis for abduction
        
        # Get joint angles
        joint_angles = get_nested_value(pred_angles, [finger, prev_joint_name], {})
        flexion, abduction = get_joint_angles(pred_angles, finger, prev_joint_name, hand_type)
        
        # Get link length
        link_key = f"{prev_key}_to_{curr_key}"
        link_length = hand_model['link_lengths'].get(link_key, 0.0)
        
        # Calculate rotation matrices
        R_flex, R_abd = calculate_rotation_matrices(
            finger, prev_x, prev_z, flexion, abduction, joint_angles, hand_type
        )
        
        # Apply rotations to coordinate system
        rotated_cs = np.column_stack([prev_x, prev_y, prev_z])
        rotated_cs = R_abd @ R_flex @ rotated_cs
        
        # Calculate current joint position
        rot_y = rotated_cs[:, 1]
        curr_pos = prev_pos + link_length * rot_y
        fk_joints[curr_idx] = curr_pos
        
        # Update coordinate system for next iteration
        prev_cs[:3, :3] = rotated_cs
        prev_cs[:3, 3] = curr_pos
        fk_rot[curr_idx, ...] = rotated_cs

def forward_kinematics(hand_model: dict, pred_angles: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics for the hand model.
    """
    # Initialize output arrays
    fk_joints = np.zeros((NUM_JOINTS, 3))
    fk_rot = np.concatenate([np.eye(3)[None, ...]] * NUM_JOINTS, axis=0)
    
    # Extract model parameters
    joint_names = hand_model['joint_names']
    calib_joints = hand_model['joint_pos']
    hand_type = hand_model.get('hand_type', 'right')
    
    # Set wrist position
    fk_joints[0] = calib_joints[0].copy()
    
    # Build joint mapping
    joint_map = build_joint_map(joint_names)
    
    # Process each finger
    for f_idx, finger in enumerate(FINGER_NAMES):
        # Handle MCP joint
        if finger == 'thumb':
            process_thumb_mcp(hand_model, pred_angles, joint_map, fk_joints, hand_type)
        else:
            mcp_name = joint_names[f_idx][1]
            mcp_key = f"{finger}_{mcp_name}"
            mcp_idx = joint_map[mcp_key]
            fk_joints[mcp_idx] = calib_joints[mcp_idx].copy()
        
        # Process remaining finger joints
        process_finger_joints(hand_model, pred_angles, finger, joint_map, fk_joints, fk_rot, hand_type)
    
    return fk_joints, fk_rot
