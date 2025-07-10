import numpy as np
import pickle
import open3d as o3d
from abc import ABC, abstractmethod
from .utils import forward_kinematics, DEFAULT_GT_ORDER
import queue

class BaseHandVisualizer(ABC):
    """
    Abstract base class for hand model visualization.
    
    This class defines the interface that all hand model implementations must follow.
    It provides abstract methods for initialization and updating the hand model visualization.
    """
    
    @abstractmethod
    def __init__(self, model_path: str):
        """
        Initialize the hand model with model data.
        
        Args:
            model_path (str): Path to the hand model file.
        """
        self.hand_model_path = model_path
        pass
    
    @abstractmethod
    def update(self, pose: np.ndarray, hand_type: str = 'right') -> None:
        """
        Update the visualization with a new hand pose.
        
        Args:
            pose (np.ndarray): Array of joint angles.
            hand_type (str): Type of hand ('left' or 'right').
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the visualization viewer if it is active.
        """
        pass


class HandVisualizer(BaseHandVisualizer):
    """
    Visualizes a hand model using 3D rendering based on joint angles and calibration data.
    """
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    def __init__(self, model_path: str):
        """
        Initialize the HandVisualizer with hand model data.
        Args:
            model_path (str): Path to the hand model file.
        """
        super().__init__(model_path)
        self.hand_model = self._load_hand_model()
        
        self.joint_radius = self.hand_model.get('joint_radius', 0.005)
        self.bone_radius = self.hand_model.get('bone_radius', 0.0025)
        self.joint_color = self.hand_model.get('joint_color', [0.9, 0.1, 0.1, 1.0])
        self.bone_color = self.hand_model.get('bone_color', [0.1, 0.1, 0.9, 1.0])
        
        # Visualization objects will be created in the background thread
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.node_map = {}
        
    def _load_hand_model(self) -> dict:
        """
        Load hand model data from the specified file.
        Returns:
            dict: Hand model configuration data.
        """
        try:
            with open(self.hand_model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Hand model file not found: {self.hand_model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading hand model from {self.hand_model_path}: {e}")
        
    def _add_camera_and_light(self) -> None:
        """
        Add a camera and lighting to the scene.
        """ 
        # Set camera parameters
        ctr = self.vis.get_view_control()
        
        # Calculate camera position based on hand model spread
        left_joints = self.hand_model['left']['joint_pos']
        right_joints = self.hand_model['right']['joint_pos']
        all_joints = np.vstack([left_joints, right_joints])
        max_spread = np.max(np.ptp(all_joints, axis=0))
        distance = max(0.3, max_spread * 3.0)
        
        # Set camera position
        eye = np.array([0, 0, distance])
        target = np.array([0, 0, 0])
        up = np.array([0, -1, 0])
        
        ctr.set_front(eye - target)
        ctr.set_lookat(target)
        ctr.set_up(up)
        ctr.set_zoom(0.8)

    def _init_hand_model(self) -> None:
        """
        Initialize the hand model with default poses.
        """
        # If no window was created, skip hand model initialization            
        print("Creating initial hand poses...")
        init_angles = np.zeros(len(DEFAULT_GT_ORDER))
        print("Updating right hand...")
        self.update(init_angles, hand_type='right')
        print("Updating left hand...")
        self.update(init_angles, hand_type='left')
        print("Hand model initialization complete")
    
    def get_joints(self, pose: np.ndarray, hand_type: str = 'right') -> np.ndarray:
        """
        Get the joints positions for a given hand pose.
        Args:
            pose (np.ndarray): Array of joint angles.
            hand_type (str): Type of hand ('left' or 'right').
        """
        angle_dict = {}
        for i, angle_name in enumerate(DEFAULT_GT_ORDER):
            finger, joint, dof = angle_name.split('_')
            angle_dict.setdefault(finger, {}).setdefault(joint, {})[dof] = pose[i]
        joints, _ = forward_kinematics(self.hand_model[hand_type], angle_dict, hand_type)
        joints = joints / 1000
        return joints
    
    def update(self, pose: np.ndarray, hand_type: str = 'right') -> None:
        """
        Update the visualization with a new hand pose.
        Args:
            pose (np.ndarray): Array of joint angles.
            hand_type (str): Type of hand ('left' or 'right').
        """ 
        joints = self.get_joints(pose, hand_type)
        
        for idx, pos in enumerate(joints):
            key = f'joint_{hand_type}_{idx}'
            if key not in self.node_map:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.joint_radius)
                sphere.paint_uniform_color(self.joint_color[:3])  # Open3D uses RGB
                sphere.translate(pos)
                self.node_map[key] = sphere
                self.vis.add_geometry(self.node_map[key])
            else:
                sphere = self.node_map[key]
                # To move the sphere to a new absolute position, we compute
                # the translation vector from its current center to the new position.
                current_center = sphere.get_center()
                required_translation = pos - current_center
                sphere.translate(required_translation)
                self.vis.update_geometry(sphere)
                
        # Update bones
        for i, j in self.HAND_CONNECTIONS:
            key = f'bone_{hand_type}_{i}_{j}'
            self._add_bone(key, joints[i], joints[j])
        self.vis.poll_events()
        self.vis.update_renderer()
            
    def _add_bone(self, key: str, start: np.ndarray, end: np.ndarray) -> None:
        """
        Add or update a bone (cylinder) between two joints in the scene.
        Args:
            key (str): Unique key for the bone.
            start (np.ndarray): Start joint position.
            end (np.ndarray): End joint position.
        """
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
            
        # Create cylinder using Open3D
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.bone_radius, height=length
        )
        cylinder.paint_uniform_color(self.bone_color[:3])  # Open3D uses RGB
        
        # Calculate rotation to align cylinder with direction
        z = direction / length
        y = np.array([0, 1, 0])
        x = np.cross(y, z) if not (np.allclose(z, y) or np.allclose(z, -y)) else np.array([1, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        
        # Create rotation matrix
        rot = np.eye(4)
        rot[:3, 0] = x
        rot[:3, 1] = y
        rot[:3, 2] = z
        rot[:3, 3] = start + direction / 2
        
        # Apply transformation
        cylinder.transform(rot)
        
        if key in self.node_map:
            # Remove old bone and add new one
            self.vis.remove_geometry(self.node_map[key], False)
            self.node_map[key] = cylinder
            self.vis.add_geometry(cylinder, False)
        else:
            # Add new bone
            self.node_map[key] = cylinder
            self.vis.add_geometry(cylinder, False)

    def close(self) -> None:
        """
        Close the visualization viewer.
        """
        self.vis.destroy_window()
