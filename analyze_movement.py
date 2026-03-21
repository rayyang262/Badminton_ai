"""
BadmintonAI - Movement Analysis Module
Calculates biomechanical metrics from pose data
"""

import numpy as np
import json
from pathlib import Path


class MovementAnalyzer:
    """Analyze movement patterns from pose data"""
    
    # MediaPipe landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8,
        'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_pinky': 17, 'right_pinky': 18,
        'left_index': 19, 'right_index': 20,
        'left_thumb': 21, 'right_thumb': 22,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }
    
    def __init__(self, pose_data_path):
        """
        Initialize analyzer with pose data
        
        Args:
            pose_data_path: Path to JSON file from extract_pose.py
        """
        with open(pose_data_path, 'r') as f:
            self.pose_data = json.load(f)
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points (in degrees)
        
        Args:
            point1, point2, point3: Each is dict with 'x', 'y', 'z' keys
            
        Returns:
            float: Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        p1 = np.array([point1['x'], point1['y'], point1['z']])
        p2 = np.array([point2['x'], point2['y'], point2['z']])
        p3 = np.array([point3['x'], point3['y'], point3['z']])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def get_joint_angle(self, frame_landmarks, joint_name):
        """
        Calculate specific joint angle
        
        Args:
            frame_landmarks: List of landmarks for a single frame
            joint_name: One of 'right_elbow', 'left_elbow', 'right_knee', 'left_knee', 
                       'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip'
                       
        Returns:
            float: Angle in degrees, or None if landmarks not visible
        """
        # Define the three points for each joint
        joint_configs = {
            'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
            'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
            'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
            'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
            'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
            'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        }
        
        if joint_name not in joint_configs:
            raise ValueError(f"Unknown joint: {joint_name}")
        
        p1_name, p2_name, p3_name = joint_configs[joint_name]
        
        # Get landmark indices
        idx1 = self.LANDMARKS[p1_name]
        idx2 = self.LANDMARKS[p2_name]
        idx3 = self.LANDMARKS[p3_name]
        
        # Check visibility (threshold = 0.5)
        if (frame_landmarks[idx1]['visibility'] < 0.5 or
            frame_landmarks[idx2]['visibility'] < 0.5 or
            frame_landmarks[idx3]['visibility'] < 0.5):
            return None
        
        # Calculate angle
        return self.calculate_angle(
            frame_landmarks[idx1],
            frame_landmarks[idx2],
            frame_landmarks[idx3]
        )
    
    def analyze_smash(self):
        """
        Analyze smash movement pattern
        
        Returns:
            dict: Analysis results with key metrics
        """
        frames_with_pose = [f for f in self.pose_data['frames'] if f['landmarks'] is not None]
        
        if not frames_with_pose:
            return {'error': 'No pose data found'}
        
        # Analyze right arm (assuming right-handed player)
        elbow_angles = []
        shoulder_angles = []
        knee_angles = []
        
        for frame in frames_with_pose:
            landmarks = frame['landmarks']
            
            # Right elbow angle
            elbow_angle = self.get_joint_angle(landmarks, 'right_elbow')
            if elbow_angle:
                elbow_angles.append(elbow_angle)
            
            # Right shoulder angle
            shoulder_angle = self.get_joint_angle(landmarks, 'right_shoulder')
            if shoulder_angle:
                shoulder_angles.append(shoulder_angle)
            
            # Right knee (for power generation check)
            knee_angle = self.get_joint_angle(landmarks, 'right_knee')
            if knee_angle:
                knee_angles.append(knee_angle)
        
        # Calculate stats
        results = {
            'shot_type': 'smash',
            'total_frames_analyzed': len(frames_with_pose),
            'elbow': {
                'min_angle': float(np.min(elbow_angles)) if elbow_angles else None,
                'max_angle': float(np.max(elbow_angles)) if elbow_angles else None,
                'avg_angle': float(np.mean(elbow_angles)) if elbow_angles else None,
                'range_of_motion': float(np.max(elbow_angles) - np.min(elbow_angles)) if elbow_angles else None
            },
            'shoulder': {
                'min_angle': float(np.min(shoulder_angles)) if shoulder_angles else None,
                'max_angle': float(np.max(shoulder_angles)) if shoulder_angles else None,
                'avg_angle': float(np.mean(shoulder_angles)) if shoulder_angles else None,
            },
            'knee': {
                'min_angle': float(np.min(knee_angles)) if knee_angles else None,
                'max_angle': float(np.max(knee_angles)) if knee_angles else None,
                'avg_angle': float(np.mean(knee_angles)) if knee_angles else None,
            }
        }
        
        return results
    
    def compare_to_reference(self, your_metrics, reference_metrics):
        """
        Compare your metrics to reference (pro) metrics
        
        Args:
            your_metrics: dict from analyze_smash()
            reference_metrics: dict with same structure (pro player data)
            
        Returns:
            dict: Comparison results
        """
        comparison = {
            'elbow_contact_angle_diff': None,
            'shoulder_angle_diff': None,
            'knee_bend_diff': None,
            'feedback': []
        }
        
        # Compare elbow at max extension (likely contact point)
        if your_metrics['elbow']['max_angle'] and reference_metrics['elbow']['max_angle']:
            diff = your_metrics['elbow']['max_angle'] - reference_metrics['elbow']['max_angle']
            comparison['elbow_contact_angle_diff'] = round(diff, 1)
            
            if abs(diff) > 10:
                if diff < 0:
                    comparison['feedback'].append(
                        f"Your elbow is {abs(diff):.1f}° less extended at contact. "
                        "Try to reach higher and snap through the shuttle."
                    )
                else:
                    comparison['feedback'].append(
                        f"Your elbow is {diff:.1f}° more extended at contact. "
                        "You might be overextending - focus on timing."
                    )
        
        # Compare knee bend (power generation)
        if your_metrics['knee']['min_angle'] and reference_metrics['knee']['min_angle']:
            diff = your_metrics['knee']['min_angle'] - reference_metrics['knee']['min_angle']
            comparison['knee_bend_diff'] = round(diff, 1)
            
            if diff > 10:
                comparison['feedback'].append(
                    f"Your knee bends {diff:.1f}° less than optimal. "
                    "Bend your knees more to generate power from your legs."
                )
        
        if not comparison['feedback']:
            comparison['feedback'].append("Your technique looks solid! Keep practicing.")
        
        return comparison


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_movement.py <pose_data.json>")
        sys.exit(1)
    
    pose_data_path = sys.argv[1]
    
    analyzer = MovementAnalyzer(pose_data_path)
    results = analyzer.analyze_smash()
    
    print("\n=== Movement Analysis Results ===\n")
    print(f"Shot type: {results['shot_type']}")
    print(f"Frames analyzed: {results['total_frames_analyzed']}\n")
    
    print("Elbow:")
    print(f"  Min angle: {results['elbow']['min_angle']:.1f}°")
    print(f"  Max angle: {results['elbow']['max_angle']:.1f}°")
    print(f"  Range of motion: {results['elbow']['range_of_motion']:.1f}°\n")
    
    print("Shoulder:")
    print(f"  Min angle: {results['shoulder']['min_angle']:.1f}°")
    print(f"  Max angle: {results['shoulder']['max_angle']:.1f}°\n")
    
    print("Knee (right):")
    print(f"  Min angle: {results['knee']['min_angle']:.1f}°")
    print(f"  Max angle: {results['knee']['max_angle']:.1f}°\n")
    
    # Example comparison to pro reference
    print("=== Comparison to Pro Reference ===\n")
    
    # These are example pro values - you'd build a real library
    pro_reference = {
        'elbow': {'max_angle': 168.0, 'min_angle': 45.0},
        'shoulder': {'max_angle': 165.0},
        'knee': {'min_angle': 95.0}
    }
    
    comparison = analyzer.compare_to_reference(results, pro_reference)
    
    for feedback in comparison['feedback']:
        print(f"• {feedback}")


if __name__ == "__main__":
    main()
