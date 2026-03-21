"""
BadmintonAI - Pose Extraction Module
Extracts body landmarks from badminton video using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class PoseExtractor:
    """Extract pose landmarks from video files"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_from_video(self, video_path, output_path=None, visualize=False):
        """
        Extract pose landmarks from a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save JSON output (optional)
            visualize: If True, creates annotated video showing detected pose
            
        Returns:
            dict: Frame-by-frame pose data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {frame_count}")
        
        # Storage for pose data
        pose_data = {
            'video_info': {
                'filename': video_path.name,
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count
            },
            'frames': []
        }
        
        # Optional: Setup video writer for visualization
        video_writer = None
        if visualize:
            output_video_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(frame_rgb)
            
            # Extract landmarks if detected
            frame_data = {'frame_number': frame_idx, 'landmarks': None}
            
            if results.pose_landmarks:
                # Convert landmarks to dict format
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['landmarks'] = landmarks
                
                # Draw pose on frame if visualizing
                if visualize and video_writer:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            pose_data['frames'].append(frame_data)
            
            # Write annotated frame
            if visualize and video_writer:
                video_writer.write(frame)
            
            # Progress indicator
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')
        
        print(f"\n  Completed: {frame_idx} frames processed")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"  Annotated video saved: {output_video_path}")
        
        # Save to JSON if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(pose_data, f, indent=2)
            print(f"  Pose data saved: {output_path}")
        
        return pose_data
    
    def get_landmark_names(self):
        """Return list of all landmark names in order"""
        return [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 extract_pose.py <video_path> [--visualize]")
        print("Example: python3 extract_pose.py my_smash.mp4 --visualize")
        sys.exit(1)
    
    video_path = sys.argv[1]
    visualize = '--visualize' in sys.argv
    
    # Create output path
    video_path_obj = Path(video_path)
    output_json = video_path_obj.parent / f"{video_path_obj.stem}_pose_data.json"
    
    # Extract pose
    extractor = PoseExtractor()
    pose_data = extractor.extract_from_video(
        video_path, 
        output_path=output_json,
        visualize=visualize
    )
    
    # Quick stats
    frames_with_pose = sum(1 for f in pose_data['frames'] if f['landmarks'] is not None)
    total_frames = len(pose_data['frames'])
    detection_rate = (frames_with_pose / total_frames) * 100
    
    print(f"\nResults:")
    print(f"  Detection rate: {detection_rate:.1f}% ({frames_with_pose}/{total_frames} frames)")
    print(f"  Output: {output_json}")


if __name__ == "__main__":
    main()
