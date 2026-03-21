"""
BadmintonAI - Pro Reference Library Builder
Analyzes multiple pro player videos and creates reference standards
"""

import json
import numpy as np
from pathlib import Path
from analyze_movement import MovementAnalyzer


class ProReferenceBuilder:
    """Build reference standards from multiple pro player videos"""
    
    def __init__(self):
        self.pro_data = []
    
    def add_pro_video(self, pose_data_path, player_name=None):
        """
        Add a pro player's pose data to the reference library
        
        Args:
            pose_data_path: Path to pose JSON file
            player_name: Optional name of the player
        """
        pose_data_path = Path(pose_data_path)
        
        if not player_name:
            player_name = pose_data_path.stem.replace('_pose_data', '')
        
        print(f"Analyzing {player_name}...")
        
        try:
            analyzer = MovementAnalyzer(pose_data_path)
            metrics = analyzer.analyze_smash()
            
            if 'error' not in metrics:
                self.pro_data.append({
                    'player_name': player_name,
                    'file': str(pose_data_path),
                    'metrics': metrics
                })
                print(f"  ✓ Added {player_name}")
                print(f"    Elbow max: {metrics['elbow']['max_angle']:.1f}°")
                print(f"    Knee min: {metrics['knee']['min_angle']:.1f}°")
            else:
                print(f"  ✗ Error analyzing {player_name}: {metrics['error']}")
        
        except Exception as e:
            print(f"  ✗ Failed to analyze {player_name}: {e}")
    
    def build_reference(self, output_path='pro_reference_smash.json'):
        """
        Calculate average metrics across all pro players
        
        Args:
            output_path: Where to save the reference file
        """
        if not self.pro_data:
            print("No pro data added yet!")
            return None
        
        print(f"\n{'='*60}")
        print("BUILDING PRO REFERENCE STANDARD")
        print(f"{'='*60}")
        print(f"Players analyzed: {len(self.pro_data)}")
        
        # Collect all metrics
        elbow_max = []
        elbow_min = []
        elbow_avg = []
        elbow_rom = []
        
        shoulder_max = []
        shoulder_min = []
        shoulder_avg = []
        
        knee_max = []
        knee_min = []
        knee_avg = []
        
        for data in self.pro_data:
            m = data['metrics']
            
            if m['elbow']['max_angle']:
                elbow_max.append(m['elbow']['max_angle'])
            if m['elbow']['min_angle']:
                elbow_min.append(m['elbow']['min_angle'])
            if m['elbow']['avg_angle']:
                elbow_avg.append(m['elbow']['avg_angle'])
            if m['elbow']['range_of_motion']:
                elbow_rom.append(m['elbow']['range_of_motion'])
            
            if m['shoulder']['max_angle']:
                shoulder_max.append(m['shoulder']['max_angle'])
            if m['shoulder']['min_angle']:
                shoulder_min.append(m['shoulder']['min_angle'])
            if m['shoulder']['avg_angle']:
                shoulder_avg.append(m['shoulder']['avg_angle'])
            
            if m['knee']['max_angle']:
                knee_max.append(m['knee']['max_angle'])
            if m['knee']['min_angle']:
                knee_min.append(m['knee']['min_angle'])
            if m['knee']['avg_angle']:
                knee_avg.append(m['knee']['avg_angle'])
        
        # Calculate averages and standard deviations
        reference = {
            'shot_type': 'smash',
            'source': f'Average of {len(self.pro_data)} pro players',
            'players': [d['player_name'] for d in self.pro_data],
            'elbow': {
                'max_angle': {
                    'mean': float(np.mean(elbow_max)) if elbow_max else None,
                    'std': float(np.std(elbow_max)) if elbow_max else None,
                    'min': float(np.min(elbow_max)) if elbow_max else None,
                    'max': float(np.max(elbow_max)) if elbow_max else None,
                },
                'min_angle': {
                    'mean': float(np.mean(elbow_min)) if elbow_min else None,
                    'std': float(np.std(elbow_min)) if elbow_min else None,
                },
                'range_of_motion': {
                    'mean': float(np.mean(elbow_rom)) if elbow_rom else None,
                    'std': float(np.std(elbow_rom)) if elbow_rom else None,
                }
            },
            'shoulder': {
                'max_angle': {
                    'mean': float(np.mean(shoulder_max)) if shoulder_max else None,
                    'std': float(np.std(shoulder_max)) if shoulder_max else None,
                },
                'min_angle': {
                    'mean': float(np.mean(shoulder_min)) if shoulder_min else None,
                    'std': float(np.std(shoulder_min)) if shoulder_min else None,
                }
            },
            'knee': {
                'max_angle': {
                    'mean': float(np.mean(knee_max)) if knee_max else None,
                    'std': float(np.std(knee_max)) if knee_max else None,
                },
                'min_angle': {
                    'mean': float(np.mean(knee_min)) if knee_min else None,
                    'std': float(np.std(knee_min)) if knee_min else None,
                }
            }
        }
        
        # Print summary
        print("\nPro Reference Standard (Smash):")
        print(f"\nElbow at Contact (max extension):")
        print(f"  Average: {reference['elbow']['max_angle']['mean']:.1f}°")
        print(f"  Range: {reference['elbow']['max_angle']['min']:.1f}° - {reference['elbow']['max_angle']['max']:.1f}°")
        print(f"  Std Dev: ±{reference['elbow']['max_angle']['std']:.1f}°")
        
        print(f"\nKnee Bend (min angle):")
        print(f"  Average: {reference['knee']['min_angle']['mean']:.1f}°")
        print(f"  Std Dev: ±{reference['knee']['min_angle']['std']:.1f}°")
        
        print(f"\nElbow Range of Motion:")
        print(f"  Average: {reference['elbow']['range_of_motion']['mean']:.1f}°")
        print(f"  Std Dev: ±{reference['elbow']['range_of_motion']['std']:.1f}°")
        
        # Save to file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(reference, f, indent=2)
        
        print(f"\n✓ Reference standard saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return reference
    
    def compare_to_reference(self, your_pose_data_path, reference_path='pro_reference_smash.json'):
        """
        Compare your technique to the pro reference standard
        
        Args:
            your_pose_data_path: Your pose data JSON
            reference_path: Pro reference JSON
        """
        # Load reference
        with open(reference_path, 'r') as f:
            reference = json.load(f)
        
        # Analyze your video
        analyzer = MovementAnalyzer(your_pose_data_path)
        your_metrics = analyzer.analyze_smash()
        
        print(f"\n{'='*60}")
        print("COMPARISON TO PRO STANDARD")
        print(f"{'='*60}")
        print(f"Your video: {Path(your_pose_data_path).stem}")
        print(f"Reference: {reference['source']}")
        print(f"Pro players: {', '.join(reference['players'])}")
        print(f"{'='*60}\n")
        
        feedback = []
        
        # Compare elbow at contact
        if your_metrics['elbow']['max_angle'] and reference['elbow']['max_angle']['mean']:
            your_elbow = your_metrics['elbow']['max_angle']
            pro_elbow = reference['elbow']['max_angle']['mean']
            pro_std = reference['elbow']['max_angle']['std']
            
            diff = your_elbow - pro_elbow
            
            print(f"Elbow Extension at Contact:")
            print(f"  You: {your_elbow:.1f}°")
            print(f"  Pro average: {pro_elbow:.1f}° (±{pro_std:.1f}°)")
            print(f"  Difference: {diff:+.1f}°")
            
            if abs(diff) > pro_std:
                if diff < 0:
                    feedback.append({
                        'priority': 'HIGH',
                        'issue': f"Your elbow is {abs(diff):.1f}° less extended at contact",
                        'suggestion': "Reach higher on your smash. Focus on full arm extension at the point of contact with the shuttle."
                    })
                else:
                    feedback.append({
                        'priority': 'MEDIUM',
                        'issue': f"Your elbow is {diff:.1f}° more extended than average",
                        'suggestion': "You might be overextending. Focus on timing - contact the shuttle slightly earlier."
                    })
            else:
                print("  ✓ Within pro range!")
            print()
        
        # Compare knee bend
        if your_metrics['knee']['min_angle'] and reference['knee']['min_angle']['mean']:
            your_knee = your_metrics['knee']['min_angle']
            pro_knee = reference['knee']['min_angle']['mean']
            pro_std = reference['knee']['min_angle']['std']
            
            diff = your_knee - pro_knee
            
            print(f"Knee Bend (power generation):")
            print(f"  You: {your_knee:.1f}°")
            print(f"  Pro average: {pro_knee:.1f}° (±{pro_std:.1f}°)")
            print(f"  Difference: {diff:+.1f}°")
            
            if diff > pro_std:
                feedback.append({
                    'priority': 'HIGH',
                    'issue': f"Your knees bend {diff:.1f}° less than pros",
                    'suggestion': "Bend your knees more! Power comes from your legs. Get lower before jumping into your smash."
                })
            elif diff < -pro_std:
                print("  ✓ Good knee bend (maybe even deeper than average)")
            else:
                print("  ✓ Within pro range!")
            print()
        
        # Compare range of motion
        if your_metrics['elbow']['range_of_motion'] and reference['elbow']['range_of_motion']['mean']:
            your_rom = your_metrics['elbow']['range_of_motion']
            pro_rom = reference['elbow']['range_of_motion']['mean']
            pro_std = reference['elbow']['range_of_motion']['std']
            
            diff = your_rom - pro_rom
            
            print(f"Elbow Range of Motion:")
            print(f"  You: {your_rom:.1f}°")
            print(f"  Pro average: {pro_rom:.1f}° (±{pro_std:.1f}°)")
            print(f"  Difference: {diff:+.1f}°")
            
            if abs(diff) > pro_std:
                if diff < 0:
                    feedback.append({
                        'priority': 'MEDIUM',
                        'issue': f"Your swing has {abs(diff):.1f}° less range of motion",
                        'suggestion': "Work on a fuller swing. Start with your elbow more bent, then explode through to full extension."
                    })
            else:
                print("  ✓ Within pro range!")
            print()
        
        # Print prioritized feedback
        print(f"{'='*60}")
        print("ACTIONABLE FEEDBACK")
        print(f"{'='*60}\n")
        
        if not feedback:
            print("✓ Your technique looks solid across all metrics!")
            print("  You're within the professional range. Keep practicing!")
        else:
            # Sort by priority
            priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            feedback.sort(key=lambda x: priority_order[x['priority']])
            
            for i, item in enumerate(feedback, 1):
                print(f"{i}. [{item['priority']}] {item['issue']}")
                print(f"   → {item['suggestion']}\n")
        
        print(f"{'='*60}\n")
        
        return feedback


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("\n1. Build pro reference from multiple pro videos:")
        print("   python3 build_pro_reference.py build pro1_pose_data.json pro2_pose_data.json ...")
        print("\n2. Compare your video to pro reference:")
        print("   python3 build_pro_reference.py compare your_pose_data.json")
        print("\nExamples:")
        print("   python3 build_pro_reference.py build axelsen_pose_data.json momota_pose_data.json")
        print("   python3 build_pro_reference.py compare my_smash_pose_data.json")
        sys.exit(1)
    
    command = sys.argv[1]
    builder = ProReferenceBuilder()
    
    if command == 'build':
        # Build reference from multiple pro videos
        if len(sys.argv) < 3:
            print("Error: Provide at least one pro player pose data file")
            sys.exit(1)
        
        pro_files = sys.argv[2:]
        
        for pro_file in pro_files:
            builder.add_pro_video(pro_file)
        
        if builder.pro_data:
            builder.build_reference()
        else:
            print("No valid pro data to build reference!")
    
    elif command == 'compare':
        # Compare your video to existing reference
        if len(sys.argv) < 3:
            print("Error: Provide your pose data file to compare")
            sys.exit(1)
        
        your_file = sys.argv[2]
        
        if not Path('pro_reference_smash.json').exists():
            print("Error: pro_reference_smash.json not found!")
            print("First build the reference with: python3 build_pro_reference.py build <pro_files>")
            sys.exit(1)
        
        builder.compare_to_reference(your_file)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'build' or 'compare'")


if __name__ == "__main__":
    main()
