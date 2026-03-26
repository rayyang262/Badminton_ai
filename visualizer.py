"""
BadmintonAI - Rich Video Visualizer
Annotates input video with pose skeleton, joint angles, and technique feedback.

Usage:
    python visualizer.py input.mp4 [--reference pro_reference_smash.json] [--output output.mp4]
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

from extract_pose import PoseExtractor
from analyze_movement import MovementAnalyzer


# ---------------------------------------------------------------------------
# FrameAnalyzer — angle math without file I/O
# ---------------------------------------------------------------------------

class FrameAnalyzer(MovementAnalyzer):
    """Subclass of MovementAnalyzer that skips the file-loading __init__."""

    JOINT_NAMES = [
        'right_elbow', 'left_elbow',
        'right_knee', 'left_knee',
        'right_shoulder', 'left_shoulder',
        'right_hip', 'left_hip',
    ]

    # Key landmark indices used for frame-level confidence
    KEY_LANDMARKS = [
        'right_shoulder', 'left_shoulder',
        'right_elbow', 'left_elbow',
        'right_wrist', 'left_wrist',
        'right_hip', 'left_hip',
        'right_knee', 'left_knee',
    ]

    def __init__(self):
        # Deliberately skip parent __init__ (which requires a file path)
        pass

    def get_all_joint_angles(self, landmarks):
        """
        Compute angles for all 8 major joints.

        Args:
            landmarks: list of 33 dicts with x, y, z, visibility keys

        Returns:
            dict: joint_name -> float (degrees) or None
        """
        return {
            joint: self.get_joint_angle(landmarks, joint)
            for joint in self.JOINT_NAMES
        }

    def get_frame_confidence(self, landmarks):
        """
        Average visibility of the key landmark points (0.0 – 1.0).
        """
        visibilities = [
            landmarks[self.LANDMARKS[name]]['visibility']
            for name in self.KEY_LANDMARKS
        ]
        return float(np.mean(visibilities))


# ---------------------------------------------------------------------------
# HUDRenderer — all OpenCV drawing
# ---------------------------------------------------------------------------

class HUDRenderer:
    """Stateless drawing helpers. All methods operate on the passed frame/canvas."""

    # BGR color scheme
    COLOR_GREEN  = (50, 205, 50)
    COLOR_YELLOW = (0, 200, 255)
    COLOR_RED    = (0, 60, 220)
    COLOR_GREY   = (150, 150, 150)
    COLOR_WHITE  = (255, 255, 255)
    COLOR_BLACK  = (0, 0, 0)
    HUD_BG       = (28, 28, 28)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Map MediaPipe connection (sorted tuple) -> which joint "owns" it
    # Connections not listed here render as grey (torso, face, etc.)
    SEGMENT_JOINT = {
        (11, 13): 'left_shoulder',
        (13, 15): 'left_elbow',
        (12, 14): 'right_shoulder',
        (14, 16): 'right_elbow',
        (23, 25): 'left_hip',
        (25, 27): 'left_knee',
        (24, 26): 'right_hip',
        (26, 28): 'right_knee',
        (11, 23): 'left_hip',
        (12, 24): 'right_hip',
    }

    # Per-joint pixel offsets for angle label placement
    LABEL_OFFSETS = {
        'right_elbow':   ( 14, -14),
        'left_elbow':    (-55, -14),
        'right_knee':    ( 14,  20),
        'left_knee':     (-55,  20),
        'right_shoulder':( 14, -20),
        'left_shoulder': (-55, -20),
        'right_hip':     ( 14,  14),
        'left_hip':      (-55,  14),
    }

    def __init__(self, frame_width, frame_height, hud_width=340):
        self.W = frame_width
        self.H = frame_height
        self.hud_width = hud_width
        self.scale = min(1.0, frame_height / 720)

    # ------------------------------------------------------------------ helpers

    def _px(self, lm):
        """Convert normalized landmark to clamped pixel coords."""
        x = max(0, min(self.W - 1, int(lm['x'] * self.W)))
        y = max(0, min(self.H - 1, int(lm['y'] * self.H)))
        return x, y

    def _fs(self, s):
        """Scale a font size by frame height."""
        return max(0.3, s * self.scale)

    def _draw_text_with_bg(self, img, text, origin, font_scale, color, thickness=1):
        """Draw text with a filled black background rectangle."""
        (tw, th), baseline = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        ox, oy = origin
        ox = max(5, min(img.shape[1] - tw - 5, ox))
        oy = max(th + 5, min(img.shape[0] - 5, oy))
        cv2.rectangle(img, (ox - 2, oy - th - 2), (ox + tw + 2, oy + baseline + 2),
                      self.COLOR_BLACK, -1)
        cv2.putText(img, text, (ox, oy), self.FONT, font_scale, color, thickness,
                    cv2.LINE_AA)

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1,
                          dash_len=8, gap_len=5):
        """Simulate a dashed line using short segments."""
        pt1, pt2 = np.array(pt1, float), np.array(pt2, float)
        dist = np.linalg.norm(pt2 - pt1)
        if dist < 1:
            return
        step = dash_len + gap_len
        n = int(dist / step)
        for i in range(n + 1):
            s_frac = i * step / dist
            e_frac = min(1.0, (i * step + dash_len) / dist)
            s = tuple((pt1 + s_frac * (pt2 - pt1)).astype(int))
            e = tuple((pt1 + e_frac * (pt2 - pt1)).astype(int))
            cv2.line(img, s, e, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ skeleton

    def draw_skeleton(self, frame, landmarks, joint_colors):
        """
        Draw skeleton onto frame.

        Args:
            landmarks: list of 33 landmark dicts
            joint_colors: dict joint_name -> BGR color tuple
        """
        mp_pose = mp.solutions.pose

        for conn in mp_pose.POSE_CONNECTIONS:
            a_idx, b_idx = int(conn[0]), int(conn[1])
            la, lb = landmarks[a_idx], landmarks[b_idx]

            vis_min = min(la['visibility'], lb['visibility'])
            low_vis = vis_min < 0.5
            thickness = max(1, int(3 * vis_min))

            key = tuple(sorted((a_idx, b_idx)))
            joint_name = self.SEGMENT_JOINT.get(key)
            color = joint_colors.get(joint_name, self.COLOR_GREY) if joint_name else self.COLOR_GREY

            pa = self._px(la)
            pb = self._px(lb)

            if low_vis:
                self._draw_dashed_line(frame, pa, pb, self.COLOR_GREY, thickness=1)
            else:
                cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

        # Draw joint circles
        for name in FrameAnalyzer.JOINT_NAMES:
            idx = MovementAnalyzer.LANDMARKS[name]
            lm = landmarks[idx]
            color = joint_colors.get(name, self.COLOR_GREY)
            radius = max(3, int(5 * self.scale))
            vis = lm['visibility']
            if vis >= 0.5:
                cv2.circle(frame, self._px(lm), radius, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, self._px(lm), radius, self.COLOR_GREY, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ labels

    def draw_joint_angle_labels(self, frame, landmarks, angles, joint_colors):
        """Draw angle value text near each major joint."""
        fs = self._fs(0.48)
        for joint_name, angle in angles.items():
            idx = MovementAnalyzer.LANDMARKS[joint_name]
            lm = landmarks[idx]
            px, py = self._px(lm)
            dx, dy = self.LABEL_OFFSETS.get(joint_name, (12, -12))
            ox, oy = px + dx, py + dy
            text = f"{angle:.0f}deg" if angle is not None else "---"
            color = joint_colors.get(joint_name, self.COLOR_GREY)
            self._draw_text_with_bg(frame, text, (ox, oy), fs, color)

    # ------------------------------------------------------------------ HUD

    def draw_hud_panel(self, canvas, hud_data):
        """
        Draw the full HUD sidebar into the right portion of canvas.

        Args:
            canvas: full-width output canvas (numpy array)
            hud_data: dict with keys: frame_idx, total_frames, confidence,
                      deviations, top_issues, low_vis_names
        """
        x0 = self.W   # HUD starts here
        hud = canvas[:, x0:]

        # Background
        hud[:] = self.HUD_BG

        pad = 8
        y = 18
        w = self.hud_width
        fs_small  = self._fs(0.40)
        fs_normal = self._fs(0.44)
        fs_header = self._fs(0.42)

        def label(text, yy, color=None, fs=None, bold=False):
            cv2.putText(hud, text, (pad, yy), self.FONT,
                        fs or fs_normal, color or self.COLOR_WHITE,
                        2 if bold else 1, cv2.LINE_AA)

        def divider(yy):
            cv2.line(hud, (pad, yy), (w - pad, yy), (60, 60, 60), 1)

        # Frame counter
        fi = hud_data['frame_idx'] + 1
        ft = hud_data['total_frames']
        label(f"FRAME  {fi:04d} / {ft:04d}", y, self.COLOR_WHITE, fs_header, bold=True)
        y += 20

        # Confidence bar
        conf = hud_data['confidence']
        bar_x, bar_w, bar_h = pad, w - pad * 2 - 38, int(11 * self.scale)
        bar_h = max(8, bar_h)
        cv2.rectangle(hud, (bar_x, y), (bar_x + bar_w, y + bar_h), (55, 55, 55), -1)
        fill = int(bar_w * conf)
        bar_color = self.COLOR_GREEN if conf > 0.7 else (self.COLOR_YELLOW if conf > 0.4 else self.COLOR_RED)
        if fill > 0:
            cv2.rectangle(hud, (bar_x, y), (bar_x + fill, y + bar_h), bar_color, -1)
        pct_text = f"{int(conf * 100)}%"
        cv2.putText(hud, pct_text, (bar_x + bar_w + 4, y + bar_h - 1),
                    self.FONT, fs_small, self.COLOR_WHITE, 1, cv2.LINE_AA)
        y += bar_h + 8

        divider(y); y += 10

        # Joint angles table header
        label("JOINT ANGLES", y, self.COLOR_GREY, fs_small)
        y += 14
        # Column headers
        cols = [(pad, "Joint"), (140, "Now"), (192, "Tgt"), (240, "Diff")]
        for cx, ch in cols:
            cv2.putText(hud, ch, (cx, y), self.FONT, fs_small, (100, 100, 100), 1, cv2.LINE_AA)
        y += 14

        display_joints = [
            ('right_elbow',    'R.Elbow'),
            ('right_knee',     'R.Knee'),
            ('right_shoulder', 'R.Shld'),
            ('right_hip',      'R.Hip'),
            ('left_elbow',     'L.Elbow'),
            ('left_knee',      'L.Knee'),
        ]

        for jkey, jlabel in display_joints:
            dev = hud_data['deviations'].get(jkey, {})
            current = dev.get('current')
            target  = dev.get('target')
            diff    = dev.get('diff')
            color   = dev.get('color', self.COLOR_GREY)

            curr_str = f"{current:.0f}d" if current is not None else "---"
            tgt_str  = f"{target:.0f}d"  if target  is not None else " --"
            diff_str = (f"{diff:+.0f}d"  if diff    is not None else "  --")

            cv2.putText(hud, jlabel,    (pad, y), self.FONT, fs_small, self.COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(hud, curr_str,  (140, y), self.FONT, fs_small, color,            1, cv2.LINE_AA)
            cv2.putText(hud, tgt_str,   (192, y), self.FONT, fs_small, self.COLOR_GREY,  1, cv2.LINE_AA)
            cv2.putText(hud, diff_str,  (240, y), self.FONT, fs_small, color,            1, cv2.LINE_AA)
            y += 16

        y += 2
        divider(y); y += 10

        # Top issues
        label("TOP ISSUES", y, self.COLOR_GREY, fs_small)
        y += 14
        for issue_text, issue_color in hud_data['top_issues']:
            cv2.putText(hud, issue_text, (pad, y), self.FONT, fs_small, issue_color, 1, cv2.LINE_AA)
            y += 15

        y += 4
        divider(y); y += 10

        # Low visibility
        low_vis = hud_data.get('low_vis_names', [])
        if low_vis:
            label("Low visibility:", y, self.COLOR_GREY, fs_small)
            y += 13
            for name, vis_val in low_vis[:4]:
                cv2.putText(hud, f"  {name} ({vis_val:.2f})", (pad, y),
                            self.FONT, fs_small, (100, 100, 100), 1, cv2.LINE_AA)
                y += 13

    # ------------------------------------------------------------------ no-pose

    def draw_pose_not_detected(self, frame):
        """Draw red wash + alert text over the video frame region."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        text = "POSE NOT DETECTED"
        fs = self._fs(1.0)
        (tw, th), _ = cv2.getTextSize(text, self.FONT, fs, 2)
        cx = (self.W - tw) // 2
        cy = self.H // 2
        cv2.putText(frame, text, (cx, cy), self.FONT, fs, self.COLOR_WHITE, 2, cv2.LINE_AA)

        sub = "Check lighting and framing"
        fs2 = self._fs(0.5)
        (sw, _), _ = cv2.getTextSize(sub, self.FONT, fs2, 1)
        cv2.putText(frame, sub, ((self.W - sw) // 2, cy + th + 10),
                    self.FONT, fs2, self.COLOR_GREY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# BadmintonVisualizer — orchestrator
# ---------------------------------------------------------------------------

class BadmintonVisualizer:
    """End-to-end: reads video, annotates each frame, writes output video."""

    # Which reference stat to compare per joint
    REFERENCE_STAT = {
        'right_elbow':    ('elbow',    'max_angle'),
        'left_elbow':     ('elbow',    'max_angle'),
        'right_knee':     ('knee',     'min_angle'),
        'left_knee':      ('knee',     'min_angle'),
        'right_shoulder': ('shoulder', 'max_angle'),
        'left_shoulder':  ('shoulder', 'max_angle'),
        'right_hip':      ('hip',      'max_angle'),
        'left_hip':       ('hip',      'max_angle'),
    }

    def __init__(self, reference_path=None):
        self.extractor = PoseExtractor()
        self.analyzer  = FrameAnalyzer()
        self.renderer  = None  # created once frame dimensions are known
        self.reference = self._load_reference(reference_path)

    def _load_reference(self, path):
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            print(f"[warn] Reference file not found: {path} — running without comparison")
            return None
        with open(path) as f:
            ref = json.load(f)
        print(f"[info] Loaded pro reference: {path.name}")
        return ref

    # ------------------------------------------------------------------ deviation

    def _deviation_color(self, std_ratio):
        if std_ratio is None:
            return HUDRenderer.COLOR_GREY
        if abs(std_ratio) <= 1.0:
            return HUDRenderer.COLOR_GREEN
        if abs(std_ratio) <= 2.0:
            return HUDRenderer.COLOR_YELLOW
        return HUDRenderer.COLOR_RED

    def _compute_deviations(self, angles):
        """
        For each joint, compute deviation from pro reference.

        Returns:
            dict: joint_name -> {current, target, diff, std_ratio, color}
        """
        result = {}
        for joint_name, angle in angles.items():
            ref_joint_key, ref_stat_key = self.REFERENCE_STAT[joint_name]
            dev = {'current': angle, 'target': None, 'diff': None, 'std_ratio': None,
                   'color': HUDRenderer.COLOR_GREY}

            if self.reference and angle is not None:
                ref_joint = self.reference.get(ref_joint_key)
                if ref_joint:
                    ref_stat = ref_joint.get(ref_stat_key)
                    if ref_stat:
                        mean = ref_stat.get('mean')
                        std  = ref_stat.get('std')
                        if mean is not None:
                            diff = angle - mean
                            dev['target'] = mean
                            dev['diff']   = diff
                            if std and std >= 0.5:
                                dev['std_ratio'] = diff / std
                            dev['color'] = self._deviation_color(dev['std_ratio'])

            result[joint_name] = dev
        return result

    def _get_top_issues(self, deviations):
        """
        Build top-issues list sorted by severity (worst first).
        Returns list of (text, color) tuples.
        """
        scored = []
        for joint_name, dev in deviations.items():
            sr = dev.get('std_ratio')
            diff = dev.get('diff')
            color = dev.get('color', HUDRenderer.COLOR_GREY)
            if sr is None or diff is None:
                continue
            label_map = {
                'right_elbow': 'R.Elbow ext', 'left_elbow': 'L.Elbow ext',
                'right_knee': 'R.Knee bend', 'left_knee': 'L.Knee bend',
                'right_shoulder': 'R.Shoulder', 'left_shoulder': 'L.Shoulder',
                'right_hip': 'R.Hip', 'left_hip': 'L.Hip',
            }
            label = label_map.get(joint_name, joint_name)
            if abs(sr) <= 1.0:
                text = f"OK {label}: on target"
            else:
                sign = "+" if diff > 0 else ""
                text = f"{'!' if abs(sr) > 2 else '~'} {label}: {sign}{diff:.0f}d from target"
            scored.append((abs(sr), text, color))

        scored.sort(key=lambda x: -x[0])
        return [(t, c) for _, t, c in scored[:5]]

    def _get_low_vis_landmarks(self, landmarks):
        """Return list of (name, visibility) for key landmarks below threshold."""
        result = []
        for name in FrameAnalyzer.KEY_LANDMARKS:
            idx = MovementAnalyzer.LANDMARKS[name]
            vis = landmarks[idx]['visibility']
            if vis < 0.5:
                result.append((name, vis))
        result.sort(key=lambda x: x[1])
        return result

    # ------------------------------------------------------------------ main loop

    def process_video(self, input_path, output_path):
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        HUD_W = 340
        out_w = fw + HUD_W

        print(f"[info] Input:  {input_path.name}  ({fw}x{fh}, {fps:.1f}fps, {frame_count} frames)")
        print(f"[info] Output: {output_path}  ({out_w}x{fh})")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, fh))
        if not writer.isOpened():
            alt = str(output_path).replace('.mp4', '.avi')
            print(f"[warn] mp4v failed, falling back to XVID -> {alt}")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(alt, fourcc, fps, (out_w, fh))
            output_path = Path(alt)

        # Initialize renderer now that we have dimensions
        self.renderer = HUDRenderer(fw, fh, hud_width=HUD_W)

        frame_idx = 0
        detected  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = self.extractor.pose.process(frame_rgb)

            # Draw annotations on frame directly — cap.read() returns a
            # C-contiguous array, so OpenCV drawing functions work correctly.
            # We copy it into the canvas AFTER drawing (not before), avoiding
            # the non-contiguous slice problem.
            if results.pose_landmarks:
                detected += 1
                landmarks = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]

                angles     = self.analyzer.get_all_joint_angles(landmarks)
                confidence = self.analyzer.get_frame_confidence(landmarks)
                deviations = self._compute_deviations(angles)
                joint_colors = {j: d['color'] for j, d in deviations.items()}
                top_issues = self._get_top_issues(deviations)
                low_vis    = self._get_low_vis_landmarks(landmarks)

                self.renderer.draw_skeleton(frame, landmarks, joint_colors)
                self.renderer.draw_joint_angle_labels(frame, landmarks, angles, joint_colors)

                hud_data = {
                    'frame_idx':    frame_idx,
                    'total_frames': frame_count,
                    'confidence':   confidence,
                    'deviations':   deviations,
                    'top_issues':   top_issues,
                    'low_vis_names': low_vis,
                }
            else:
                self.renderer.draw_pose_not_detected(frame)
                hud_data = {
                    'frame_idx':    frame_idx,
                    'total_frames': frame_count,
                    'confidence':   0.0,
                    'deviations':   {j: {'current': None, 'target': None, 'diff': None,
                                         'std_ratio': None, 'color': HUDRenderer.COLOR_GREY}
                                     for j in FrameAnalyzer.JOINT_NAMES},
                    'top_issues':   [],
                    'low_vis_names': [],
                }

            # Build full-width canvas with the annotated frame on the left
            canvas = np.zeros((fh, out_w, 3), dtype=np.uint8)
            canvas[:, :fw] = frame

            self.renderer.draw_hud_panel(canvas, hud_data)
            writer.write(canvas)

            frame_idx += 1
            if frame_idx % 15 == 0 or frame_idx == frame_count:
                pct = frame_idx / max(frame_count, 1) * 100
                print(f"  Processing: {pct:.1f}%  ({frame_idx}/{frame_count})", end='\r')

        print()
        cap.release()
        writer.release()

        det_rate = detected / max(frame_idx, 1) * 100
        print(f"[done] Detection rate: {det_rate:.1f}% ({detected}/{frame_idx} frames)")
        print(f"[done] Saved: {output_path}")
        return str(output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Badminton technique visualizer — annotates video with pose & feedback'
    )
    parser.add_argument('input',
                        help='Input video path (mp4, mov, avi, …)')
    parser.add_argument('--reference', default=None,
                        help='Pro reference JSON (e.g. pro_reference_smash.json)')
    parser.add_argument('--output', default=None,
                        help='Output video path (default: <input>_visualized.mp4)')
    args = parser.parse_args()

    output = args.output
    if output is None:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}_visualized.mp4")

    viz = BadmintonVisualizer(reference_path=args.reference)
    viz.process_video(args.input, output)


if __name__ == '__main__':
    main()
