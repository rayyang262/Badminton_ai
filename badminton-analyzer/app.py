import os
import cv2
import base64
import tempfile
import json
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB max

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv', 'MOV', 'MP4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {
        ext.lower() for ext in ALLOWED_EXTENSIONS
    }


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if total_frames == 0:
        cap.release()
        return []

    # Skip first/last 5% to avoid intros/outros
    start = int(total_frames * 0.05)
    end = int(total_frames * 0.95)
    available = end - start

    actual_frames = min(num_frames, available)
    if actual_frames < 1:
        actual_frames = 1

    indices = [
        int(start + available * i / max(actual_frames - 1, 1))
        for i in range(actual_frames)
    ]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to max 800px wide to reduce token usage
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        timestamp = idx / fps
        frames.append({
            'data': frame_b64,
            'timestamp': f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}",
        })

    cap.release()
    return frames


ANALYSIS_PROMPT = """\
You are an expert badminton coach with 20+ years of experience at national and international level. \
I will show you {n} frames sampled from a player's badminton video. \
Analyze the player's technique across every frame and provide a detailed, actionable coaching report.

Structure your report exactly as follows:

## 🏸 Overall Player Assessment
A 2–3 sentence summary of the player's apparent level (beginner / intermediate / advanced) and general strengths.

## 📋 Technical Analysis

### Footwork & Court Movement
Assess stance, split-step, recovery, and court coverage.

### Racket Technique & Grip
Assess grip style (forehand/backhand panhandle/thumb), racket angle at contact, wrist snap.

### Swing Mechanics
Assess backswing preparation, point of contact, follow-through, and body rotation.

### Court Positioning & Tactics
Assess base position, net coverage, rear-court depth, and tactical shot placement.

### Body Posture & Balance
Assess balance, weight transfer, shoulder alignment, and core engagement.

## 🎯 Priority Training Recommendations

List the **top 5 specific improvement areas**, ranked from most important to least. \
For each, include:
- **Issue**: What needs fixing
- **Drill**: A specific exercise or drill to address it
- **Cue**: One coaching cue to keep in mind during practice

## 💪 Strengths to Build On
2–3 genuine positives observed.

Be direct, specific, and actionable. Avoid vague advice."""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    def generate():
        if 'video' not in request.files:
            yield f"data: {json.dumps({'error': 'No video file provided.'})}\n\n"
            return

        file = request.files['video']
        if not file or file.filename == '':
            yield f"data: {json.dumps({'error': 'No file selected.'})}\n\n"
            return

        if not allowed_file(file.filename):
            yield f"data: {json.dumps({'error': 'Unsupported file type. Please upload MP4, MOV, AVI, WebM, or MKV.'})}\n\n"
            return

        api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
        if not api_key:
            yield f"data: {json.dumps({'error': 'ANTHROPIC_API_KEY is not set. Add it to your .env file.'})}\n\n"
            return

        suffix = os.path.splitext(file.filename)[1] or '.mp4'
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            yield f"data: {json.dumps({'status': 'Extracting frames from video...'})}\n\n"

            frames = extract_frames(tmp_path, num_frames=8)
            if not frames:
                yield f"data: {json.dumps({'error': 'Could not read frames. Ensure the video is a valid, non-corrupted file.'})}\n\n"
                return

            yield f"data: {json.dumps({'status': f'Sending {len(frames)} frames to Claude for analysis...'})}\n\n"

            content = [
                {
                    "type": "text",
                    "text": ANALYSIS_PROMPT.format(n=len(frames)),
                }
            ]
            for i, frame in enumerate(frames):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame['data'],
                    },
                })
                content.append({
                    "type": "text",
                    "text": f"[Frame {i + 1} — timestamp {frame['timestamp']}]",
                })

            client = anthropic.Anthropic(api_key=api_key)

            yield f"data: {json.dumps({'start': True})}\n\n"

            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=3000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": content}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': text})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Check your ANTHROPIC_API_KEY in .env.'})}\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait a moment and try again.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Unexpected error: {str(e)}'})}\n\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
