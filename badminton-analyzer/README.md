# 🏸 Badminton AI Coach

Upload a badminton video and get an expert coaching report powered by **Claude Opus 4.6**.

The app extracts 8 key frames from your video, sends them to Claude for vision analysis, and streams back a structured coaching report covering footwork, swing mechanics, grip, court positioning, and prioritised training recommendations.

## Features

- Drag-and-drop video upload (MP4, MOV, AVI, WebM, MKV — up to 150 MB)
- In-browser video preview before submitting
- Real-time streaming analysis
- Structured coaching report with ranked training priorities
- Copy report to clipboard

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/badminton-analyzer.git
cd badminton-analyzer

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com/).

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Deployment

The app is a standard Flask application. Deploy to any platform that supports Python:

- **Railway / Render / Fly.io** — push the repo and set the `ANTHROPIC_API_KEY` environment variable
- **Heroku** — add a `Procfile`: `web: python app.py`
- **Docker** — works out of the box; add `gunicorn` for production

## Tech Stack

| Layer    | Tech |
|----------|------|
| Backend  | Python · Flask · OpenCV |
| AI       | Claude Opus 4.6 (vision + adaptive thinking) |
| Frontend | Vanilla JS · marked.js for markdown |
| Styling  | Pure CSS (dark theme) |
