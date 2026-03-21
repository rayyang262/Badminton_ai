# BadmintonAI - Movement Analysis Tool

AI-powered badminton technique analyzer that compares your movements to professional players and gives you specific, quantified feedback.

---

## 🎯 What This Does

Record yourself playing badminton. BadmintonAI will:
1. Extract your body movement data frame-by-frame
2. Calculate biomechanical metrics (joint angles, range of motion)
3. Compare your technique to professional players
4. Tell you **exactly** what's different and how to improve

Instead of vague advice like "swing harder," you get:
> "Your elbow is 15.6° less extended at contact than Viktor Axelsen's. Reach higher on your smash."

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

This installs:
- `mediapipe` - Google's pose estimation
- `opencv-python` - Video processing
- `numpy` - Math calculations

### 2. Test It With Your Video

```bash
python3 extract_pose.py your_smash.mp4 --visualize
```

This creates:
- `your_smash_pose_data.json` - Your movement data
- `your_smash_annotated.mp4` - Video with skeleton overlay

### 3. Analyze Your Movement

```bash
python3 analyze_movement.py your_smash_pose_data.json
```

See your joint angles and get basic feedback.

---

## 📊 Building the Pro Reference Library

This is what makes the app actually useful. You need real pro player data to compare against.

### Step 1: Download Pro Videos

Find 5-10 YouTube videos of pro players doing smashes:
- Search: "Viktor Axelsen smash slow motion"
- Look for side-angle shots with one player clearly visible
- Download as MP4 (use https://yt5s.io or similar)

### Step 2: Extract Pose from Pro Videos

```bash
python3 extract_pose.py axelsen_smash.mp4 --visualize
python3 extract_pose.py momota_smash.mp4 --visualize
python3 extract_pose.py chen_smash.mp4 --visualize
# ... for each pro video
```

### Step 3: Build the Reference Standard

```bash
python3 build_pro_reference.py build axelsen_smash_pose_data.json momota_smash_pose_data.json chen_smash_pose_data.json
```

This creates `pro_reference_smash.json` - your pro baseline!

Output example:
```
BUILDING PRO REFERENCE STANDARD
Players analyzed: 5

Pro Reference Standard (Smash):

Elbow at Contact (max extension):
  Average: 168.3°
  Range: 165.2° - 172.1°
  Std Dev: ±3.4°

Knee Bend (min angle):
  Average: 97.8°
  Std Dev: ±5.2°
```

### Step 4: Compare Yourself to Pros

```bash
python3 build_pro_reference.py compare my_smash_pose_data.json
```

Output example:
```
COMPARISON TO PRO STANDARD
Your video: my_smash
Reference: Average of 5 pro players
Pro players: Viktor Axelsen, Kento Momota, Chen Long, ...

Elbow Extension at Contact:
  You: 152.7°
  Pro average: 168.3° (±3.4°)
  Difference: -15.6°

Knee Bend (power generation):
  You: 115.2°
  Pro average: 97.8° (±5.2°)
  Difference: +17.4°

ACTIONABLE FEEDBACK
1. [HIGH] Your elbow is 15.6° less extended at contact
   → Reach higher on your smash. Focus on full arm extension.

2. [HIGH] Your knees bend 17.4° less than pros
   → Bend your knees more! Power comes from your legs.
```

---

## 📁 Project Structure

```
badminton_ai/
├── requirements.txt              # Dependencies
├── extract_pose.py              # Extract skeleton from video
├── analyze_movement.py          # Calculate joint angles
├── build_pro_reference.py       # Build & compare to pro library
└── README.md                    # This file
```

---

## 🎥 Tips for Best Results

### Recording Your Videos
- **Side angle** works best (easier for pose detection)
- **Full body visible** throughout the shot
- **Good lighting** - bright, even light
- **One person only** in frame
- **Stable camera** - prop your phone up, don't handheld
- **3-10 seconds** per clip - focus on one movement

### What Videos Work Best
✅ Solo practice smashes from side angle  
✅ Shadow training footwork  
✅ Slow motion technique videos  
✅ Clean court with uncluttered background  

❌ Multi-person rallies (unless you edit to one person)  
❌ Very fast motion blur  
❌ Poor lighting / dark courts  
❌ Camera too far away  

---

## 🔧 Troubleshooting

**"No pose detected"**
- Make sure you're fully visible in frame
- Try better lighting
- Camera should be court-level, not looking down from above

**"Low detection rate"**
- Try a side-angle view instead of front/back
- Move closer to camera
- Avoid busy backgrounds

**"SSL certificate error" (Mac)**
Run: `/Applications/Python\ 3.11/Install\ Certificates.command`

**"Module not found"**
```bash
pip3 install <module_name>
```

---

## 🎯 What's Next

### Week 1 (Current): ✅ Core Foundation
- ✅ Pose extraction working
- ✅ Movement analysis
- ✅ Pro reference library system

### Week 2-3: Shot Type Detection
- Auto-detect shot type (smash vs serve vs drop)
- Build separate references for each shot
- Handle different shot types automatically

### Week 4-5: AI Feedback Layer
- Integrate Claude API for natural language coaching
- Prioritize feedback by impact
- Track improvement over time

### Week 6-7: UI
- Streamlit web interface
- Drag-and-drop video upload
- Side-by-side comparison view
- Download PDF reports

### Week 8+: Polish
- Mobile app
- Progress tracking dashboard
- Share results with coach
- Multi-person tracking

---

## 💡 The Core Insight

**Most amateur players know they're doing something wrong, but can't quantify what.**

This app fills that gap - turning "I think my smash is weaker than pros" into "Your elbow is 15° less extended at contact - here's how to fix it."

---

## 🤝 Your Friends Were Wrong

They said: "The tech is too hard to build"

The reality:
- MediaPipe (Google's pose detection): **pip install**
- Angle calculations: **High school trigonometry**
- Comparison logic: **Basic subtraction**
- No custom ML training needed
- No PhD required

You're not building the AI - Google already built it. You're just using it cleverly.

---

## 📝 License

Personal use only for now.

---

**Ready to analyze your technique? Start with Quick Start above!**
