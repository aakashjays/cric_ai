import os
import base64
import time
import tempfile
import threading
import hashlib
import cv2
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cricai-secret-2026'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max video
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── API KEYS — Load from environment ─────────────────────────────────────────
GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY', '')
CLAUDE_API_KEY  = os.getenv('CLAUDE_API_KEY', '')
# ───────────────────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini_model   = genai.GenerativeModel("gemini-1.5-flash")
claude_client  = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# ── Global Stats ────────────────────────────────────────────────────────────
stats = {
    "start_time": time.time(),
    "comments_generated": 0,
    "connected_clients": 0
}
recent_commentaries = []  # Track last 10 lines to avoid repetitionrecent_descriptions = []  # Track descriptions to detect duplicate frames
last_frame_hash = None  # Detect if we're looking at the same static frame
description_commentaries = {}  # Cache: description -> list of varied commentaries
# ── Prompts ─────────────────────────────────────────────────────────────────
GEMINI_VISION_PROMPT = """Look at this cricket match image carefully.
Describe in 2-3 sentences ONLY what you see:
- What action is happening (batting, bowling, fielding, celebration, crowd)?
- Player positions, body language, shot type if visible
- Scoreboard details if visible
- Overall match atmosphere
Be specific and factual. No commentary — just describe what you see."""

CLAUDE_HINGLISH_PROMPT = """You are CricAI — India's most electric live cricket commentator.
You speak in Hinglish — a natural mix of Hindi and English like real Indian cricket fans talk.

Style examples:
"Aur yeh toh ZABARDAST shot hai bhai! Ball sidha boundary ke paar gaya — koi roko isko!"
"Arre yaar, kya yorker daala Bumrah ne — batsman ke paas bilkul koi jawab nahi tha!"  
"SIX! Seedha stands mein gaya — crowd pagal ho gaya hai! Kya tha yeh shot!"
"Dot ball. Tight over chal raha hai — pressure build ho raha hai dono sides pe!"
"OH! Caught behind! Yeh toh sab ne expect nahi kiya tha — tremendous delivery!"

Rules:
- ONE commentary line only (2-3 sentences max)
- Mix Hindi and English naturally — not forced
- Use cricket terms: yorker, cover drive, mid-wicket, googly, powerplay, death overs
- Match the ENERGY of the moment — explosive for sixes, tense for dot balls
- NEVER repeat same line twice
- Start directly with commentary — no preamble like "Here is:"
- Sound like a real Indian TV commentator"""


def get_gemini_description(image_bytes):
    """Step 1: Gemini sees the image and describes it"""
    try:
        response = gemini_model.generate_content([
            GEMINI_VISION_PROMPT,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        return f"Cricket match in progress with players on the field"


def get_claude_commentary(description):
    """Step 2: Claude turns description into Hinglish commentary - generates 3 variations"""
    global recent_commentaries, description_commentaries
    
    # Check if we have cached varied commentaries for this description
    if description in description_commentaries:
        cache = description_commentaries[description]
        # Pick the next one in rotation that isn't in recent list
        for comment in cache:
            if comment not in recent_commentaries[-10:]:
                recent_commentaries.append(comment)
                if len(recent_commentaries) > 20:
                    recent_commentaries.pop(0)
                return comment
        # If all are recent, just return a new one anyway
        return cache[0]
    
    try:
        recent_text = "\n".join([f"- {c}" for c in recent_commentaries[-8:]]) if recent_commentaries else "None yet"
        
        message = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=CLAUDE_HINGLISH_PROMPT,
            messages=[
                {"role": "user", "content": f"""Cricket scene: {description}

Generate 3 different UNIQUE Hinglish commentary lines. Make them ALL different in tone, energy, and phrasing.

NEVER use ANY of these lines:
{recent_text}

Format as:
1. [first variation]
2. [second variation]
3. [third variation]"""}
            ]
        )
        
        response_text = message.content[0].text.strip()
        commentaries = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and not line[0].isdigit():
                commentaries.append(line)
        
        if not commentaries:
            commentaries = [response_text]
        
        # Cache these variations
        description_commentaries[description] = commentaries[:3]
        
        # Return first one and track it
        commentary = commentaries[0]
        recent_commentaries.append(commentary)
        if len(recent_commentaries) > 20:
            recent_commentaries.pop(0)
        
        return commentary
    
    except Exception as e:
        print(f"Claude error: {e}")
        # Fallback
        recent_commentaries.append("Aur khel jaari hai!")
        return "Aur khel jaari hai!"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "comments_generated": stats["comments_generated"],
        "uptime_seconds": int(time.time() - stats["start_time"]),
        "connected_clients": stats["connected_clients"]
    })


@app.route('/commentary', methods=['POST'])
def get_commentary():
    """Live camera commentary endpoint"""
    global last_frame_hash
    data = request.json
    image_data = data.get('image', '')

    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Hash frame to detect if it's the same static image
        frame_hash = hashlib.md5(image_bytes).hexdigest()
        if frame_hash == last_frame_hash:
            # Same frame - skip for now to avoid unnecessary API calls
            return jsonify({'commentary': '', 'description': 'Same frame'}), 200
        
        last_frame_hash = frame_hash
        
        description  = get_gemini_description(image_bytes)
        commentary   = get_claude_commentary(description)

        stats["comments_generated"] += 1

        socketio.emit('new_commentary', {
            'text': commentary,
            'description': description,
            'count': stats["comments_generated"]
        })

        return jsonify({'commentary': commentary, 'description': description})

    except Exception as e:
        print(f"Commentary error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Video upload → frame extraction → commentary pipeline"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400

        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        allowed = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
        ext = video_file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed:
            return jsonify({'error': f'Unsupported format. Use: {", ".join(allowed)}'}), 400

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False)
        video_file.save(tmp.name)
        tmp.close()
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 400

    def process_video(filepath):
        try:
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_secs = int(total_frames / fps)
            interval_frames = int(fps * 2)  # 1 frame every 2 seconds

            # Count how many frames we'll process
            frame_positions = list(range(0, total_frames, interval_frames))
            total_to_process = len(frame_positions)

            socketio.emit('video_start', {
                'total_frames': total_to_process,
                'duration': duration_secs
            })

            results = []
            processed = 0

            for frame_num in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                timestamp_sec = int(frame_num / fps)
                mins = timestamp_sec // 60
                secs = timestamp_sec % 60
                timestamp_str = f"{mins}:{secs:02d}"

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                image_bytes = buffer.tobytes()

                socketio.emit('video_progress', {
                    'current': processed + 1,
                    'total': total_to_process,
                    'timestamp': timestamp_str
                })

                description = get_gemini_description(image_bytes)
                commentary  = get_claude_commentary(description)
                stats["comments_generated"] += 1

                result = {
                    'timestamp': timestamp_str,
                    'commentary': commentary,
                    'description': description
                }
                results.append(result)

                socketio.emit('video_commentary', result)
                processed += 1
                time.sleep(0.2)  # small delay to not overwhelm API

            cap.release()
            os.unlink(filepath)

            socketio.emit('video_complete', {
                'total': len(results),
                'commentaries': results
            })

        except Exception as e:
            socketio.emit('video_error', {'error': str(e)})
            try:
                os.unlink(filepath)
            except:
                pass

    # Run in background thread so response returns immediately
    thread = threading.Thread(target=process_video, args=(tmp.name,))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'processing', 'message': 'Video upload received, processing started'})


# ── Socket Events ────────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    stats["connected_clients"] += 1
    emit('status', {'message': 'Connected to CricAI', 'clients': stats["connected_clients"]})
    socketio.emit('client_count', {'count': stats["connected_clients"]})


@socketio.on('disconnect')
def handle_disconnect():
    stats["connected_clients"] = max(0, stats["connected_clients"] - 1)
    socketio.emit('client_count', {'count': stats["connected_clients"]})


if __name__ == '__main__':
    print("=" * 55)
    print("  🏏 CricAI — Live Cricket Commentary Server")
    print("  Open:  http://localhost:5000")
    print("  Share: http://<your-local-ip>:5000")
    print("=" * 55)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
