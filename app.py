import os
import json
import base64
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rapidfuzz import fuzz  # Added for fuzzy matching
import re
import subprocess
from tempfile import NamedTemporaryFile
import tempfile
import hashlib
from datetime import datetime, timedelta
from threading import Thread
import functools
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Environment variables
QWEN_API_KEY = os.environ.get('QWEN_API_KEY')
SERP_API_KEY = os.environ.get('SERP_API_KEY')
VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')
WHATSAPP_TOKEN = os.environ.get('WHATSAPP_TOKEN')
PHONE_NUMBER_ID = os.environ.get('PHONE_NUMBER_ID')
LOG_RECIPIENT = os.environ.get('LOG_RECIPIENT')
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Custom WhatsApp log handler
class WhatsAppLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            # Only send to WhatsApp if LOG_RECIPIENT is set and not empty
            if LOG_RECIPIENT and "send_whatsapp_message" in globals():
                send_whatsapp_message(LOG_RECIPIENT, f"[LOG] {log_entry}")
        except Exception as e:
            # Fallback to file logging if WhatsApp fails
            logging.getLogger(__name__).error(f"Failed to send log to WhatsApp: {e}")

# Configure logging to a file and WhatsApp
logging.basicConfig(
    filename='messages.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Add WhatsApp log handler for WARNING and above
wa_handler = WhatsAppLogHandler()
wa_handler.setLevel(logging.WARNING)
wa_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(wa_handler)

app = Flask(__name__)

_scraped_data_cache = None
_processed_messages = {}  # Store processed message IDs
_CLEANUP_INTERVAL = timedelta(hours=1)  # Clean up old messages every hour
_executor = ThreadPoolExecutor(max_workers=5)

try:
    with open("d2place_data.json", "r", encoding="utf-8") as f:
        CACHED_DATA = json.load(f)
    logger.info("Loaded scraped data into CACHED_DATA")
except Exception as e:
    logger.error("Failed to load d2place_data.json into cache: %s", e)
    CACHED_DATA = {}


def async_worker(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper

#########################
# 1. Search & Scraping
#########################



def perform_web_search(query):
    """
    Uses SerpAPI to get top results for 'query' and return a short summary.
    """
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "num": "5",
        "api_key": SERP_API_KEY
    }
    try:
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        summary_lines = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "")
                summary_lines.append(f"{title}: {snippet}")
        return "\n".join(summary_lines)
    except Exception as e:
        logger.error("Error in perform_web_search: %s", e)
        return "Web search unavailable."

def query_json_llm(user_query: str, json_data: dict) -> str:
    """
    Ask Qwen to find exactly what the user wants in the given JSON,
    in *any* language, without us hard-coding any keywords.
    """
    system_prompt = """
    You are a JSON-query assistant for **D2 Place**.  I will give you a JSON object with these keys:
    • mall_info      (address, hours, transport, etc)  
    • dining         (a list of restaurants with name, location, opening_hours)  
    • shopping       (list of shops)  
    • play           (list of play facilities)  
    • events         (list of events)

    The user will ask in ANY language—sometimes just one word like “晚餐” or “玩”, sometimes a full sentence.  
    Your job is two-fold:
    1. **Interpret** what they want (e.g. 晚餐 → “dinner restaurants”; 购物 → “shops”; 玩 → “play facilities”; 
        or hours of a specific place; or address of the mall; etc.)
    2. **Extract** exactly that information from the JSON and return it.

    **You have only two valid replies**:
    - If the JSON contains the answer: output **just** the information, either a raw string or a very short list.  
    - Otherwise: output exactly the single word:
        ```
        UNKNOWN
        ```
    —no apologies, no extra text.

    **Few-shot examples**  
    ```json
    JSON:
    {"mall_info":{
    "location":"9 Cheung Yee Street, Lai Chi Kok, Kowloon, Hong Kong"
    }}
    Q: Where is D2 Place ONE?
    A: 9 Cheung Yee Street, Lai Chi Kok, Kowloon, Hong Kong

    Q: Where is D2 Place?
    A: 9 Cheung Yee Street, Lai Chi Kok, Kowloon, Hong Kong; 15 Cheung Shun Street, Lai Chi Kok, Kowloon, Hong Kong

    JSON:
    {"dining":[
    {"name":"La Trattoria","opening_hours":"11:00–22:00"},
    {"name":"Sushi House","opening_hours":"12:00–15:00; 18:00–22:00"}
    ]}
    Q: 午餐
    A: La Trattoria (11:00–15:00), Sushi House (12:00–15:00)

    Q: 晚餐
    A: La Trattoria (18:00–22:00), Sushi House (18:00–22:00)

    JSON:
    {"shopping":[{"name":"Book World"},{"name":"Gadget Zone"}]}
    Q: 购物
    A: Book World; Gadget Zone

    JSON:
    {"play":[{"name":"Game Station"},{"name":"VR Arena"}]}
    Q: 玩
    A: Game Station; VR Arena
    """
    # Serialize your full D2 Place JSON (or sub-sections you want)  
    json_text = json.dumps(json_data, ensure_ascii=False)
    user_content = f"JSON:\n{json_text}\n\nQuestion: {user_query}"
    payload = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }
    return call_qwen_api(payload)


def retrieve_relevant_data(query):
    """
    Load the scraped D2 Place data from d2place_data.json and return
    a targeted summary based on the query using fuzzy matching.
    """
    data = CACHED_DATA
    if not data:
        logger.warning("CACHED_DATA is empty; no scraped data available.")
        return ""

    query_lower = query.lower()
    threshold = 20  # fuzzy-match threshold
    summary_parts = []

    # Check for opening hours queries
    if any(keyword in query_lower for keyword in ["opening hours", "hours", "time", "when open", "business hours"]):
        # First check for specific venue
        venue_name = None
        for word in query_lower.split():
            if word not in ["what", "are", "the", "opening", "hours", "for", "when", "is", "open", "time"]:
                venue_name = word
                break

        if venue_name:
            # Search in all categories
            for category in ["dining", "shopping", "play"]:
                for venue in data.get(category, []):
                    if fuzz.partial_ratio(venue_name, venue.get("name", "").lower()) >= threshold:
                        hours = venue.get("opening_hours", "Opening hours not available")
                        return f"Opening hours for {venue['name']}:\n{hours}"

        # If no specific venue found or no venue specified, return mall hours
        mall_info = data.get("mall_info", {})
        if mall_info.get("about"):
            return f"General mall hours:\n{mall_info['about']}"
        return "Opening hours information is not available in our database. Please check the D2 Place website for the most up-to-date information."

    # Regular query handling
    def shop_matches_query(shop):
        combined = (
            shop.get("name","") + " "
            + shop.get("location","") + " "
            + shop.get("google_review_snippet","") + " "
            + shop.get("description","")
        )
        score = fuzz.partial_ratio(query_lower, combined.lower())
        return score >= threshold

    # Dining
    matched_dining = []
    for shop in data.get("dining", []):
        if shop_matches_query(shop):
            matched_dining.append(shop)
    if matched_dining:
        s = "🍴 Dining options:\n"
        for r in matched_dining[:5]:
            hours = r.get("opening_hours", "Hours not available")
            s += f"- {r['name']} (at {r['location']})\n  Hours: {hours}\n"
        summary_parts.append(s)

    # Shopping
    matched_shops = []
    for shop in data.get("shopping", []):
        if shop_matches_query(shop):
            matched_shops.append(shop)
    if matched_shops:
        s = "🛍️ Shops:\n"
        for r in matched_shops[:5]:
            hours = r.get("opening_hours", "Hours not available")
            s += f"- {r['name']} (at {r['location']})\n  Hours: {hours}\n"
        summary_parts.append(s)

    # Play facilities
    matched_play = []
    for facility in data.get("play", []):
        if shop_matches_query(facility):
            matched_play.append(facility)
    if matched_play:
        s = "🎮 Play facilities:\n"
        for r in matched_play[:5]:
            hours = r.get("opening_hours", "Hours not available")
            s += f"- {r['name']} (at {r['location']})\n  Hours: {hours}\n"
        summary_parts.append(s)

    # Events
    matched_events = []
    for ev in data.get("events", []):
        combined = f"{ev.get('name','')} {ev.get('description','')}"
        if fuzz.partial_ratio(query_lower, combined.lower()) >= threshold:
            matched_events.append(ev)
    if matched_events:
        s = "🎉 Events:\n"
        for ev in matched_events[:3]:
            s += f"- {ev['name']} (on {ev.get('date','')})\n"
        summary_parts.append(s)

    # Final fallback
    if not summary_parts:
        return None

    return "\n\n".join(summary_parts)

def extract_meal_query(text: str) -> str | None:
    """Return 'breakfast', 'lunch' or 'dinner' if text looks like a meal query."""
    lowered = text.lower()
    if any(k in lowered for k in ["dinner", "晚餐", "晚飯"]):
        return "dinner"
    if any(k in lowered for k in ["lunch", "午餐", "午飯"]):
        return "lunch"
    if any(k in lowered for k in ["breakfast", "早餐"]):
        return "breakfast"
    return None

def restaurants_open_for(meal: str) -> str:
    """Return a formatted list of restaurants open during the given meal."""
    meal_hours = {
        "breakfast": (7, 11),
        "lunch": (11, 15),
        "dinner": (17, 22),
    }
    start_h, end_h = meal_hours.get(meal, (None, None))
    if start_h is None:
        return ""

    results = []
    for r in CACHED_DATA.get("dining", []):
        hours = r.get("opening_hours", "")
        for part in hours.split(";"):
            m = re.search(r"(\d{1,2}):(\d{2}).*(\d{1,2}):(\d{2})", part)
            if not m:
                continue
            sh = int(m.group(1))
            eh = int(m.group(3))
            if sh <= start_h and eh >= end_h:
                results.append(f"- {r['name']} ({hours})")
                break

    if not results:
        return ""
    header = {
        "breakfast": "🍳 Breakfast options:",
        "lunch": "🍽 Lunch options:",
        "dinner": "🍴 Dinner options:",
    }[meal]
    return header + "\n" + "\n".join(results[:5])


def is_smalltalk(text: str) -> bool:
    """Return True if the text looks like a greeting or other small talk."""
    t = text.strip().lower()
    greetings = {
        "hi", "hello", "hey", "你好", "您好", "嗨",
        "good morning", "good afternoon", "good evening", "早安", "早上好",
    }
    farewells = {"bye", "goodbye", "再見", "bye bye"}
    thanks = {"thanks", "thank you", "謝謝", "多謝"}
    return t in greetings or t in farewells or t in thanks


def should_call_web_search(query: str, scraped: str) -> bool:
    """Decide whether to call SerpAPI for this query."""
    if is_smalltalk(query):
        return False

    cleaned = scraped.strip().lower()
    if not cleaned or cleaned == "unknown" or cleaned.startswith("i'm sorry") or cleaned.startswith("sorry"):
        return True
    return False

#########################
# 2. Qwen Handlers
#########################

def call_qwen_api(payload):
    """
    Common function to call the Qwen endpoint with the given payload.
    """
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/chat/completions"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        content = result['choices'][0]['message']['content']
        if isinstance(content, str):
            return content.strip()
        else:
            return json.dumps(content)
    except Exception as e:
        logger.error("Qwen API error: %s", e)
        return "Sorry, I'm having trouble generating an answer right now."

def handle_text_query(user_text):
    system_prompt = (
       """
        You are a data-driven assistant for D2 Place mall in Hong Kong, answer any questions the user has about D2 Place.
        Do NOT give out any information that is not related to D2 Place.
        ONLY use the information under “SCRAPED DATA” or “WEB SEARCH RESULTS” below.
        Do NOT invent, embellish, or guess anything outside those sources.
        If the answer is not fully contained in those two inputs, reply: “I’m sorry, I don’t know.”
        Respond in the same language as the user.
        """
    )


    meal = extract_meal_query(user_text)
    if meal:
        reply = restaurants_open_for(meal)
        if reply:
            return reply

    if is_smalltalk(user_text):
        return "Hello! How can I assist you with information about D2 Place?"

    # 1) Always pull from cache / fuzzy logic
    scraped_data = query_json_llm(user_text, CACHED_DATA)

    # 2) Only call SerpAPI if scraped_data is empty or a "general fallback"
    if should_call_web_search(user_text, scraped_data):
        web_results = perform_web_search(user_text)
    else:
        web_results = ""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"User Query: {user_text}\n\n"
        f"Scraped Data:\n{scraped_data}\n\n"
        f"Web Search Results:\n{web_results}\n\n"
    )
    payload = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }
    return call_qwen_api(payload)

def handle_image_query(image_b64, caption=""):
    """
    Calls Qwen-VL-Max with the image and caption, using scraping and web search on the caption.
    """
    scraped_data = retrieve_relevant_data(caption)
    web_results = perform_web_search(caption)
    system_prompt = (
        "You are Qwen-VL. You received an image with accompanying text context. "
        "Use the provided scraped data and web search results to answer the query."
    )
    text_context = (
        f"Caption: {caption}\n\n"
        f"Scraped Data:\n{scraped_data}\n\n"
        f"Web Search Results:\n{web_results}\n\n"
    )
    user_content = [
        {"type": "image", "image": image_b64},
        {"type": "text", "text": text_context}
    ]
    payload = {
        "model": "qwen-vl-max",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    }
    return call_qwen_api(payload)

def convert_ogg_to_mp3(ogg_bytes: bytes) -> bytes:
    """
    Convert raw OGG/Opus bytes to MP3 bytes via ffmpeg.
    Requires 'ffmpeg' on PATH.
    """
    # write OGG to temp file
    with NamedTemporaryFile(suffix=".ogg", delete=False) as in_f:
        in_f.write(ogg_bytes)
        in_path = in_f.name

    mp3_path = in_path.replace(".ogg", ".mp3")
    # run ffmpeg to convert
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", in_path,
            "-ac", "1",           # mono
            "-ar", "16000",       # 16kHz sample rate
            mp3_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.error("ffmpeg not found while converting audio")
        raise
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg conversion failed: %s", e)
        raise

    # read back mp3 bytes
    with open(mp3_path, "rb") as mp3_f:
        mp3_bytes = mp3_f.read()

    return mp3_bytes

def transcode_to_mp3(raw_bytes: bytes, in_format: str) -> bytes:
    """
    Uses ffmpeg to convert raw audio (e.g. OGG Opus) into MP3 bytes.
    """
    input_path = None
    output_path = None
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix="."+in_format, delete=False) as inp:
            input_path = inp.name
            inp.write(raw_bytes)
            inp.flush()

        # Create temporary output file
        output_path = input_path + ".mp3"
        
        # Run ffmpeg conversion
        logger.info(f"Running ffmpeg conversion from {input_path} to {output_path}")
        try:
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", input_path,
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "libmp3lame",
                output_path
            ], capture_output=True, text=True)
        except FileNotFoundError:
            logger.error("ffmpeg not found while transcoding audio")
            raise

        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            raise Exception(f"ffmpeg conversion failed: {result.stderr}")
            
        # Read the converted file
        with open(output_path, "rb") as outf:
            mp3_bytes = outf.read()
            
        logger.info(f"Successfully converted audio to MP3. Size: {len(mp3_bytes)} bytes")
        return mp3_bytes
        
    except Exception as e:
        logger.error(f"Error in transcode_to_mp3: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        try:
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

def transcribe_audio(audio_bytes: bytes, content_type: str) -> str:
    """Transcribe audio bytes to text using Qwen's transcription API."""
    logger.info(
        "Starting audio transcription. Content type: %s, Audio size: %d bytes",
        content_type,
        len(audio_bytes),
    )
    
    # Save incoming audio for debugging
    try:
        with open("debug_incoming_audio.bin", "wb") as f:
            f.write(audio_bytes)
        logger.info("Saved incoming audio to debug_incoming_audio.bin")
    except Exception as e:
        logger.error(f"Failed to save incoming audio: {e}")
    
    # if content_type contains "ogg", convert first
    if "ogg" in content_type:
        try:
            logger.info("Converting OGG to MP3...")
            audio_bytes = transcode_to_mp3(audio_bytes, "ogg")
            logger.info(f"Conversion complete. New audio size: {len(audio_bytes)} bytes")
            with open("debug_converted_audio.mp3", "wb") as f:
                f.write(audio_bytes)
            logger.info("Saved converted audio to debug_converted_audio.mp3")
        except Exception as e:
            logger.exception(f"Failed to ffmpeg‐transcode OGG→MP3: {str(e)}")
            return ""

    url = f"{BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}"}
    files = {"file": ("voice.mp3", audio_bytes, "audio/mpeg")}
    data = {"model": "qwen2-audio-instruct"}

    try:
        logger.info("Sending request to Qwen transcription API...")
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        logger.info("Qwen transcription status: %s", resp.status_code)

        if resp.status_code == 404:
            logger.warning("Transcription endpoint returned 404, falling back to chat completions")
            prompt_payload = {
                "model": "qwen2-audio-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": {"data": base64.b64encode(audio_bytes).decode("utf-8"), "format": "mp3"},
                            }
                        ],
                    }
                ],
            }
            return call_qwen_api(prompt_payload)

        if resp.status_code != 200:
            logger.error("❌ transcribe_audio got %s: %s", resp.status_code, resp.text)
            with open("debug_failed_audio.mp3", "wb") as f:
                f.write(audio_bytes)
            logger.info("Saved failed audio to debug_failed_audio.mp3")
            return ""

        resp_json = resp.json()
        transcript = resp_json.get("text", "")
        logger.info("Transcription result: %s", transcript)
        return transcript
    except Exception as e:
        logger.exception("🔥 Exception in transcribe_audio: %s", e)
        try:
            with open("debug_failed_audio.mp3", "wb") as f:
                f.write(audio_bytes)
            logger.info("Saved failed audio to debug_failed_audio.mp3 (exception)")
        except Exception as e2:
            logger.error("Failed to save failed audio: %s", e2)
        return ""

def handle_audio_query(audio_bytes, caption=""):
    """Transcribe the audio then pass the transcript to ``handle_text_query``."""
    transcript = transcribe_audio(audio_bytes, "audio/mpeg")
    logger.info("Audio transcribed to: %s", transcript)

    if not transcript or transcript.lower().startswith("sorry"):
        return "抱歉，無法轉錄你的語音訊息。請稍後再試。"

    # --- Step 2: treat that transcript as a normal text query ---
    return handle_text_query(transcript)

#########################
# 3. Logging & Log Forwarding
#########################

def forward_logs_via_whatsapp(recipient):
    """
    Reads the local log file 'messages.log' and sends its content to a specific WhatsApp number.
    """
    try:
        with open('messages.log', 'r', encoding='utf-8') as f:
            log_content = f.read()
        send_whatsapp_message(recipient, log_content)
    except Exception as e:
        logger.error("Error forwarding logs: %s", e)

#########################
# 4. WhatsApp Webhook
#########################

@app.route('/webhook', methods=['GET'])
def verify():
    """
    GET endpoint for WhatsApp verification.
    """
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            logger.info("Webhook verified successfully.")
            return challenge, 200
        else:
            return "Verification token mismatch", 403
    return "Bad Request", 400

def cleanup_old_messages():
    """Clean up old processed message IDs."""
    now = datetime.now()
    global _processed_messages
    _processed_messages = {
        msg_id: timestamp 
        for msg_id, timestamp in _processed_messages.items() 
        if now - timestamp < _CLEANUP_INTERVAL
    }

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    
    # 1) WhatsApp sometimes sends 'statuses' updates or duplicates;
    #    we only want real user messages under data['entry'][…]['changes'][…]['value']['messages']

    # 1) WhatsApp sometimes sends 'statuses' updates or duplicates;
    #    we only want real user messages under data['entry'][…]['changes'][…]['value']['messages']
    entry    = data.get('entry',    [{}])[0]
    change   = entry.get('changes', [{}])[0]
    value    = change.get('value',   {})

    # If it's only a status update, ignore it
    if 'statuses' in value and not value.get('messages'):
        return jsonify(status="ignored_status"), 200

    messages = value.get('messages', [])
    if not messages:
        return jsonify(status="no_messages"), 200

    for m in messages:
        _executor.submit(process_message, m)

    return jsonify(status="processing", count=len(messages)), 200

def process_message(msg):
    """Handle a single WhatsApp message in a background thread."""
    msg_id = msg.get('id')
    if msg_id:
        if msg_id in _processed_messages:
            logger.info("Duplicate message %s ignored", msg_id)
            return
        _processed_messages[msg_id] = datetime.now()
        cleanup_old_messages()
    from_user = msg.get('from', '')
    msg_type = msg.get('type', '')

    BOT_NUMBER = PHONE_NUMBER_ID  # your "from" WhatsApp number
    LOG_NUMBER = LOG_RECIPIENT    # where you send logs
    if from_user in (BOT_NUMBER, LOG_NUMBER):
        return

    try:
        if msg_type == 'text':
            inbound_text = msg['text']['body']
        elif msg_type == 'image':
            inbound_text = f"<image id:{msg['image']['id']}> caption:{msg['image'].get('caption','')}"
        elif msg_type == 'audio':
            media_id = msg['audio']['id']
            mime = msg['audio'].get('mime_type', 'audio/unknown')
            inbound_text = f"<audio id:{media_id} mime:{mime}>"
        else:
            inbound_text = f"<{msg_type}>"

        send_whatsapp_message(LOG_RECIPIENT, f"📥 From {from_user} ({msg_type}): {inbound_text}")

        if msg_type == 'text':
            bot_reply = handle_text_query(inbound_text)
        elif msg_type == 'audio':
            audio_bytes, content_type = download_media_file(msg['audio']['id'])
            transcript = transcribe_audio(audio_bytes, content_type)
            if not transcript:
                bot_reply = "抱歉，我無法轉錄你的語音訊息。請稍後再試。"
            else:
                bot_reply = handle_text_query(transcript)
        else:
            bot_reply = "Sorry, I only handle text and audio messages for now."

        send_whatsapp_message(LOG_RECIPIENT, f"📤 To   {from_user}: {bot_reply}")
        send_whatsapp_message(from_user, bot_reply)

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")

def download_media_file(media_id):
    """
    Retrieves the media file from WhatsApp Cloud API and returns raw bytes and content type.
    """
    # 1) get media URL
    info_url = f"https://graph.facebook.com/v16.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    info_resp = requests.get(info_url, headers=headers, timeout=10)
    info_resp.raise_for_status()
    media_url = info_resp.json().get("url")
    mime = info_resp.json().get("mime_type", "")

    # 2) download raw media
    file_resp = requests.get(media_url, headers=headers, timeout=10)
    file_resp.raise_for_status()
    raw_bytes = file_resp.content

    # 3) if it's OGG/Opus, convert to MP3
    if mime.startswith("audio/ogg"):
        try:
            mp3_bytes = convert_ogg_to_mp3(raw_bytes)
            return mp3_bytes, "audio/mpeg"
        except Exception as e:
            logger.error("Audio conversion failed: %s", e)
            # fall back to raw bytes if conversion fails
            return raw_bytes, mime

    # otherwise, return raw bytes
    return raw_bytes, mime

def send_whatsapp_message(recipient, message_text):
    """
    Sends a text message back to the user via WhatsApp Cloud API.
    """
    url = f"https://graph.facebook.com/v16.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "text": {"body": message_text}
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        logger.info("WhatsApp send response: %s", resp.text)
    except Exception as e:
        logger.error("Error sending WhatsApp message: %s", e)

#########################
# 5. Run the Flask App
#########################

if __name__ == '__main__':
    print("Starting the Flask app. Callback URL is not shown here because it's local.")
    print("To get a public URL, run: ngrok http 4040 and use the domain + /webhook as callback.")
    app.run(port=4040, debug=True)
