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
from threading import Thread, Lock
import functools
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup

PHONE_REGEX = re.compile(r"(?:\+?852[-\s]?)?\d{4}[-\s]?\d{4}")

def remove_phone_numbers(text: str) -> str:
    """Return ``text`` with Hong Kong style phone numbers removed."""
    return PHONE_REGEX.sub("", text)

def strip_phone_numbers(obj):
    """Recursively remove phone numbers from all string values in ``obj``."""
    if isinstance(obj, dict):
        return {k: strip_phone_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_phone_numbers(x) for x in obj]
    if isinstance(obj, str):
        return remove_phone_numbers(obj)
    return obj

load_dotenv()

# Environment variables
QWEN_API_KEY = os.environ.get('QWEN_API_KEY')
SERP_API_KEY = os.environ.get('SERP_API_KEY')
VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')
WHATSAPP_TOKEN = os.environ.get('WHATSAPP_TOKEN')
PHONE_NUMBER_ID = os.environ.get('PHONE_NUMBER_ID')
LOG_RECIPIENT = os.environ.get('LOG_RECIPIENT')
PUBLIC_URL = os.environ.get('PUBLIC_URL', 'http://chatbot.d2place.com')
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# Use a fixed model so the chatbot always calls the same Qwen version
QWEN_MODEL = "qwen-plus"

# Small-talk phrases used to mimic Festival Walk style responses
GREETINGS = {
    "hi", "hello", "hey", "你好", "您好", "嗨",
    "good morning", "good afternoon", "good evening", "早安", "早上好",
}
FAREWELLS = {"bye", "goodbye", "再見", "bye bye"}
THANKS = {"thanks", "thank you", "謝謝", "多謝"}

# Fuzzy matching helper utilities
FUZZY_THRESHOLD = 80

def fuzzy_match(text: str, phrases: list[str] | set[str], threshold: int = FUZZY_THRESHOLD) -> bool:
    """Return True if ``text`` fuzzily matches any phrase in ``phrases``."""
    lowered = text.lower()
    for p in phrases:
        pl = p.lower()
        if pl in lowered or fuzz.partial_ratio(lowered, pl) >= threshold:
            return True
    return False

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

app = Flask(__name__, static_url_path='/assets', static_folder='Assets')

_scraped_data_cache = None
_processed_messages = {}  # Store processed message IDs
_messages_lock = Lock()  # synchronize access to processed messages
_CLEANUP_INTERVAL = timedelta(hours=1)  # Clean up old messages every hour
# Allow more concurrent message processing to reduce queue delays
_executor = ThreadPoolExecutor(max_workers=20)

try:
    with open("d2place_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    CACHED_DATA = strip_phone_numbers(raw_data)
    logger.info("Loaded scraped data into CACHED_DATA")
except Exception as e:
    logger.error("Failed to load d2place_data.json into cache: %s", e)
    CACHED_DATA = {}


# Pre-serialize the full JSON once for LLM context
FULL_JSON_TEXT = json.dumps(CACHED_DATA, ensure_ascii=False)


def async_worker(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper

#########################
# 1. Search & Scraping
#########################



@lru_cache(maxsize=32)
def perform_web_search(query):
    """
    Uses SerpAPI to get top results for 'query' and return a short summary.
    """
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query + " d2 place hong kong",
        "num": "3",
        "gl": "hk",
        "api_key": SERP_API_KEY
    }
    try:
        start = time.monotonic()
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        logger.info("Web search request took %.2f seconds", time.monotonic() - start)
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


def fetch_first_image(url: str) -> str | None:
    """Return the first image URL found on ``url`` (e.g. og:image or first <img>)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        meta = soup.find("meta", property="og:image")
        if meta and meta.get("content"):
            return urljoin(resp.url, meta["content"])
        img = soup.find("img")
        if img and img.get("src"):
            return urljoin(resp.url, img["src"])
    except Exception as e:
        logger.error("Error fetching image from %s: %s", url, e)
    return None


def search_social_media_links(query):
    """Return (links, image_url) for social media related to ``query`` using SerpAPI."""
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{query} d2 place hong kong facebook instagram openrice",
        "gl": "hk",
        "num": "3",
        "api_key": SERP_API_KEY,
    }
    links = []
    image_url = None
    try:
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        for r in results.get("organic_results", []):
            link = r.get("link", "")
            if any(d in link for d in ["facebook.com", "instagram.com", "openrice.com"]):
                links.append(link)
                if not image_url:
                    image_url = r.get("thumbnail") or fetch_first_image(link)
                if len(links) >= 3:
                    break
            elif not image_url:
                image_url = r.get("thumbnail") or fetch_first_image(link)
    except Exception as e:
        logger.error("Error searching social media links: %s", e)
    return links, image_url


def search_instagram_openrice(query: str) -> tuple[str, str]:
    """Return Instagram and OpenRice links for ``query`` using SerpAPI."""
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{query} d2 place hong kong instagram openrice",
        "gl": "hk",
        "num": "2",
        "api_key": SERP_API_KEY,
    }
    ig = ""
    or_link = ""
    try:
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        for r in results.get("organic_results", []):
            link = r.get("link", "")
            if not ig and "instagram.com" in link:
                ig = link
            if not or_link and "openrice.com" in link:
                or_link = link
            if ig and or_link:
                break
    except Exception as e:
        logger.error("Error searching Instagram/OpenRice: %s", e)
    return ig, or_link


def extract_building(venue: dict) -> str:
    """Return 'D2 Place ONE (D2 1)' or 'D2 Place TWO (D2 2)' if found in any venue fields."""
    pieces = [
        venue.get("google_review_data", {}).get("address", ""),
        venue.get("extra_google_info", {}).get("top_snippet", ""),
        venue.get("location", ""),
    ]
    text = " ".join(p for p in pieces if p)

    # Check street names first as some addresses omit the building name
    if re.search(r"Cheung\s*Yee|長義街", text, re.IGNORECASE):
        return "D2 Place ONE (D2 1)"
    if re.search(r"Cheung\s*Shun|長順街", text, re.IGNORECASE):
        return "D2 Place TWO (D2 2)"

    # Look for explicit references to ONE/TWO or Phase 1/2
    patterns = [
        r"D2\s*Place[^A-Za-z0-9]*(ONE|TWO)",
        r"D2\s*Place[^A-Za-z0-9]*(1|2)\s*期",
        r"D2廣場\s*(一期|二期)",
        r"D2\s*Place[^A-Za-z0-9]*(1|2)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).upper()
            if val in ["1", "一期", "ONE"]:
                return "D2 Place ONE (D2 1)"
            if val in ["2", "二期", "TWO"]:
                return "D2 Place TWO (D2 2)"
    return ""


def extract_openrice_link(venue: dict) -> str:
    """Return an OpenRice URL found in ``venue`` if any."""
    candidates = [
        venue.get("website"),
        venue.get("google_review_data", {}).get("website"),
        venue.get("extra_google_info", {}).get("kg_website"),
    ]
    for c in candidates:
        if c and "openrice.com" in c:
            return c
    snippet = venue.get("extra_google_info", {}).get("top_snippet", "")
    m = re.search(r"https?://[^\s,]*openrice\.com[^\s,]*", snippet)
    if m:
        return m.group(0)
    return ""


def format_venue_details(venue: dict) -> str:
    """Return name, address, hours and D2 Place link for the venue."""

    name = venue.get("name") or venue.get("title") or "Unknown"

    address = (
        venue.get("location")
        or venue.get("google_review_data", {}).get("address")
        or venue.get("venue")
        or ""
    )
    building = extract_building(venue)
    if address and building and building.lower() not in address.lower():
        address = f"{address}, {building}"
    elif not address:
        address = building

    hours = (
        venue.get("opening_hours")
        or venue.get("date")
        or ""
    )

    d2_url = venue.get("detail_url", "")

    lines = [name]
    if address:
        lines.append(f"Address: {address}")
    if hours:
        lines.append(f"Business hour: {hours}")
    if d2_url:
        lines.append(f"D2 Place Page: {d2_url}")

    return "\n".join(lines)



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
    threshold = 5  # fuzzy-match threshold
    summary_parts = []

    # Check for opening hours queries
    if fuzzy_match(query_lower, ["opening hours", "hours", "time", "when open", "business hours"]):
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
            s += format_venue_details(r) + "\n\n"
        summary_parts.append(s.strip())

    # Shopping
    matched_shops = []
    for shop in data.get("shopping", []):
        if shop_matches_query(shop):
            matched_shops.append(shop)
    if matched_shops:
        s = "🛍️ Shops:\n"
        for r in matched_shops[:5]:
            s += format_venue_details(r) + "\n\n"
        summary_parts.append(s.strip())

    # Play facilities
    matched_play = []
    for facility in data.get("play", []):
        if shop_matches_query(facility):
            matched_play.append(facility)
    if matched_play:
        s = "🎮 Play facilities:\n"
        for r in matched_play[:5]:
            s += format_venue_details(r) + "\n\n"
        summary_parts.append(s.strip())

    # Events
    matched_events = []
    for ev in data.get("events", []):
        event_name = ev.get("name") or ev.get("title", "")
        combined = f"{event_name} {ev.get('description','')}"
        if fuzz.partial_ratio(query_lower, combined.lower()) >= threshold:
            matched_events.append(ev)
    if not matched_events and is_market_event_query(query):
        matched_events.extend(data.get("events", []))
    if matched_events:
        s = "🎉 Events:\n"
        for ev in matched_events[:3]:
            s += format_venue_details(ev) + "\n\n"
        summary_parts.append(s.strip())

    # Final fallback
    if not summary_parts:
        return None

    return "\n\n".join(summary_parts)

def extract_meal_query(text: str) -> str | None:
    """Return 'breakfast', 'lunch' or 'dinner' if text looks like a meal query."""
    lowered = text.lower()
    if fuzzy_match(lowered, ["dinner", "晚餐", "晚飯"]):
        return "dinner"
    if fuzzy_match(lowered, ["lunch", "午餐", "午飯"]):
        return "lunch"
    if fuzzy_match(lowered, ["breakfast", "早餐"]):
        return "breakfast"
    return None

def is_market_event_query(text: str) -> bool:
    """Return True if the text seems to ask about markets or weekend markets."""
    lowered = text.lower()
    if fuzzy_match(lowered, ["market", "weekend market", "市集"]):
        if not fuzzy_match(lowered, ["supermarket"]):
            return True
    return False

def restaurants_open_for(meal: str, lang: str = "en") -> str:
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
            times = re.findall(r"(\d{1,2}):(\d{2})", part)
            if len(times) >= 2:
                sh = int(times[0][0])
                eh = int(times[1][0])
                if sh <= start_h and eh >= end_h:
                    results.append(r)
                    break

    if not results:
        return ""
    if lang == "zh":
        header_map = {
            "breakfast": "🍳 早餐餐廳:",
            "lunch": "🍽 午餐餐廳:",
            "dinner": "🍴 晚餐餐廳:",
        }
    else:
        header_map = {
            "breakfast": "🍳 Breakfast options:",
            "lunch": "🍽 Lunch options:",
            "dinner": "🍴 Dinner options:",
        }
    header = header_map[meal]

    lines = [format_venue_details(v) for v in results[:5]]
    return header + "\n" + "\n\n".join(lines)


def is_smalltalk(text: str) -> bool:
    """Return True if the text looks like a greeting or other small talk."""
    t = text.strip().lower()
    return (
        fuzzy_match(t, GREETINGS)
        or fuzzy_match(t, FAREWELLS)
        or fuzzy_match(t, THANKS)
    )


def smalltalk_response(text: str) -> str:
    """Generate a friendly response for small-talk messages."""
    t = text.strip()
    lowered = t.lower()
    lang = detect_language(t)
    reply_lang = "en" if lang == "en" else "zh"
    if fuzzy_match(lowered, GREETINGS):
        return (
            "您好，這裡是D2 Place AI客服助理。請問需要什麼協助？"
            if reply_lang == "zh"
            else "Good day. This is D2 Place AI Customer Service Assistant. How may I help you?"
        )
    if fuzzy_match(lowered, THANKS):
        return (
            "不客氣！如果還有其他問題，隨時提出。"
            if reply_lang == "zh"
            else "You're welcome! If you have any more questions in the future, feel free to ask."
        )
    if fuzzy_match(lowered, FAREWELLS):
        return (
            "感謝你的查詢，對話已結束。如需協助，請再與我們聯絡。祝你有美好的一天！"
            if reply_lang == "zh"
            else "Thank you for your enquiry. The conversation has ended. Feel free to reach out again if you need help. Have a good day!"
        )
    return (
        "您好！有什麼可以為你效勞？"
        if reply_lang == "zh"
        else "Hello! How can I assist you with information about D2 Place?"
    )


def detect_language(text: str) -> str:
    """Return 'en', 'zh', or 'mixed' based on the characters in the text."""
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    return "en"


def postprocess_text(text: str) -> str:
    """Clean up chatbot responses for a consistent style."""
    cleaned = text.strip()
    # remove markdown style emphasis characters
    cleaned = re.sub(r"\*+([^\*]+)\*+", r"\1", cleaned)
    # unify bullet markers and remove leading hashes
    cleaned = re.sub(r"^\s*[\*#]+\s*", "- ", cleaned, flags=re.MULTILINE)
    return cleaned


def maybe_replace_unknown(text: str) -> str:
    """Replace generic unknown replies with a friendly message and hotline."""
    lowered = text.lower()
    if re.search(r"i\s+do(?:n't| not)\s+know", lowered):
        return (
            "I might not have the full details right now. "
            "Please let me know if there's anything else I can help with."
        )
    return text


def follow_up_promotion(query: str) -> str | None:
    """Return an extra promotional line based on the user query."""
    q = query.lower()
    if "pet" in q or "dog" in q:
        return (
            "You may also be interested in our pet-friendly weekend market "
            "where furry friends are welcome!"
        )
    return None


def should_send_image(text: str) -> bool:
    """Return True if the query explicitly asks for an image."""
    lowered = text.lower()
    return fuzzy_match(lowered, ["photo", "image", "picture", "poster", "instagram", "facebook"])


def is_parking_query(text: str) -> bool:
    """Return True if the user is asking about parking."""
    lowered = text.lower()
    keywords = ["parking", "car park", "carpark", "car-park", "泊車", "停車", "停車場", "車位"]
    return any(k in lowered for k in keywords)


def is_parking_promotion_query(text: str) -> bool:
    """Return True if the query is about parking promotions or discounts."""
    lowered = text.lower()
    if not is_parking_query(text):
        return False
    promo_keywords = ["promo", "promotion", "discount", "優惠", "折扣"]
    return any(k in lowered for k in promo_keywords)


def send_parking_images(recipient):
    """Send the parking images stored locally."""
    files = ["parking1.jpg", "parking2.jpg", "parking3.jpg", "parking4.png"]
    for f in files:
        url = urljoin(PUBLIC_URL + '/', f"assets/{f}")
        send_whatsapp_image(recipient, url)


@lru_cache(maxsize=1)
def get_parking_image_urls() -> list[str]:
    """Fetch parking page and return all image URLs."""
    url = "https://www.d2place.com/parking"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        imgs = []
        # Regular <img> tags
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            alt = (img.get("alt") or "").lower()
            if not src:
                continue
            full = urljoin(resp.url, src)
            if any(k in full.lower() for k in ["parking", "carpark", "car-park"]) or any(
                k in alt for k in ["parking", "car park", "carpark"]
            ):
                imgs.append(full)

        # Styles like style="background-image:url(...)"
        for el in soup.find_all(style=True):
            style = el.get("style", "")
            m = re.search(r"background-image:\s*url\(['\"]?([^')\"]+)['\"]?\)", style)
            if m:
                src = m.group(1)
                full = urljoin(resp.url, src)
                if "parking" in full.lower() and full not in imgs:
                    imgs.append(full)

        return imgs
    except Exception as e:
        logger.error("Failed fetching parking images: %s", e)
        return []




#########################
# 2. Qwen Handlers
#########################

def call_qwen_api(payload, retries: int = 2):
    """Call the Qwen endpoint and return the response text."""
    payload.setdefault("model", QWEN_MODEL)
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}/chat/completions"

    for attempt in range(retries + 1):
        try:
            start = time.monotonic()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            logger.info("Qwen API call took %.2f seconds", time.monotonic() - start)
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return postprocess_text(content)
            return postprocess_text(json.dumps(content))
        except Exception as e:
            logger.error("Qwen API error: %s", e)
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return (
                "I'm having trouble generating an answer right now. "
                "Please try again in a moment."
            )

def handle_text_query(user_text):
    system_prompt = (
       """
        You are a friendly assistant for D2 Place mall in Hong Kong. Answer user
        questions using the provided scraped data. You may enrich replies with
        web search results **only if** the restaurant, shop or event mentioned is
        confirmed to exist in the scraped JSON data. Otherwise politely indicate
        that the venue was not found. Only mention events that are happening currently or in the future; do not mention events that have ended already.

        Keep all answers focused on D2 Place or the LAWSGROUP community only.

        If details are missing, offer any related information you have instead of
        simply saying you don't know. Do not mention any concierge phone number
        and do not share or provide any phone numbers in your replies.
        Avoid using tables. Format each venue with its name, address, business
        hour and D2 Place page, separated by blank lines. Maintain a warm tone.
        """
    )

    user_lang = detect_language(user_text)
    reply_lang = "en" if user_lang == "en" else "zh"
    lang_instruction = (
        "Please answer in English." if reply_lang == "en" else "請以中文回答。"
    )

    meal = extract_meal_query(user_text)
    if meal:
        reply = restaurants_open_for(meal, reply_lang)
        if reply:
            return reply, None

    if is_smalltalk(user_text):
        return smalltalk_response(user_text), None

    # Gather relevant local data
    scraped_data = retrieve_relevant_data(user_text) or ""

    # Always attempt a small web search to enrich the response
    web_results = perform_web_search(user_text)

    # Try to find social media links and an image. Previously this only
    # triggered when certain keywords were present, which meant queries like
    # "PowerPlay Arena" would not fetch any images. We now also trigger this
    # logic when the query matches our scraped data so that images are returned
    # for venue specific queries.
    social_links, social_image = ([], None)
    lower_query = user_text.lower()
    if scraped_data or fuzzy_match(lower_query, ["event", "shop", "happening", "store"]):
        social_links, img = search_social_media_links(user_text)
        if social_links:
            web_results += "\nSocial Media:\n" + "\n".join(social_links)
        if should_send_image(user_text):
            social_image = img


    full_prompt = (
        f"{system_prompt}\n\n"
        f"Full Mall Data JSON:\n{FULL_JSON_TEXT}\n\n"
        # f"User Question: {user_text}\n\n"
        # f"SCRAPED DATA:\n{scraped_data}\n\n"
        f"WEB SEARCH:\n{web_results}\n"
        f"{lang_instruction}\n"

    )
    payload = {
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    response = call_qwen_api(payload)
    final_reply = maybe_replace_unknown(postprocess_text(response))
    final_reply = remove_phone_numbers(final_reply)
    promo = follow_up_promotion(user_text)
    if promo:
        final_reply += "\n\n" + promo
    return final_reply, social_image

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
    return postprocess_text(call_qwen_api(payload))

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
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
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
        return "抱歉，無法轉錄你的語音訊息。請稍後再試。", None

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
    with _messages_lock:
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

    BOT_NUMBER = PHONE_NUMBER_ID
    LOG_NUMBER = LOG_RECIPIENT
    for m in messages:
        from_user = m.get('from', '')
        msg_type = m.get('type', '')
        if from_user not in (BOT_NUMBER, LOG_NUMBER):
            summary = summarize_message_for_log(m)
            send_whatsapp_message(LOG_RECIPIENT, f"📥 From {from_user} ({msg_type}): {summary}")
        _executor.submit(process_message, m)

    return jsonify(status="processing", count=len(messages)), 200

def process_message(msg):
    """Handle a single WhatsApp message in a background thread."""
    msg_id = msg.get('id')
    if msg_id:
        with _messages_lock:
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

        # Send immediate greeting/kind note to reduce perceived wait time
        lang = detect_language(inbound_text) if msg_type == 'text' else 'zh'
        note = (
            "Good day. This is D2 Place AI Customer Service Assistant. Processing your request, please wait……"
            if lang == 'en'
            else "系統正在處理，請稍等……"
        )
        send_whatsapp_message(from_user, note)

        if msg_type == 'text' and is_parking_query(inbound_text):
            send_parking_images(from_user)

        if msg_type == 'text':
            bot_reply, image_url = handle_text_query(inbound_text)
        elif msg_type == 'audio':
            audio_bytes, content_type = download_media_file(msg['audio']['id'])
            transcript = transcribe_audio(audio_bytes, content_type)
            if not transcript:
                bot_reply, image_url = "抱歉，我無法轉錄你的語音訊息。請稍後再試。", None
            else:
                bot_reply, image_url = handle_text_query(transcript)
        else:
            bot_reply, image_url = "Sorry, I only handle text and audio messages for now.", None

        send_whatsapp_message(LOG_RECIPIENT, f"📤 To   {from_user}: {bot_reply}")
        send_whatsapp_message(from_user, bot_reply)
        if image_url:
            send_whatsapp_image(from_user, image_url)

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")

def download_media_file(media_id):
    """
    Retrieves the media file from WhatsApp Cloud API and returns raw bytes and content type.
    """
    # 1) get media URL
    info_url = f"https://graph.facebook.com/v16.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    info_resp = requests.get(info_url, headers=headers, timeout=20)
    info_resp.raise_for_status()
    media_url = info_resp.json().get("url")
    mime = info_resp.json().get("mime_type", "")

    # 2) download raw media
    file_resp = requests.get(media_url, headers=headers, timeout=20)
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
        start = time.monotonic()
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        logger.info("WhatsApp send response: %s (%.2fs)", resp.text, time.monotonic() - start)
    except Exception as e:
        logger.error("Error sending WhatsApp message: %s", e)

def send_whatsapp_image(recipient, image_url, caption=""):
    """Send an image via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v16.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "image",
        "image": {"link": image_url, "caption": caption},
    }
    try:
        start = time.monotonic()
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        logger.info(
            "WhatsApp image response: %s (%.2fs)",
            resp.text,
            time.monotonic() - start,
        )
    except Exception as e:
        logger.error("Error sending WhatsApp image: %s", e)

def summarize_message_for_log(msg: dict) -> str:
    """Return a short description of the incoming message for logging."""
    msg_type = msg.get('type', '')
    if msg_type == 'text':
        return msg.get('text', {}).get('body', '')
    if msg_type == 'image':
        return f"<image id:{msg.get('image', {}).get('id')}> caption:{msg.get('image', {}).get('caption', '')}"
    if msg_type == 'audio':
        media = msg.get('audio', {})
        return f"<audio id:{media.get('id')} mime:{media.get('mime_type', 'audio/unknown')}>"
    return f"<{msg_type}>"

#########################
# 5. Run the Flask App
#########################

if __name__ == '__main__':
    print("Starting the Flask app. Callback URL is not shown here because it's local.")
    print("To get a public URL, run: ngrok http 4040 and use the domain + /webhook as callback.")
    app.run(port=4040, debug=True)
