import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # Ensure your .env file is in the same directory
QWEN_API_KEY = os.environ.get('QWEN_API_KEY')
SERP_API_KEY = os.environ.get('SERP_API_KEY')
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

print("Your Qwen API key:", QWEN_API_KEY)
print("Your SerpAPI key:", SERP_API_KEY)

def perform_web_search(query):
    """
    Perform a web search using the SerpAPI and return a summary of the top results.
    """
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "num": "5",  # Retrieve top 3 results
        "api_key": SERP_API_KEY
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        summary_lines = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "")
                summary_lines.append(f"{title}: {snippet}")
                print("Web search completed successfully.")
        return "\n".join(summary_lines)
    except Exception as e:
        print("Error during SerpAPI web search:", e)
        return "Web search unavailable."

def retrieve_relevant_data(query):
    """
    Load the scraped D2 Place data from d2place_data.json and return
    a brief summary of relevant information based on keywords in the query.
    """
    try:
        with open("d2place_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            print("Scraped data loaded successfully.")
    except Exception as e:
        print("Error loading scraped data:", e)
        return ""

    query_lower = query.lower()
    summary_parts = []

    # Check dining keywords
    if any(keyword in query_lower for keyword in ["dining", "restaurant", "food", "eat"]):
        dining_list = data.get("dining", [])
        if dining_list:
            summary = "Dining Options:\n"
            for restaurant in dining_list[:2]:
                summary += f"- {restaurant.get('name', 'Unknown')} (Hours: {restaurant.get('hours', 'N/A')})\n"
            summary_parts.append(summary)

    # Check shopping keywords
    if any(keyword in query_lower for keyword in ["shopping", "store", "shop", "buy"]):
        shopping_list = data.get("shopping", [])
        if shopping_list:
            summary = "Shopping Options:\n"
            for shop in shopping_list[:2]:
                summary += f"- {shop.get('name', 'Unknown')} (Hours: {shop.get('hours', 'N/A')})\n"
            summary_parts.append(summary)

    # Check events keywords
    if any(keyword in query_lower for keyword in ["event", "promotion", "activity", "show", "exhibition"]):
        events_list = data.get("events", [])
        if events_list:
            summary = "Upcoming Events:\n"
            for event in events_list[:2]:
                summary += f"- {event.get('name', 'Unknown')} (Date: {event.get('date', 'N/A')})\n"
            summary_parts.append(summary)

    # If no specific category is triggered, include basic mall info
    if not summary_parts:
        mall_info = data.get("mall_info", {})
        if mall_info:
            summary = "Mall Information:\n"
            for key, value in mall_info.items():
                if value:
                    summary += f"{key.capitalize()}: {value}\n"
            summary_parts.append(summary)

    return "\n".join(summary_parts)

def handle_text_query(query):
    """
    Constructs a prompt including a system message with relevant scraped data and web search results,
    sends it to Qwen-Turbo, and returns the assistant's response.
    """
    system_prompt = (
        "You are a helpful assistant for D2 Place mall in Hong Kong. "
        "Answer user queries regarding dining, shopping, events, directions, promotions, parking, and special services. "
        "Use the most current information scraped from the D2 Place website as well as searching online yourself on https://www.d2place.com/ and going into each sub domain in the website to find the relevant information."
        "Respond in the same language as the query."
    )
    scraped_data = retrieve_relevant_data(query)
    web_results = perform_web_search(query)
    full_prompt = (
        f"{system_prompt}\n\n"
        f"User Query: {query}"
        f"Scraped Data:\n{scraped_data}\n\n"
        f"Web Search Results:\n{web_results}\n\n"
    )
    print(web_results)

    payload = {
        "model": "qwen-turbo",  # Adjust this model name if needed.
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.5,
        "max_tokens": 150
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/chat/completions"
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Parse the response according to the Qwen API's response format.
        answer = result['choices'][0]['message']['content']
        return answer.strip()
    except Exception as e:
        print("Qwen API error (text):", e)
        return "Sorry, I'm having trouble generating an answer right now."

def main():
    print("Welcome to the D2 Place Terminal Chatbot Test using Qwen-Turbo with SerpAPI web search.")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        answer = handle_text_query(user_input)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
