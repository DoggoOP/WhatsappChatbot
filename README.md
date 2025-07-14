# D2Place WhatsApp Chatbot

This repository contains a Flask application that powers a WhatsApp chatbot for **D2 Place** mall in Hong Kong. The bot answers queries about shops, dining options, events and other mall information using data scraped from the D2 Place website as well as small web searches.

## Features

- **WhatsApp integration** via the WhatsApp Cloud API.
- Uses **Qwen** APIs for text generation and audio transcription.
- Performs lightweight web searches through **SerpAPI** to enrich responses.
- Includes a Selenium based **scraper** (`scraper.py`) that gathers shop, event and venue details into `d2place_data.json`.

## Setup

1. Create a Python 3 environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env` and set the following variables:
   - `QWEN_API_KEY` – key for the Qwen API
   - `SERP_API_KEY` – SerpAPI key for Google search
   - `VERIFY_TOKEN` – token used when validating the webhook URL with WhatsApp
   - `WHATSAPP_TOKEN` – WhatsApp Cloud API access token
   - `PHONE_NUMBER_ID` – your WhatsApp phone number ID
   - `LOG_RECIPIENT` – phone number where log messages should be sent
  - `PUBLIC_URL` – base URL of your Flask server used for serving images.
    This must be a publicly reachable URL (e.g. an ngrok tunnel) so WhatsApp
    can download files from the `/Assets` path. The app exposes this route
    using `@app.route('/Assets/<path:filename>')`, serving files from the local
    `Assets` folder. Set this to the public domain of your server (for example
    `https://chatbot.d2place.com`).

## Running the bot

The Flask app listens on port **4040**. Run it with:
```bash
python app.py
```
Expose the port to the internet (for example with `ngrok http 4040`) and configure the resulting URL as your webhook on the WhatsApp Cloud console.

## Updating scraped data

The file `scraper.py` fetches data from the D2 Place website and saves it to `d2place_data.json`. You can run it manually:
```bash
python scraper.py
```
The script also schedules a weekly run every Monday at 02:00 Hong Kong time when executed directly.

The JSON file contains a `manual_info` section where you can store custom notes.
This portion of the file is preserved whenever the scraper runs, so feel free to
edit it manually without worrying about it being overwritten.

## License

This project is provided as‑is under the MIT license. See `LICENSE` for details.
