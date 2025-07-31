# D2Place WhatsApp Chatbot

This repository contains a Flask application that powers a WhatsApp chatbot for **D2 Place** mall in Hong Kong. The bot answers queries about shops, dining options, events and other mall information using data scraped from the D2 Place website as well as small web searches.

## Features
<img width="801" height="375" alt="image" src="https://github.com/user-attachments/assets/b3584f4e-aa36-4281-a7fb-6696c4c69c44" />


- **WhatsApp integration** via the WhatsApp Cloud API.
- Uses **Qwen** APIs for text generation and audio transcription.
- Performs lightweight web searches through **SerpAPI** to enrich responses.
- Includes a Selenium based **scraper** (`scraper.py`) that gathers shop, event and venue details into `d2place_data.json`.

## Architecture Overview
The bot runs on an Alibaba Cloud ECS instance (Ubuntu 22.04) listening on `localhost:4040`. An ngrok tunnel exposes the `/webhook` endpoint publicly at `https://chatbot.d2place.com/webhook`.

### Message flow
1. A shopper sends a WhatsApp message.
2. The WhatsApp Cloud API POSTs the event to the webhook through ngrok.
3. `app.py` receives the JSON, queries **Qwen** for language output and **SerpApi** for Google snippets, merges them with mall data from `d2place_data.json` and formats the reply.
4. The formatted text is POSTed back to WhatsApp via the Meta API, which delivers it to the user.

### Scheduled scraping
`scraper.py` re-scrapes d2place.com every Monday to refresh `d2place_data.json`. The job runs as a systemd service using the `schedule` library.

### Services and infrastructure
- `d2place-app.service` – runs `app.py`
- `d2place-scraper.service` – runs `scraper.py`
- `ngrok.service` – maintains the public tunnel
Systemd keeps these units alive and restarts them on failure.

The project relies on several third-party services:
- WhatsApp Cloud API for messaging
- SerpApi for search
- Ngrok for HTTPS tunneling
- Qwen (Alibaba Cloud) for language model inference
- GitHub for source control

### Environment variables
API tokens and configuration values are stored in `.env` (e.g. `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `VERIFY_TOKEN`, `QWEN_API_KEY`, `SERP_API_KEY` and `LOG_RECIPIENTS`). Store these secrets securely.

### Operations
- Update the code with `git pull` and restart the services with `systemctl`.
- View logs with `journalctl -u d2place-app.service -f`.
- Trigger a manual scrape using `systemctl start d2place-scraper.service`.

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
   - `LOG_RECIPIENTS` – comma-separated WhatsApp numbers that should receive log messages
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
The script also schedules a daily run every day at 02:00 Hong Kong time when executed directly.
The Flask app reloads `d2place_data.json` at 03:00 so it can use the freshly scraped data without restarting.

The JSON file contains a `manual_info` section where you can store custom notes.
This portion of the file is preserved whenever the scraper runs, so feel free to
edit it manually without worrying about it being overwritten.

## License

This project is provided as‑is under the MIT license. See `LICENSE` for details.
