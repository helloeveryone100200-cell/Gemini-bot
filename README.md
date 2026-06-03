# Telegram Gemini AI Bot

A smart Telegram bot powered by Google Gemini AI that supports both text and image understanding.

## Features

- Text conversation with memory (keeps recent context across messages)
- Photo analysis and image-based follow-up questions
- Smart context routing — knows when to continue vs. start fresh
- Auto-retries across multiple Gemini models (2.5 Flash Lite → 2.5 Flash → 2.0 Flash)
- Supports both polling (local) and webhook (Render / Hugging Face Spaces) modes
- SQLite-based conversation memory with automatic pruning

## Setup

### Required environment variables

| Variable | Description |
|---|---|
| `BOT_TOKEN` | Telegram bot token from @BotFather |
| `GEMINI_API_KEY` | Google Gemini API key |

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_BASE_URL` | _(empty)_ | Public HTTPS URL — enables webhook mode |
| `WEBHOOK_PATH` | `/telegram/webhook` | Path for webhook endpoint |
| `WEBHOOK_SECRET` | _(auto)_ | Webhook secret (auto-derived from token) |
| `PORT` | `7860` | HTTP server port |
| `MAX_HISTORY_MESSAGES` | `12` | Number of messages kept in memory |
| `REQUEST_TIMEOUT_SECONDS` | `90` | Gemini API timeout |
| `MAX_OUTPUT_TOKENS` | `1400` | Max tokens per Gemini response |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in BOT_TOKEN and GEMINI_API_KEY
python bot.py
```

## Deploy on Render

1. Fork or push this repo to your GitHub account
2. Create a new Web Service on [Render](https://render.com)
3. Set `BOT_TOKEN`, `GEMINI_API_KEY`, and `WEBHOOK_BASE_URL` in the Render environment
4. Render will build and deploy automatically

## Deploy on Hugging Face Spaces (Docker)

Set the following in your Space Secrets:
- `BOT_TOKEN`
- `GEMINI_API_KEY`
- `WEBHOOK_BASE_URL` — e.g. `https://username-space-name.hf.space`
- `WEBHOOK_PATH` — default: `/telegram/webhook`
- `WEBHOOK_SECRET` — any secret without spaces

## Usage

| Command | Description |
|---|---|
| `/start` | Start the bot and see the menu |
| `/new` or `/reset` | Clear conversation memory |
| `/cancel` | Reset mode, keep memory |

- Tap **Text Mode** to explicitly switch to text questions
- Tap **Photo Mode** to switch to image analysis
- In Photo Mode, send a follow-up text to ask more about the last photo
