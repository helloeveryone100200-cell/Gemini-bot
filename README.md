# Telegram Gemini AI Bot

A smart Telegram bot powered by Google Gemini AI with text and image understanding.

## Features

- Text conversation with memory (keeps recent context)
- Photo analysis and image-based follow-up questions
- Smart context routing — knows when to continue vs. start fresh
- Auto-retries across Gemini models (2.5 Flash Lite → 2.5 Flash → 2.0 Flash)
- Supports polling (local) and webhook (Render / Hugging Face) modes
- SQLite conversation memory with automatic pruning

## Required environment variables

| Variable | Description |
|---|---|
| `BOT_TOKEN` | Telegram bot token from @BotFather |
| `GEMINI_API_KEY` | Google Gemini API key |

## Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `WEBHOOK_BASE_URL` | _(empty)_ | Public HTTPS URL — enables webhook mode |
| `WEBHOOK_PATH` | `/telegram/webhook` | Webhook endpoint path |
| `PORT` | `7860` | HTTP server port |
| `MAX_HISTORY_MESSAGES` | `12` | Messages kept in memory |
| `REQUEST_TIMEOUT_SECONDS` | `90` | Gemini API timeout |
| `MAX_OUTPUT_TOKENS` | `1400` | Max tokens per response |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in BOT_TOKEN and GEMINI_API_KEY
python bot.py
```

## Deploy on Render

1. Push this repo to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Set `BOT_TOKEN`, `GEMINI_API_KEY`, `WEBHOOK_BASE_URL` in environment
4. Render auto-deploys on every push

## Commands

| Command | Description |
|---|---|
| `/start` | Start the bot |
| `/new` or `/reset` | Clear conversation memory |
| `/cancel` | Reset mode, keep memory |
