# Chatbot — Setup, Deployment & Integration Guide

This document is everything you need to deploy this chatbot on your own
server and connect it to your website or app. It assumes no prior experience
with this specific codebase — just basic comfort with a terminal.

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [How It Works (Architecture)](#2-how-it-works-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Environment Variables Reference](#4-environment-variables-reference)
5. [Deployment: Step-by-Step](#5-deployment-step-by-step)
6. [Verifying the Deployment](#6-verifying-the-deployment)
7. [API Reference](#7-api-reference)
8. [Frontend Integration](#8-frontend-integration)
9. [Managing Your Content (Adding/Updating What the Bot Knows)](#9-managing-your-content-addingupdating-what-the-bot-knows)
10. [Going Live: Domain, HTTPS, and CORS](#10-going-live-domain-https-and-cors)
11. [Daily Operations](#11-daily-operations)
12. [Security Notes](#12-security-notes)
13. [Troubleshooting](#13-troubleshooting)
14. [Support Checklist](#14-support-checklist)

---

## 1. What This Is

A self-hosted AI chatbot that answers questions using **your own content**
(products, services, policies, team info, FAQs — whatever you feed it). It
does not rely on the AI's general knowledge to answer; it only answers from
the documents you give it, and says so honestly when it doesn't know
something.

Under the hood it's a Retrieval-Augmented Generation (RAG) system: it
searches your content for the most relevant information, then asks an AI
model to write a natural-language answer strictly from that information.

---

## 2. How It Works (Architecture)

```
                 User asks a question
                          │
                          ▼
              ┌───────────────────────┐
              │  Understanding Layer   │   Figures out what the user
              │  (intent + planning)   │   is really asking, resolves
              └───────────┬────────────┘   pronouns using chat history
                          │
                          ▼
              ┌───────────────────────┐
              │    Retrieval Layer     │   Pulls the most relevant
              │ (metadata + semantic)  │   content: exact filters
              └───────────┬────────────┘   (price, category) AND
                          │                 semantic/meaning-based search
                          ▼
              ┌───────────────────────┐
              │    Generation Layer    │   Writes the final answer
              │   (AI model call)      │   using ONLY the retrieved
              └───────────┬────────────┘   content — no guessing
                          │
                          ▼
                   Answer sent back
                 to your website/app
```

**Components running on your server:**

| Component | Purpose |
|---|---|
| Chatbot API (FastAPI, in Docker) | The main service your frontend talks to |
| Redis (in Docker) | Stores short-term conversation memory (1 hour) and rate-limit counters |
| Chroma (embedded in the chatbot container's data volume) | Vector database — powers "search by meaning" |
| OpenAI API (external) | The actual AI model that understands questions and writes answers |

This is a single-server setup — appropriate for a small-to-medium business.
There's no need for multiple servers, load balancers, or cloud infrastructure
beyond what's described here.

---

## 3. Prerequisites

You only need **Docker** on your server. No Python, no manual dependency
installation.

- A Linux server (Ubuntu 22.04/24.04 recommended) with at least **2 CPU
  cores and 4GB RAM** free.
- Docker installed: https://docs.docker.com/engine/install/
- An **OpenAI API key** (this is required — the bot cannot run without it):
  https://platform.openai.com/api-keys
- A domain name pointing at your server, if you want a proper `https://`
  address (recommended for production — see [Section 10](#10-going-live-domain-https-and-cors)).

Verify Docker is installed:
```bash
docker --version
docker compose version
```

---

## 4. Environment Variables Reference

All configuration lives in a single `.env` file at the project root. Copy
`.env.example` to `.env` and fill in the values below.

```bash
cp .env.example .env
nano .env
```

| Variable | Required? | Description |
|---|---|---|
| `ENVIRONMENT` | Yes | `production` for live deployment, `development` for local testing. Controls cookie security settings and hides `/docs` in production. |
| `DEBUG` | No | `true`/`false`. Leave `false` in production. |
| `HOST` | No | Leave as `0.0.0.0`. |
| `PORT` | No | Port the API listens on inside the container. Default `8000`. |
| `API_KEY` | **Yes** | A secret key that protects your chatbot from unauthorized use. Must be **32+ characters**. Generate one at https://randomkeygen.com or run `openssl rand -hex 32`. Your frontend must send this in every request. |
| `OPENAI_API_KEY` | **Yes** | Your OpenAI API key. Without this, the service will not start. |
| `LLM_MODEL` | No | The model used to write final answers. Default `gpt-4o-mini` — good balance of quality and cost. |
| `LOGIC_MODEL` | No | A cheaper/faster model used internally to understand the question. Default `gpt-4o-mini`. |
| `EMBED_MODEL` | No | The OpenAI embedding model used for semantic search. Default `text-embedding-3-small`. Don't change this after you've built your data indexes (see [Section 9](#9-managing-your-content-addingupdating-what-the-bot-knows)) without rebuilding them. |
| `CHROMA_DB_PATH` | No | Where the vector database is stored inside the container. Leave as default unless you know why you're changing it. |
| `METADATA_INDEX_PATH` | No | Where the structured content index is stored. Leave as default. |
| `CHROMA_COLLECTION` | No | Internal name for your vector database collection. Any short name, no spaces (e.g. `mycompany_rag_collection`). |
| `COMPANY_EMAIL` | Yes | Shown to users when the bot can't answer something ("please contact us at..."). |
| `COMPANY_NAME` | Yes | Used in the bot's own self-description and API title. |
| `SYSTEM_NAME` | Yes | The bot's name, e.g. `"Sherpa Assistant"`. |
| `GREETING` | No | A greeting word/phrase the bot may use, e.g. `"Hello"` or `"Namaste"`. |
| `REDIS_URL` | No | Leave as `redis://redis:6379/0` in Docker Compose (the service name `redis` resolves automatically inside the Docker network). |
| `ALLOWED_ORIGINS` | **Yes** | Comma-separated list of the exact website domain(s) allowed to call this API, e.g. `https://yourcompany.com,https://www.yourcompany.com`. **Never set this to `*`** — the app will refuse to start if you do (see [Section 10](#10-going-live-domain-https-and-cors) for why). |
| `DAILY_TOKEN_LIMIT` | No | Soft internal budget guard, in tokens. Not currently enforced as a hard cutoff — treat as informational/for future use. |

---

## 5. Deployment: Step-by-Step

### Step 1 — Get the project files onto your server

```bash
unzip chatbot-delivery.zip
cd chatbot-delivery
```

### Step 2 — Configure your environment

```bash
cp .env.example .env
nano .env
```

Fill in at minimum: `API_KEY`, `OPENAI_API_KEY`, `COMPANY_NAME`,
`COMPANY_EMAIL`, `SYSTEM_NAME`, `ALLOWED_ORIGINS`. See [Section 4](#4-environment-variables-reference)
for what each means. Save and exit (`Ctrl+X` → `Y` → `Enter`).

### Step 3 — Add your content

Your chatbot's knowledge comes from JSON files under
`data/datasets/rag_ready/`. See [Section 9](#9-managing-your-content-addingupdating-what-the-bot-knows)
for the full guide on formatting and adding your own content. If you were
handed a pre-built dataset along with this project, it's likely already
in place — check with whoever delivered this project to you.

### Step 4 — Build the indexes

This turns your content into the two search indexes the bot uses (exact
filtering + semantic search):

```bash
docker compose run --rm indexer
```

You should see:
```
=== Step 1: Building metadata index ===
=== Step 2: Building vector DB ===
=== Indexing complete ===
```

This step calls the OpenAI API to generate embeddings for your content, so
it will consume a small amount of API credit and may take a few minutes
depending on how much content you have. **Re-run this any time you add or
change content** (see Section 9).

### Step 5 — Start the chatbot

```bash
docker compose up -d chatbot redis
```

### Step 6 — Verify it's running

```bash
curl http://localhost:8000/v1/health/
```

Expected response:
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "environment": "production",
  "service": "Your Company Name Chatbot API"
}
```

> Give it up to 60 seconds after starting — the AI pipeline needs a moment to
> warm up on first launch.

If you see `"status": "degraded"`, wait longer, then check logs:
```bash
docker compose logs chatbot
```

---

## 6. Verifying the Deployment

A real chat conversation requires two calls: first mint a session, then chat
using that session. This mirrors exactly what your frontend will do.

```bash
# 1. Get a session (saves the cookie to cookies.txt)
curl -X POST http://localhost:8000/v1/session/ \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -c cookies.txt

# 2. Ask a question using that session
curl -X POST http://localhost:8000/v1/chat/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -b cookies.txt \
  -d '{"message": "Hello, what services do you offer?"}'
```

Replace `YOUR_API_KEY_HERE` with the value you set in `.env`. A healthy
response looks like:

```json
{
  "success": true,
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "answer": "We offer ...",
  "sections": [],
  "suggestions": ["What are your prices?", "Do you offer support?"],
  "sources": [{"title": "Services Overview", "source": "services-overview.json"}],
  "latency_ms": 1180,
  "is_fallback": false
}
```

If this works, your frontend integration will work too — it's doing exactly
this, just from browser JavaScript instead of curl.

---

## 7. API Reference

All endpoints are under the `/v1` prefix. `Content-Type: application/json`
is required on all `POST` bodies.

### `POST /v1/session/` — Start a conversation

Call this once per user visit (or whenever there's no valid session cookie
yet). Mints a session and sets it as a cookie automatically.

**Headers:** `X-API-Key: <your API key>`

**Response `201`:**
```json
{ "session_id": "uuid-string", "expires_in": 3600 }
```

A `Set-Cookie: session_id=...` header is included automatically — your
frontend doesn't need to read or store this value manually; the browser
handles it, as long as requests are made with credentials enabled (see
[Section 8](#8-frontend-integration)).

Rate limited to 10 requests/minute per caller.

### `POST /v1/chat/` — Send a message

**Headers:** `X-API-Key: <your API key>`, plus the `session_id` cookie
(automatically attached by the browser after calling `/v1/session/`).

**Body:**
```json
{ "message": "What is the price of the Everest Base Camp trek?" }
```
`message` must be 1–5000 characters.

**Response `200`:**
```json
{
  "success": true,
  "session_id": "uuid-string",
  "answer": "Plain-language answer for display",
  "sections": [
    { "type": "table", "title": "Pricing", "items": [ {"...": "..."} ], "content": null }
  ],
  "suggestions": ["Follow-up question 1", "Follow-up question 2"],
  "sources": [ {"title": "Document title", "source": "file-or-id"} ],
  "latency_ms": 1180,
  "is_fallback": false
}
```

- `answer` — always safe to display as the primary response text.
- `sections` — optional structured content (tables/lists/cards) you can
  render specially if your frontend supports it; otherwise ignore and just
  show `answer`.
- `is_fallback: true` — the bot couldn't find a confident answer and gave a
  generic "please contact us" response. Useful to detect and log on your
  side too.

**Error responses:**
| Status | Meaning |
|---|---|
| `401` | Missing/invalid `X-API-Key`, or missing `session_id` cookie |
| `400` | `session_id` cookie present but malformed |
| `429` | Rate limit exceeded (20 messages/minute per session) |
| `503` | Server still warming up — retry shortly |
| `500` | Unexpected server error — check `docker compose logs chatbot` |

Rate limited to 20 requests/minute per session.

### `GET /v1/health/` — Health check

No API key required. Used by Docker's own health monitoring and useful for
your own uptime monitoring.

```json
{ "status": "healthy", "pipeline_ready": true, "environment": "production", "service": "..." }
```

Returns HTTP `503` if `pipeline_ready` is `false`.

---

## 8. Frontend Integration

### The flow, in plain terms

1. When your chat widget loads, check if you already have an active
   session. If not, call `POST /v1/session/`.
2. From then on, every call to `/v1/session/` or `/v1/chat/` must be made
   **with credentials included**, so the browser sends the session cookie.
3. Call `POST /v1/chat/` with the user's message each time they send one.

### Example: vanilla JavaScript (fetch)

```javascript
const API_BASE = "https://api.yourcompany.com/v1";
const API_KEY = "YOUR_API_KEY_HERE"; // see security note below

async function ensureSession() {
  // A simple check: try a chat call; if it 401s for a missing session,
  // mint one. In practice, most teams just call /session/ once per
  // page load / widget open and don't worry about reusing it across visits.
  await fetch(`${API_BASE}/session/`, {
    method: "POST",
    credentials: "include", // REQUIRED -- lets the browser store/send the cookie
    headers: { "X-API-Key": API_KEY },
  });
}

async function sendMessage(message) {
  const res = await fetch(`${API_BASE}/chat/`, {
    method: "POST",
    credentials: "include", // REQUIRED on every call
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify({ message }),
  });

  if (res.status === 401) {
    // Session expired or missing -- mint a new one and retry once.
    await ensureSession();
    return sendMessage(message);
  }

  return res.json();
}

// Usage:
await ensureSession();
const reply = await sendMessage("What is the price of the Everest Base Camp trek?");
console.log(reply.answer);
```

### Example: React

```jsx
import { useState, useCallback, useEffect } from "react";

const API_BASE = "https://api.yourcompany.com/v1";
const API_KEY = process.env.REACT_APP_CHATBOT_API_KEY;

export function useChatbot() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/session/`, {
      method: "POST",
      credentials: "include",
      headers: { "X-API-Key": API_KEY },
    }).then(() => setReady(true));
  }, []);

  const sendMessage = useCallback(async (message) => {
    const res = await fetch(`${API_BASE}/chat/`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({ message }),
    });
    return res.json();
  }, []);

  return { ready, sendMessage };
}
```

### Important: don't expose `API_KEY` in a fully public bundle without limits

The `X-API-Key` protects against casual abuse and scraping, but any key
shipped to the browser is technically visible to anyone who opens dev tools.
For most small-business use cases (a support widget on your own site) this
is an acceptable, standard tradeoff — combine it with:
- Setting `ALLOWED_ORIGINS` to your exact domain (already required — see
  Section 10), so the API only accepts credentialed requests from your site.
- Keeping the built-in rate limits (20 messages/minute/session) in place.

If you later need stronger protection (e.g. the bot is expensive to run
and a target for abuse), the standard fix is to proxy `/v1/chat/` through
your own backend instead of calling the chatbot API directly from the
browser, so the API key never reaches the client at all. That's a larger
change and not included in this delivery — ask your developer if you need it.

---

## 9. Managing Your Content (Adding/Updating What the Bot Knows)

The chatbot only knows what's in `data/datasets/rag_ready/`. Everything else
is generic scaffolding — swapping content here is how you make it "yours."

### Content format

Each piece of content is one JSON file, organized into subfolders by
category (the folder name becomes the `category` used for filtering):

```
data/datasets/rag_ready/
├── Products/
│   └── wireless-headphones.json
├── Policies/
│   └── refund-policy.json
├── Team/
│   └── jane-doe.json
└── FAQs/
    └── shipping-times.json
```

Minimal example (`Products/wireless-headphones.json`):

```json
{
  "source_url": "https://yourstore.com/products/wireless-headphones",
  "filename": "products-wireless-headphones.md",
  "category": "Products",
  "meta_tags": {
    "summary": "Over-ear wireless headphones with 30-hour battery life.",
    "keywords": ["headphones", "wireless", "audio"],
    "price": 149,
    "brand": "SoundWave",
    "in_stock": true
  },
  "content": "Full product description goes here. This is the text the AI reads to answer questions about this item."
}
```

Rules:
- `content` must be at least 500 characters (shorter files are skipped as
  likely junk/stubs — set `MIN_CONTENT_LEN` in `.env` to change this).
- Any field you add inside `meta_tags` becomes automatically filterable/
  sortable (price ranges, categories, boolean flags, etc.) — nothing to
  configure.
- `content` is what gets semantically searched — write it the way you'd
  want a customer to read it.

### After adding or changing any content

You must rebuild **both** indexes, every time:

```bash
docker compose run --rm indexer
```

Updating only some content and forgetting this step is the #1 cause of
"the bot doesn't know about the thing I just added." There is no automatic
watch/reload — a rebuild is required after every content change.

### Changing your branding

Edit these in `.env` and restart (`docker compose restart chatbot`) — no
code changes needed:

```env
COMPANY_NAME=Your Company Name
SYSTEM_NAME=Your Bot Name
COMPANY_EMAIL=support@yourcompany.com
GREETING=Hello
```

---

## 10. Going Live: Domain, HTTPS, and CORS

The chatbot listens on port `8000` inside your server. To make it reachable
at a real domain with `https://`, put a reverse proxy in front of it — this
part requires a developer/sysadmin if you don't already have one set up:

- **Nginx** (or Caddy/Traefik) reverse-proxying `https://api.yourcompany.com`
  → `http://localhost:8000`
- **SSL certificate** — free via [Let's Encrypt](https://letsencrypt.org/)
- **DNS** — point `api.yourcompany.com` at your server's IP

**Then set `ALLOWED_ORIGINS` in `.env`** to the exact domain(s) your website
lives on — not the API's own domain:

```env
ALLOWED_ORIGINS=https://yourcompany.com,https://www.yourcompany.com
```

**Do not set this to `*`.** The chat feature relies on cookies to remember a
conversation, and browsers refuse to send cookies on cross-site requests
unless the server names an exact origin — a wildcard is incompatible with
that and the app will refuse to start rather than deploy silently broken.

After changing `ALLOWED_ORIGINS`, restart: `docker compose restart chatbot`.

---

## 11. Daily Operations

| Task | Command |
|---|---|
| View live logs | `docker compose logs -f chatbot` |
| Stop the chatbot | `docker compose down` |
| Start the chatbot | `docker compose up -d chatbot redis` |
| Restart after a `.env` change | `docker compose restart chatbot` |
| Check container status | `docker compose ps` |
| Rebuild indexes after content changes | `docker compose run --rm indexer` |
| Check health | `curl http://localhost:8000/v1/health/` |

---

## 12. Security Notes

- **`API_KEY`** gates all `/v1/chat/` and `/v1/session/` calls. Treat it like
  a password — regenerate it (and update your frontend) if you ever suspect
  it's leaked beyond your own website.
- **Rate limits** are already in place: 20 chat messages/minute per
  conversation, 10 session creations/minute per caller. This limits abuse
  and cost from any single source.
- **Your `.env` file contains secrets** (`API_KEY`, `OPENAI_API_KEY`) —
  never commit it to a public repository or share it outside your team.
- **Logs** (`data/logs/app.log`) contain the full text of user messages, for
  debugging and quality purposes. If your users might type personal
  information into the chat, treat these logs with the same care as any
  other file containing customer data, and set a deletion schedule that
  matches your privacy policy.
- **This chatbot only answers from your content.** If it doesn't find a
  relevant answer, it says so and points the user to `COMPANY_EMAIL` rather
  than guessing — this is intentional and reduces the risk of it giving
  wrong information.

---

## 13. Troubleshooting

**Chatbot won't start**
```bash
docker compose logs chatbot
```
Common cause: a missing or invalid `.env` value (especially `OPENAI_API_KEY`,
`API_KEY` under 32 characters, or `ALLOWED_ORIGINS=*`). The error message
in the logs will name the specific problem.

**Health check returns `degraded` or times out**
Wait 60 seconds after starting (first-time index loading), then retry. If it
persists, check logs for an OpenAI authentication error (bad/missing API key)
or a Redis connection error.

**`/v1/chat/` always returns 401 "Session cookie is missing"**
Your frontend isn't calling `POST /v1/session/` first, or isn't sending
requests with `credentials: "include"` (fetch) / `withCredentials: true`
(axios). See [Section 8](#8-frontend-integration).

**Bot doesn't know about content I just added**
Re-run `docker compose run --rm indexer` — content changes require an index
rebuild, they aren't picked up automatically.

**Port 8000 already in use**
Edit `docker-compose.yml`, change:
```yaml
"8000:8000"
```
to:
```yaml
"8080:8000"
```
then `docker compose up -d chatbot redis`, and update your reverse proxy /
frontend `API_BASE` accordingly.

**CORS errors in the browser console**
Confirm `ALLOWED_ORIGINS` in `.env` exactly matches the origin your website
is served from (including `https://` and any `www.` prefix — these count as
different origins), then `docker compose restart chatbot`.

---

## 14. Support Checklist

If you need to contact your developer, please have the output of these
three commands ready — it covers 90% of what's needed to diagnose an issue:

```bash
docker compose logs --tail=200 chatbot
docker compose ps
curl http://localhost:8000/v1/health/
```