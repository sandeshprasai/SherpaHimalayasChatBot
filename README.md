# Sherpa Chatbot — Setup & Deployment Guide

## Requirements

You only need **Docker** installed on your server. Nothing else — no Python, no pip, no virtual environments.

- Install Docker: https://docs.docker.com/engine/install/
- Verify installation:
  ```bash
  docker --version
  docker compose version
  ```

---

## Step 1 — Extract the Delivery Package

```bash
unzip chatbot-delivery.zip
cd chatbot-delivery
```

---

## Step 2 — Set Up Your Environment

```bash
cp .env.example .env
nano .env
```

Fill in the following required values:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key — get one free at https://console.groq.com |
| `API_KEY` | A strong random password to protect your chatbot API — generate one at https://randomkeygen.com |
| `ALLOWED_ORIGINS` | Your website URL e.g. `https://yourwebsite.com` |
| `COMPANY_NAME` | Your company name |
| `COMPANY_EMAIL` | Your company email |

Save and close the file (`Ctrl+X` → `Y` → `Enter` in nano).

---

## Step 3 — Load the Docker Image

```bash
docker load < SherpaChatbot.tar.gz
```

This may take 1–2 minutes. You should see:
```
Loaded image: chatbot:latest
```

---

## Step 4 — Restore the Chatbot Data

```bash
docker volume create sherpa-chatbot_chatbot_data

docker run --rm \
  -v sherpa-chatbot_chatbot_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/chatbot_indexes.tar.gz -C /data
```

This restores the pre-built knowledge base into Docker.

---

## Step 5 — Start the Chatbot

```bash
docker compose up -d chatbot
```

---

## Step 6 — Verify It Is Running

```bash
curl http://localhost:8000/health/
```

You should see:

```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "environment": "production"
}
```

> **Note:** Wait up to 60 seconds after starting — the AI model needs time to warm up.

---

## Step 7 — Test a Chat Message

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"message": "Hello, what services do you offer?"}'
```

Replace `YOUR_API_KEY_HERE` with the `API_KEY` value you set in Step 2.

---

## Daily Management

| Task | Command |
|---|---|
| View live logs | `docker compose logs -f chatbot` |
| Stop the chatbot | `docker compose down` |
| Start the chatbot | `docker compose up -d chatbot` |
| Restart the chatbot | `docker compose restart chatbot` |
| Check status | `docker compose ps` |

---

## Troubleshooting

### Chatbot won't start
Check the logs for errors:
```bash
docker compose logs chatbot
```

### Health check returns `unhealthy`
Wait 60 seconds after starting — the AI model needs time to warm up — then check again:
```bash
curl http://localhost:8000/health/
```

### Port 8000 already in use
Edit `docker-compose.yml` and change the port mapping from:
```yaml
"8000:8000"
```
to:
```yaml
"8080:8000"
```
Then restart:
```bash
docker compose up -d chatbot
```

---

## Making It Public (Recommended)

The chatbot runs on port `8000` by default. To make it accessible via your domain with HTTPS, you need a reverse proxy in front of it. Contact your developer to configure:

- **Nginx** reverse proxy pointing to `localhost:8000`
- **SSL certificate** via Let's Encrypt (free)
- **DNS** pointing your domain to this server

---

## Support

For technical issues, contact your developer with the output of:

```bash
docker compose logs chatbot
docker compose ps
curl http://localhost:8000/health/
```
