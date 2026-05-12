# Cloudflare Worker API Proxy

This folder contains the Cloudflare Worker used by the React app as a small API gateway for sentiment inference.

The Worker receives browser requests from the React app, applies CORS, adds the private API key on the server side, and forwards the request to one of two model APIs:

| React target | Model | Upstream endpoint variable |
| --- | --- | --- |
| `api1` | `BAAI/bge-m3` | `BGE_M3_API_URL` |
| `api2` | `Qwen/Qwen3-4B-Instruct-2507` | `QWEN_API_URL` |

The browser should call the Worker, not the model APIs directly. This keeps `TEXT_CLASSIFICATION_API_KEY` out of the React bundle.

## Worker Behavior

`main.js` exposes a single `fetch` handler.

```text
POST https://<worker-domain>/?api=api1
POST https://<worker-domain>/?api=api2
```

Request body:

```json
{
  "text": "this is a good movie!"
}
```

BGE-m3 expected response:

```json
{
  "predicted_class_id": 2,
  "probability": [0.01, 0.04, 0.95]
}
```

Qwen expected response:

```json
{
  "text": "this is a good movie!",
  "label": "positive",
  "raw_output": "positive"
}
```

The Worker accepts `OPTIONS` preflight requests and `POST` prediction requests. Any other method returns `405`.

## Create the Worker

Install Wrangler if you do not already have it:

```bash
npm install -g wrangler
wrangler login
```

Create a Worker project:

```bash
npm create cloudflare@latest sentiment-classification-worker
```

Choose:

- `Worker only`
- JavaScript
- No framework
- No Git initialization if this repo already tracks your files

Replace the generated Worker entry file with this repository file:

```text
cloudflare/main.js
```

In the Cloudflare project, set the entry point to `main.js`. If you use `wrangler.toml`, the important part is:

```toml
name = "sentiment-analysis-classification"
main = "main.js"
compatibility_date = "2026-05-12"

[vars]
BGE_M3_API_URL = "https://<bge-m3-api-host>/predict"
QWEN_API_URL = "https://<qwen-api-host>/predict"
```

Set the API key as a secret:

```bash
wrangler secret put TEXT_CLASSIFICATION_API_KEY
```

Then deploy:

```bash
wrangler deploy
```

Wrangler prints the Worker URL after deployment, for example:

```text
https://sentiment-analysis-classification.<your-subdomain>.workers.dev
```

## Configure CORS

Edit `allowedOrigins` in `main.js` when you add a new frontend domain.

Current allowed origins:

```js
[
  "https://ahmed-nezar.github.io",
  "http://localhost:5173",
  "http://127.0.0.1:5173",
]
```

Use the localhost origins for Vite development and the GitHub Pages origin for production.

## Test the Worker

Test BGE-m3:

```bash
curl -X POST "https://<worker-domain>/?api=api1" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"this is a good movie!\"}"
```

Test Qwen:

```bash
curl -X POST "https://<worker-domain>/?api=api2" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"this is a bad movie!\"}"
```

If `api` is missing, the Worker defaults to `api1`.

## React Integration

The React app reads one public Worker URL from the root `.env` file:

```env
TEXT_CLASSIFICATION_URL="https://<worker-domain>/"
```

The React inference page maps each selected model to the Worker query parameter:

```ts
api1 -> BAAI/bge-m3
api2 -> Qwen/Qwen3-4B-Instruct-2507
```

The fetch request is:

```ts
const url = new URL(TEXT_CLASSIFICATION_URL, window.location.origin)
url.searchParams.set('api', selectedModel.apiTarget)

await fetch(url.toString(), {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ text }),
})
```

The BGE-m3 result panel displays probability scores. The Qwen result panel displays the returned `label`, the input `text`, and `raw_output`, because the decoder endpoint does not return probabilities.

This means React only needs the Worker URL. The Worker decides which upstream API to call and attaches the private bearer token.

## Local Development

Run the React app:

```bash
cd sentimentApp
npm run dev
```

Open:

```text
http://localhost:5173
```

Run the Worker locally from the `cloudflare` folder:

```bash
wrangler dev main.js
```

When testing a local Worker, set:

```env
TEXT_CLASSIFICATION_URL="http://127.0.0.1:8787/"
```

## Troubleshooting

- `400 Invalid API target`: use `?api=api1` or `?api=api2`.
- `405 Method not allowed`: send a `POST` request.
- Browser CORS error: add the React app origin to `allowedOrigins`.
- `401` or `403` from the model API: check `TEXT_CLASSIFICATION_API_KEY`.
- Network error from the Worker: check `BGE_M3_API_URL` and `QWEN_API_URL`.
