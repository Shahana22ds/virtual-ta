# Virtual TA (TDS - Project 1 - Submission)

## Setup 

```
uv sync
```

## To run locally

```
uv run fastapi dev app/main.py
```

## To run the scraper

### TDS Scraper 

```
uv run python -m app.scrape_tds
```

### Discourse Scraper 

```
uv run python -m app.scrape_discourse
```

## To run vector db ingestion of embeddings from the scraped content

```
uv run python -m app.ingest
```

## To run ngrok to expose the locally running API server (not required normally)

```
ngrok http http://localhost:8000
```

## Testing the API locally using curl example

```
curl -sS -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Can I use Qdrant instead of ChromaDB?"}'
```

---

Environment variables that need to be setup in the .env file

```
QDRANT_URL=""
QDRANT_API_KEY=""
OPENAI_API_KEY=""
DISCOURSE_URL=""
DISCOURSE_SEARCH_FILTERS=""
DISCOURSE_COOKIE=""
```
