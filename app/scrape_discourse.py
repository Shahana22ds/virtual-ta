import os
import requests
from datetime import datetime
from app.config import settings
import json
import logging
import time

# Directory to save scraped Discourse posts
RAW_DIR = "data/raw/discourse"
LDJSON_DIR = os.path.join(RAW_DIR, "line_delimited")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def parse_post_from_json(p):
    topic_id = p["topic_id"]
    url = p.get("url", "")
    topic_slug = None
    if url and "/t/" in url:
        try:
            topic_slug = url.split("/t/")[1].split("/")[0]
        except Exception:
            topic_slug = None
    return p["id"], p, topic_id, topic_slug


def load_existing_posts(filename):
    existing_posts_map = {}
    topics_map = {}
    all_posts = []
    # Load posts from full JSON file
    if os.path.exists(filename):
        logging.info(f"Loading existing posts from {filename} to avoid re-fetching.")
        with open(filename, "r", encoding="utf-8") as f:
            loaded_posts = json.load(f)
            for p in loaded_posts:
                post_id, post_obj, topic_id, topic_slug = parse_post_from_json(p)
                all_posts.append(post_obj)
                existing_posts_map[post_id] = post_obj
                topics_map[topic_id] = topics_map.get(topic_id, None) or topic_slug
        logging.info(f"Loaded {len(all_posts)} posts from existing file.")

    # Also load posts from line-delimited JSON files to avoid refetching if script crashed.
    if os.path.exists(LDJSON_DIR):
        logging.info(f"Loading existing posts from line-delimited files in {LDJSON_DIR} to avoid re-fetching.")
        for fname in os.listdir(LDJSON_DIR):
            if fname.endswith(".ldjson"):
                path = os.path.join(LDJSON_DIR, fname)
                with open(path, "r", encoding="utf-8") as fld:
                    for line in fld:
                        if not line.strip():
                            continue
                        try:
                            p = json.loads(line)
                        except Exception:
                            logging.warning(f"Failed to parse line in {path}, skipping.")
                            continue
                        post_id, post_obj, topic_id, topic_slug = parse_post_from_json(p)
                        if post_id and post_id not in existing_posts_map:
                            all_posts.append(post_obj)
                            existing_posts_map[post_id] = post_obj
                            topics_map[topic_id] = topics_map.get(topic_id, None) or topic_slug
        logging.info(f"Loaded total {len(all_posts)} posts including line-delimited files.")
    return existing_posts_map, topics_map, all_posts


def get_auth_headers():
    if settings.discourse_cookie:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) Gecko/20100101 Firefox/140.0",
            "Cookie": settings.discourse_cookie
        }
        logging.info("Using discourse cookie for authentication.")
        return headers
    else:
        logging.error("Discourse credentials not configured. Set DISCOURSE_API_KEY and DISCOURSE_API_USERNAME, or DISCOURSE_COOKIE.")
        raise ValueError(
            "Discourse credentials not configured. Set DISCOURSE_API_KEY and DISCOURSE_API_USERNAME, or DISCOURSE_COOKIE."
        )


def build_topic_map(topics, topics_map):
    for topic in topics:
        if topic["id"] not in topics_map:
            topics_map[topic["id"]] = topic["slug"]
    logging.debug(f"Updated topics map with {len(topics)} entries.")


def format_post_url(base_url, topic_slug, topic_id, post_number):
    if post_number == 1:
        return f"{base_url}/t/{topic_slug}/{topic_id}"
    else:
        return f"{base_url}/t/{topic_slug}/{topic_id}/{post_number}"


def build_post_entry(p, topics_map):
    base_url = settings.discourse_url.rstrip('/')
    topic_slug = topics_map.get(p["topic_id"], "unknown-topic")
    topic_id = p["topic_id"]
    post_number = p["post_number"]
    url = format_post_url(base_url, topic_slug, topic_id, post_number)
    post_entry = {
        "id": p["id"],
        "username": p["username"],
        "created_at": p["created_at"],
        "post_number": post_number,
        "topic_id": topic_id,
        "url": url,
        "raw": p["raw"],
        "replies": p["replies"]
    }
    return post_entry


def save_posts(filename, all_posts, topics_map, total):
    logging.info(f"Saving {total} posts to {filename}.")
    output = []
    for p in all_posts:
        post_entry = build_post_entry(p, topics_map)
        output.append(post_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ Scraped {total} posts into {filename}")


def append_post_ldjson(post, topics_map, ldjson_file_handle):
    post_entry = build_post_entry(post, topics_map)
    ldjson_file_handle.write(json.dumps(post_entry, ensure_ascii=False) + "\n")
    ldjson_file_handle.flush()


def fetch_topic_posts(topic_id, headers):
    while True:
        topic_resp = requests.get(
            f"{settings.discourse_url.rstrip('/')}/t/{topic_id}.json",
            params={},
            headers=headers
        )
        if topic_resp.status_code == 422:
            try:
                error_data = topic_resp.json()
                errors = error_data.get("errors", [])
                if any("too many times" in e for e in errors):
                    logging.warning("Received 422 Too Many Requests error. Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    continue
            except Exception:
                # If parsing JSON fails, re-raise
                topic_resp.raise_for_status()
        topic_resp.raise_for_status()
        topic_data = topic_resp.json()
        post_ids = topic_data.get("post_stream", {}).get("stream", [])
        logging.debug(f"Topic {topic_id} has {len(post_ids)} posts in stream.")
        return post_ids


def fetch_post_details(post_id, headers):
    post_url = f"{settings.discourse_url.rstrip('/')}/posts/{post_id}.json"
    while True:
        logging.debug(f"Fetching post {post_id}")
        post_resp = requests.get(post_url, headers=headers)
        if post_resp.status_code == 422:
            try:
                error_data = post_resp.json()
                errors = error_data.get("errors", [])
                if any("too many times" in e for e in errors):
                    logging.warning("Received 422 Too Many Requests error while fetching post. Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    continue
            except Exception:
                post_resp.raise_for_status()
        post_resp.raise_for_status()
        post_data = post_resp.json()

        replies_url = f"{settings.discourse_url.rstrip('/')}/posts/{post_id}/replies.json"
        while True:
            logging.debug(f"Fetching replies for post {post_id}")
            replies_resp = requests.get(replies_url, headers=headers)
            if replies_resp.status_code == 422:
                try:
                    error_data = replies_resp.json()
                    errors = error_data.get("errors", [])
                    if any("too many times" in e for e in errors):
                        logging.warning("Received 422 Too Many Requests error while fetching replies. Waiting 60 seconds before retrying...")
                        time.sleep(60)
                        continue
                except Exception:
                    replies_resp.raise_for_status()
            replies_resp.raise_for_status()
            replies_data = replies_resp.json()
            replies_texts = [reply.get("cooked", "") for reply in replies_data]
            logging.debug(f"Found {len(replies_texts)} replies for post {post_id}")
            break

        return post_data, replies_texts


def process_topic_posts(post_ids, existing_post_ids, start_dt, end_dt, all_posts, total, headers, ldjson_file_handle, topics_map):
    for post_id in post_ids:
        if post_id in existing_post_ids:
            logging.info(f"Post {post_id} already exists in local data, skipping fetch.")
            continue

        post_data, replies_texts = fetch_post_details(post_id, headers)        
        post_data["replies"] = replies_texts

        created_at_post = datetime.fromisoformat(post_data["created_at"]).replace(tzinfo=None)
        if created_at_post < start_dt or created_at_post > end_dt:
            continue
        all_posts.append(post_data)
        total += 1
        logging.info(f"Added post {post_id} created at {post_data['created_at']}")
        # Append immediately to line delimited json file
        append_post_ldjson(post_data, topics_map, ldjson_file_handle)
    return all_posts, total


def scrape_discourse(start_date: str = "2025-01-01", end_date: str = "2025-04-14"):
    """
    Scrape Discourse posts between start_date and end_date (inclusive).  
    Uses API key/username if configured; otherwise falls back to session cookie.
    Saves JSON with posts correlated with topics, including generated URLs.

    start_date, end_date: ISO format dates (YYYY-MM-DD)
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LDJSON_DIR, exist_ok=True)
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    filename = os.path.join(RAW_DIR, f"posts_{start_date}_to_{end_date}.json")
    ldjson_filename = os.path.join(LDJSON_DIR, f"posts_{start_date}_to_{end_date}.ldjson")

    existing_posts_map, topics_map, all_posts = load_existing_posts(filename)

    headers = get_auth_headers()

    page = 1
    total = len(all_posts)

    base_url = settings.discourse_url.rstrip('/')

    # Open the line-delimited file in append mode once
    with open(ldjson_filename, "a", encoding="utf-8") as ldjson_file_handle:

        while True:
            query = f"{settings.discourse_search_filters} after:{start_date} before:{end_date}"
            url = f"{base_url}/search.json"
            headers["Referer"] = url
            logging.info(f"Requesting page {page} of search results with query: {query}")
            resp = requests.get(
                url,
                params={"q": query, "page": page},
                headers=headers
            )
            if resp.status_code == 422:
                try:
                    error_data = resp.json()
                    errors = error_data.get("errors", [])
                    if any("too many times" in e for e in errors):
                        logging.warning("Received 422 Too Many Requests error on search. Waiting 60 seconds before retrying...")
                        time.sleep(60)
                        continue
                except Exception:
                    resp.raise_for_status()
            resp.raise_for_status()

            data = resp.json()
            posts = data.get("posts", [])
            topics = data.get("topics", [])
            logging.info(f"Page {page}: Found {len(posts)} posts and {len(topics)} topics.")
            if not posts:
                logging.info("No more posts found, ending pagination.")
                break

            build_topic_map(topics, topics_map)

            for post in posts:
                created_at = datetime.fromisoformat(post["created_at"]).replace(tzinfo=None)

                if created_at < start_dt:
                    logging.info(f"Reached posts before start date {start_date}, saving collected posts.")
                    save_posts(filename, all_posts, topics_map, total)
                    return

                if created_at <= end_dt:
                    topic_id = post.get("topic_id")
                    topic_posts_in_existing = [p for p in all_posts if p.get("topic_id") == topic_id]
                    existing_post_ids = {p["id"] for p in topic_posts_in_existing}

                    post_ids = fetch_topic_posts(topic_id, headers)
                    all_posts, total = process_topic_posts(post_ids, existing_post_ids, start_dt, end_dt, all_posts, total, headers, ldjson_file_handle, topics_map)

            page += 1
            logging.info(f"Moving to next page: {page}")

    logging.info(f"All pages processed. Saving total {total} collected posts.")
    save_posts(filename, all_posts, topics_map, total)


if __name__ == "__main__":
    # Example: scrape from Jan 1 to Apr 14, 2025
    scrape_discourse(start_date="2025-01-01", end_date="2025-04-14")
