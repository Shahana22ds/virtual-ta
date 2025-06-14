import os
import logging
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

# Directory to save scraped TDS content
RAW_DIR = "data/raw/tds"
BASE_URL = "https://tds.s-anand.net"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def scrape_url(start_url):
    """
    Use Playwright to recursively navigate the Docsify site starting from start_url,
    extracting all visible sidebar links under aside.sidebar > div.sidebar-nav,
    scraping each page's main content into individual text files, and avoid revisiting URLs.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    visited = set()
    total = 0

    def normalize_url(url):
        # Normalize URLs to ignore "?id=..." query parameters for revisiting avoidance
        parsed = urlparse(url)
        # Drop the query string to treat anchors with different ?id= as same page
        base_hash = parsed.fragment.split('?', 1)[0] if parsed.fragment else ""
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}#{base_hash}"
        return normalized

    def visit(page, url):
        nonlocal total
        normalized_url = normalize_url(url)
        if normalized_url in visited:
            logging.debug(f"Skipping already visited URL: {normalized_url}")
            return
        logging.info(f"Visiting: {url}")
        visited.add(normalized_url)

        page.goto(url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("article.markdown-section#main")
        main = page.query_selector("article.markdown-section#main")
        content = main.text_content().strip().replace("Copy to clipboardErrorCopied", "\n")

        # Create a safe filename slug based on the URL's hash path without query parameters
        hash_path = ""
        if "#" in url:
            hash_part = url.split("#", 1)[1]
            # Remove query parameters in hash part for filename uniqueness
            hash_part = hash_part.split('?', 1)[0]
            hash_path = hash_part.lstrip("/").replace("/", "_")
        else:
            # fallback if no hash present
            hash_path = url.replace("://", "_").replace("/", "_")

        filename = os.path.join(RAW_DIR, f"{hash_path}.txt")
        with open(filename, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        total += 1
        logging.info(f"Saved content to: {filename}")

        # Wait for sidebar-nav container
        page.wait_for_selector("aside.sidebar > div.sidebar-nav")

        # Collect all visible links in the sidebar
        links = page.query_selector_all("aside.sidebar > div.sidebar-nav a")
        logging.debug(f"Found {len(links)} sidebar links on page.")
        for link in links:
            href = link.get_attribute("href")  # e.g. "#/README" or "#/../vscode"
            if not href or not href.startswith("#/"):
                logging.debug(f"Ignoring link with href: {href}")
                continue
            next_url = f"{BASE_URL}/{href}"
            normalized_next_url = normalize_url(next_url)
            if normalized_next_url in visited:
                logging.debug(f"Already visited next URL: {normalized_next_url}")
                continue
            visit(page, next_url)

    logging.info(f"Starting scraping from: {start_url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        # Grant clipboard read and write permissions
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        visit(page, start_url)
        browser.close()
    logging.info(f"✅ Scraped {total} pages into {RAW_DIR}")


if __name__ == "__main__":
    # Entrypoint expects a full URL
    scrape_url("https://tds.s-anand.net/#/2025-01/")
