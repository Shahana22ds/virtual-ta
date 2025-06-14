import os
from playwright.sync_api import sync_playwright

# Directory to save scraped TDS content
RAW_DIR = "data/raw/tds"
BASE_URL = "https://tds.s-anand.net"


def scrape_url(start_url):
    """
    Use Playwright to recursively navigate the Docsify site starting from start_url,
    extracting all visible sidebar links under aside.sidebar > div.sidebar-nav,
    scraping each page's main content into individual text files, and avoid revisiting URLs.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    visited = set()
    total = 0

    def visit(page, url):
        nonlocal total
        if url in visited:
            return
        visited.add(url)

        page.goto(url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("article.markdown-section#main")
        main = page.query_selector("article.markdown-section#main")
        content = main.text_content().strip().replace("Copy to clipboardErrorCopied", "\n")

        # Create a safe filename slug based on the URL's hash path
        # Extract hash path if exists, fallback to entire URL
        hash_path = ""
        if "#" in url:
            hash_part = url.split("#", 1)[1]
            hash_path = hash_part.lstrip("/").replace("/", "_")
        else:
            # fallback if no hash present
            hash_path = url.replace("://", "_").replace("/", "_")

        # Use the hash path or fallback for filename, prefix with a number to keep uniqueness
        filename = os.path.join(RAW_DIR, f"{hash_path}.txt")
        with open(filename, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        total += 1

        # Wait for sidebar-nav container
        page.wait_for_selector("aside.sidebar > div.sidebar-nav")

        # Collect all visible links in the sidebar
        links = page.query_selector_all("aside.sidebar > div.sidebar-nav a")
        for link in links:
            # Check if link is visible
            # if not link.is_visible():
            #     continue
            href = link.get_attribute("href")  # e.g. "#/README" or "#/../vscode"
            if not href or not href.startswith("#/"):
                continue
            # Construct full URL preserving hash routing
            next_url = f"{BASE_URL}/{href}"
            if next_url in visited:
                continue
            visit(page, next_url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        # Grant clipboard read and write permissions
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        visit(page, start_url)
        browser.close()

    print(f"✅ Scraped {total} pages into {RAW_DIR}")


if __name__ == "__main__":
    # Entrypoint expects a full URL
    scrape_url("https://tds.s-anand.net/#/2025-01/")
