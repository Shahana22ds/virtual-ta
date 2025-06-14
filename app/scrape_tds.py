import os
from playwright.sync_api import sync_playwright

# Directory to save scraped TDS content
RAW_DIR = "data/raw/tds"
BASE_URL = "https://tds.s-anand.net"


def scrape_month(year_month: str = "2025-01"):
    """
    Use Playwright to navigate the Docsify site for the given month,
    extract all sidebar links under aside.sidebar > div.sidebar-nav,
    and scrape each page's main content into individual text files.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    # Initial landing URL with hash
    start_url = f"{BASE_URL}/#/{year_month}/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        # Grant clipboard read and write permissions
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.goto(start_url)
        # Wait for sidebar-nav container
        page.wait_for_selector("aside.sidebar > div.sidebar-nav")

        # Collect all links in the sidebar
        links = page.query_selector_all("aside.sidebar > div.sidebar-nav a")
        total = 0

        for link in links:
            href = link.get_attribute("href")  # e.g. "#/README" or "#/../vscode"
            if not href or not href.startswith("#/"):
                continue
            # Construct full URL preserving hash routing
            page_url = f"{BASE_URL}/{href}"
            # Create a safe filename slug based on the hash path
            slug = href.lstrip("#/").replace("/", "_")
            filename = os.path.join(RAW_DIR, f"{year_month}_{slug}.txt")

            # Navigate and scrape
            page.goto(page_url)
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("article.markdown-section#main")
            main = page.query_selector("article.markdown-section#main")
            content = main.text_content().strip().replace("Copy to clipboardErrorCopied", "\n")
            # Write content explicitly with UTF-8 encoding to avoid encoding issues
            with open(filename, "w", encoding="utf-8", errors="replace") as f:
                f.write(content)
            total += 1

        browser.close()
    print(f"✅ Scraped {total} pages for {year_month} into {RAW_DIR}")


if __name__ == "__main__":
    # Scrape only January 2025
    scrape_month("2025-01")