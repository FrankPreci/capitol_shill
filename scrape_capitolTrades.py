# This script scrapes trade data from Capitol Trades for the last 365 days. It uses Playwright to navigate the site, handle 
# pagination, and extract trade details. The results are filtered to include only trades from the last 365 days and saved to a 
# JSON file. The script includes error handling and logging for transparency. 
from playwright.sync_api import sync_playwright
import json
import time
import random
from datetime import datetime, timedelta


def scrape_capitol_trades_365d():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("Starting scraper for the last 365 days...")

        current_page = 1
        # 365 days will have more pages, but we set 100 to be safe.
        # The script will auto-stop when it hits an empty page.
        max_pages = 100
        time_frame = "365" # In days

        while current_page <= max_pages:
            # UPDATED URL: Added 'txDate=365d' to filter results
            url = f"https://www.capitoltrades.com/trades?txDate={time_frame}d&pageSize=96&page={current_page}"
            print(f"Scraping {url} ...")

            try:
                page.goto(url)

                # Handle Cookie Banner (First page only)
                if current_page == 1:
                    try:
                        cookie_button = page.get_by_role("button", name="Accept All")
                        if cookie_button.is_visible(timeout=3000):
                            cookie_button.click()
                            print("Accepted cookies.")
                    except:
                        pass

                # Wait for table data
                try:
                    page.wait_for_selector("tbody tr", timeout=8000)
                except:
                    print("Timed out waiting for data. Reached the end or page is empty.")
                    break

                # Extract Rows
                rows = page.locator("tbody tr").all()
                row_count = len(rows)
                print(f"  - Found {row_count} trades.")

                # STOP CONDITION: If no rows are found, we are done.
                if row_count == 0:
                    print("No trades found on this page. Stopping.")
                    break

                for row in rows:
                    cells = row.locator("td").all()
                    if len(cells) < 8: continue

                    try:
                        trade_data = {
                            "politician": cells[0].inner_text().split('\n')[0],
                            "party_state": cells[0].inner_text().split('\n')[1] if '\n' in cells[
                                0].inner_text() else "",
                            "issuer": cells[1].inner_text().split('\n')[0],
                            "ticker": cells[1].inner_text().split('\n')[1] if '\n' in cells[1].inner_text() else "",
                            "pub_date": cells[2].inner_text(),
                            "trade_date": cells[3].inner_text(),
                            "filed_after": cells[4].inner_text().replace("days", "").strip(),
                            "owner": cells[5].inner_text(),
                            "type": cells[6].inner_text(),
                            "size": cells[7].inner_text(),
                            "price": cells[8].inner_text()
                        }
                        results.append(trade_data)
                    except Exception:
                        continue

                current_page += 1
                time.sleep(random.uniform(1.0, 2.0))

            except Exception as e:
                print(f"Error on page {current_page}: {e}")
                break

        browser.close()

    # Filter results to only include trades from the last 365 days
    three_six_five_days_ago = datetime.now() - timedelta(days=365)
    filtered_results = []

    for trade in results:
        try:
            # Parse the trade date (assuming format like "Mar 15, 2024" or similar)
            trade_date_str = trade.get('trade_date', '')
            if trade_date_str:
                # Try to parse various date formats that might appear on the site
                try:
                    trade_date = datetime.strptime(trade_date_str, '%b %d, %Y')
                except ValueError:
                    try:
                        trade_date = datetime.strptime(trade_date_str, '%m/%d/%Y')
                    except ValueError:
                        # If we can't parse the date, skip this trade
                        continue

                # Only keep trades from the last 365 days
                if trade_date >= three_six_five_days_ago:
                    filtered_results.append(trade)
        except Exception:
            # If there's any issue parsing, skip this trade
            continue

    print(f"Filtered {len(results)} total trades to {len(filtered_results)} trades within 365 days.")

    # Save to JSON
    # if json file already exists, we will overwrite it with the new data. In a real application, you might want to append or handle this differently.
    filename = "capitol_trades_365d.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, indent=4)

    print(f"Done! Saved {len(filtered_results)} trades within the 365-day timeframe.")


if __name__ == "__main__":
    scrape_capitol_trades_365d()