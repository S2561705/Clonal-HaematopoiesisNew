"""
Find the quietest showing of a Ticket Tailor event (e.g. The Odyssey at the
Glasgow Science Centre IMAX).

WHY PLAYWRIGHT, NOT requests/BeautifulSoup:
Ticket Tailor's booking widget renders dates/times and ticket-quantity
selectors via JavaScript after the page loads, and its checkout endpoints
are blocked by robots.txt for simple scrapers. A real (headless) browser
is the reliable way to get the numbers that actually appear on screen.

WHAT "QUIETEST" MEANS HERE:
Ticket Tailor shows "X tickets remaining" (or "Sold out") per ticket type
for each date/time slot, not a seat map. This script reads that number for
every showing of the event and reports whichever slot has the MOST tickets
still available (a good proxy for "fewest people already booked in" —
assuming all showings started with the same total allocation).

SETUP:
    pip install playwright
    playwright install chromium

USAGE:
    python quietest_showing.py

You will very likely need to tweak the CSS selectors marked "ADJUST ME"
below — Ticket Tailor's widget markup can change, and I couldn't load the
live site from this environment to verify them against the real DOM. Open
the event page in your own browser, right-click the date list / time list
/ availability text, choose "Inspect", and update the selectors to match.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass

from playwright.sync_api import sync_playwright

# ---- CONFIG -----------------------------------------------------------

# The Ticket Tailor "select date" widget URL for the event you want to check.
# (Found by right-clicking "Buy Tickets" on the event page and copying the link.)
EVENT_SELECT_DATE_URL = (
    "https://www.tickettailor.com/events/gscimax/2264253/select-date"
    "?modal_widget=true&widget=true"
)

# Be a polite scraper: pause between interactions so you're not hammering
# their server, and don't run this in a tight loop / on a schedule.
CLICK_DELAY_SECONDS = 1.5


@dataclass
class Showing:
    date_label: str
    time_label: str
    remaining_text: str
    remaining_count: int | None  # None if we couldn't parse a number

    def __str__(self):
        avail = (
            f"{self.remaining_count} left"
            if self.remaining_count is not None
            else self.remaining_text
        )
        return f"{self.date_label:<20} {self.time_label:<10} {avail}"


def parse_remaining(text: str) -> int | None:
    """Pull a number of remaining tickets out of strings like
    '12 tickets remaining' or '3 left'. Returns None for 'Sold out' etc."""
    if not text:
        return None
    if "sold out" in text.lower():
        return 0
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def scrape_showings(page) -> list[Showing]:
    showings: list[Showing] = []

    # ADJUST ME: selector for each clickable date tab/button in the widget.
    date_buttons = page.query_selector_all("[data-testid='date-picker-day']")
    if not date_buttons:
        # Fallback guess if Ticket Tailor uses a different structure/class.
        date_buttons = page.query_selector_all(".tt-date-picker button, .date-list button")

    print(f"Found {len(date_buttons)} date option(s) on the page.")

    for date_btn in date_buttons:
        date_label = date_btn.inner_text().strip()
        date_btn.click()
        time.sleep(CLICK_DELAY_SECONDS)

        # ADJUST ME: selector for each time slot shown after picking a date.
        time_buttons = page.query_selector_all("[data-testid='time-picker-slot']")
        if not time_buttons:
            time_buttons = page.query_selector_all(".tt-time-picker button, .time-list button")

        for time_btn in time_buttons:
            time_label = time_btn.inner_text().strip()
            time_btn.click()
            time.sleep(CLICK_DELAY_SECONDS)

            # ADJUST ME: selector for the "X tickets remaining" / "Sold out" text
            # that appears once a date+time is selected.
            avail_el = page.query_selector(
                "[data-testid='ticket-availability'], .tt-availability, .availability-text"
            )
            remaining_text = avail_el.inner_text().strip() if avail_el else "(not found)"

            showings.append(
                Showing(
                    date_label=date_label,
                    time_label=time_label,
                    remaining_text=remaining_text,
                    remaining_count=parse_remaining(remaining_text),
                )
            )

    return showings


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # A realistic user agent makes basic bot-detection less likely to
        # block or serve a stripped-down page.
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        # "networkidle" often never fires on pages with background polling
        # (analytics, chat widgets, etc.) and just times out. Wait for the
        # DOM instead, then give the widget's JS a few seconds to render.
        page.goto(EVENT_SELECT_DATE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)

        # Handy for debugging selectors: uncomment to see exactly what the
        # browser actually loaded before we start querying it.
        # page.screenshot(path="debug_page.png", full_page=True)
        # print(page.content())

        showings = scrape_showings(page)
        browser.close()

    if not showings:
        print(
            "No showings scraped — the selectors above almost certainly need "
            "adjusting. Open the URL in a normal browser, inspect the date "
            "buttons / time buttons / availability text, and update the "
            "'ADJUST ME' selectors in this script."
        )
        sys.exit(1)

    print("\nAll showings:")
    for s in showings:
        print(" ", s)

    ranked = sorted(
        showings,
        key=lambda s: (s.remaining_count is None, -(s.remaining_count or 0)),
    )
    print("\nQuietest (most tickets still available) first:")
    for s in ranked:
        print(" ", s)


if __name__ == "__main__":
    main()