"""Fetch a small real financial document corpus for the RAG pipeline.

Pulls excerpts from two public, auth-free sources:
  - SEC EDGAR (10-K / 10-Q filings for a handful of large-cap tickers),
    via the stable data.sec.gov submissions API + sec.gov/Archives documents.
  - The Federal Reserve's public FOMC statement archive.

Saves raw text excerpts to data/corpus/*.txt with a metadata.json index.
Falls back to a small set of hand-curated excerpts (defined below) for any
source that fails to fetch, so the corpus is always reproducible offline.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
USER_AGENT = "FinSight-research contact@example.com"

# (name, CIK) for recognisable large-cap companies
SEC_COMPANIES = [
    ("Apple Inc.", "0000320193", "AAPL"),
    ("Microsoft Corporation", "0000789019", "MSFT"),
    ("JPMorgan Chase & Co.", "0000019617", "JPM"),
    ("Tesla, Inc.", "0001318605", "TSLA"),
]

_ANCHORS = [
    r"item\s+1a\.?\s+risk\s+factors",
    r"item\s+7\.?\s+management.?s\s+discussion",
    r"risk\s+factors",
]

# Hand-curated fallback excerpts — used only if live fetching fails.
# Short, publicly-known statements paraphrasing each company's well-documented
# risk disclosures / the Fed's standard statement language, clearly marked as
# a fallback (not a verbatim filing) in metadata.
_SEC_FALLBACK = {
    "AAPL": (
        "Apple Inc. — Risk Factors (paraphrased fallback excerpt). "
        "The Company's business, reputation, results of operations, and "
        "financial condition can be adversely affected by global and "
        "regional economic conditions, including inflation, interest rate "
        "fluctuations, and recessionary conditions. The Company faces "
        "substantial competition in highly competitive smartphone, "
        "personal computer, and digital content markets. Global markets "
        "for the Company's products and services are highly competitive "
        "and subject to rapid technological change."
    ),
    "MSFT": (
        "Microsoft Corporation — Risk Factors (paraphrased fallback "
        "excerpt). Microsoft's business is subject to risks from "
        "competition in the cloud computing, software, and AI markets. "
        "Cybersecurity incidents and data breaches could harm Microsoft's "
        "reputation and business. Microsoft's substantial investments in "
        "AI infrastructure, including Azure data centers, may not produce "
        "the returns anticipated."
    ),
    "JPM": (
        "JPMorgan Chase & Co. — Risk Factors (paraphrased fallback "
        "excerpt). JPMorgan Chase's businesses and earnings are affected "
        "by capital and credit market conditions, interest rate changes, "
        "and regulatory developments. Credit risk, market risk, and "
        "operational risk are inherent in the Firm's businesses. Adverse "
        "macroeconomic conditions could increase loan delinquencies and "
        "charge-offs."
    ),
    "TSLA": (
        "Tesla, Inc. — Risk Factors (paraphrased fallback excerpt). "
        "Tesla's business depends on its ability to design, manufacture, "
        "and deliver vehicles at scale and on time. Tesla faces "
        "significant competition in the automotive and energy storage "
        "industries. Changes to government incentives for electric "
        "vehicles could materially affect demand."
    ),
}

_FOMC_FALLBACK = (
    "Federal Open Market Committee Statement (paraphrased fallback "
    "excerpt). Recent indicators suggest that economic activity has "
    "continued to expand at a solid pace. The Committee seeks to achieve "
    "maximum employment and inflation at the rate of 2 percent over the "
    "longer run. In support of these goals, the Committee decided to "
    "maintain the target range for the federal funds rate. The Committee "
    "will continue to monitor the implications of incoming information "
    "for the economic outlook."
)


def _http_get(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="ignore")


def _strip_html(raw_html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_excerpt(text: str, max_chars: int = 2500) -> str:
    """Find the LAST occurrence of a section anchor (the first is usually
    the table-of-contents entry, which has no real prose after it) and
    require a reasonable amount of alphabetic text following it."""
    for pattern in _ANCHORS:
        matches = list(re.finditer(pattern, text, flags=re.I))
        for match in reversed(matches):
            candidate = text[match.start() : match.start() + max_chars].strip()
            letters = sum(c.isalpha() for c in candidate)
            if letters > max_chars * 0.6:
                return candidate
    # fall back to a mid-document slice to avoid cover-page boilerplate
    mid = len(text) // 3
    return text[mid : mid + max_chars].strip()


def fetch_sec_filings(limit_per_company: int = 3) -> list[dict]:
    """Fetch recent 10-K/10-Q excerpts for each company in SEC_COMPANIES."""
    records: list[dict] = []
    for name, cik, ticker in SEC_COMPANIES:
        try:
            submissions = json.loads(
                _http_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            )
            recent = submissions["filings"]["recent"]
            count = 0
            for i, form in enumerate(recent["form"]):
                if form not in ("10-K", "10-Q") or count >= limit_per_company:
                    continue
                accession = recent["accessionNumber"][i].replace("-", "")
                doc = recent["primaryDocument"][i]
                filing_date = recent["filingDate"][i]
                cik_int = str(int(cik))
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{accession}/{doc}"
                )
                html = _http_get(url, timeout=20.0)
                text = _strip_html(html)
                excerpt = _extract_excerpt(text)
                if len(excerpt) < 200:
                    continue
                records.append(
                    {
                        "filename": f"sec_{ticker.lower()}_{form.lower().replace('-', '')}_{filing_date}.txt",
                        "text": excerpt,
                        "source": "SEC EDGAR",
                        "company": name,
                        "ticker": ticker,
                        "document_type": form,
                        "date": filing_date,
                        "url": url,
                        "fallback": False,
                    }
                )
                count += 1
                time.sleep(0.3)  # be polite to SEC's rate limits
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as exc:
            print(f"  WARN: live fetch failed for {name} ({exc}); using fallback excerpt")
            records.append(
                {
                    "filename": f"sec_{ticker.lower()}_fallback.txt",
                    "text": _SEC_FALLBACK[ticker],
                    "source": "SEC EDGAR (fallback — paraphrased, not verbatim)",
                    "company": name,
                    "ticker": ticker,
                    "document_type": "10-K (paraphrased excerpt)",
                    "date": "n/a",
                    "url": "https://www.sec.gov/cgi-bin/browse-edgar",
                    "fallback": True,
                }
            )
    return records


def fetch_fomc_statements(target_count: int = 10) -> list[dict]:
    """Fetch recent FOMC statement excerpts from the Fed's public archive."""
    records: list[dict] = []
    try:
        archive_html = _http_get(
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        )
        links = sorted(
            set(
                re.findall(
                    r'href="(/newsevents/pressreleases/monetary\d{8}a\.htm)"',
                    archive_html,
                )
            ),
            reverse=True,
        )
        for link in links[:target_count]:
            date_match = re.search(r"monetary(\d{8})a\.htm", link)
            date = date_match.group(1) if date_match else "unknown"
            url = f"https://www.federalreserve.gov{link}"
            try:
                statement_html = _http_get(url)
                text = _strip_html(statement_html)
                anchor = re.search(r"for release at", text, flags=re.I)
                excerpt = (
                    text[anchor.start() : anchor.start() + 2500].strip()
                    if anchor
                    else text[-2500:].strip()
                )
                if len(excerpt) < 200:
                    continue
                records.append(
                    {
                        "filename": f"fomc_statement_{date}.txt",
                        "text": excerpt,
                        "source": "Federal Reserve FOMC press release archive",
                        "company": "Federal Reserve",
                        "ticker": "n/a",
                        "document_type": "FOMC Statement",
                        "date": date,
                        "url": url,
                        "fallback": False,
                    }
                )
                time.sleep(0.2)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                continue
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  WARN: FOMC archive page fetch failed ({exc})")

    if len(records) < target_count:
        missing = target_count - len(records)
        print(f"  WARN: only {len(records)} FOMC statements fetched live; "
              f"padding with {missing} fallback excerpt(s)")
        for i in range(missing):
            records.append(
                {
                    "filename": f"fomc_statement_fallback_{i}.txt",
                    "text": _FOMC_FALLBACK,
                    "source": "Federal Reserve (fallback — paraphrased, not verbatim)",
                    "company": "Federal Reserve",
                    "ticker": "n/a",
                    "document_type": "FOMC Statement (paraphrased excerpt)",
                    "date": "n/a",
                    "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    "fallback": True,
                }
            )
    return records[:target_count]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching SEC EDGAR filing excerpts...")
    sec_records = fetch_sec_filings()
    print(f"  {len(sec_records)} SEC excerpts")

    print("Fetching FOMC statement excerpts...")
    fomc_records = fetch_fomc_statements()
    print(f"  {len(fomc_records)} FOMC excerpts")

    all_records = sec_records + fomc_records
    metadata = []
    for rec in all_records:
        out_path = CORPUS_DIR / rec["filename"]
        out_path.write_text(rec["text"], encoding="utf-8")
        metadata.append({k: v for k, v in rec.items() if k != "text"})

    metadata_path = CORPUS_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_size = sum((CORPUS_DIR / m["filename"]).stat().st_size for m in metadata)
    print(f"\nWrote {len(metadata)} documents ({total_size / 1024:.1f} KB) to {CORPUS_DIR}")
    print(f"Metadata index: {metadata_path}")


if __name__ == "__main__":
    main()
