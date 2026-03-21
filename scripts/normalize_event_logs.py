"""
One-time maintenance script to normalize legacy HTML-like audit event descriptions.

Usage:
  python scripts/normalize_event_logs.py          # dry-run (no DB writes)
  python scripts/normalize_event_logs.py --apply  # apply updates
"""
from __future__ import annotations

import argparse
import html
import re

from src.database.db import SessionLocal, SystemEvent


def sanitize_description(text: str) -> str:
    if not text:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_events(apply_changes: bool = False) -> tuple[int, int]:
    scanned = 0
    changed = 0

    with SessionLocal() as session:
        rows = session.query(SystemEvent).all()
        for row in rows:
            scanned += 1
            original = row.event_description or ""
            normalized = sanitize_description(original)
            if normalized != original:
                changed += 1
                if apply_changes:
                    row.event_description = normalized

        if apply_changes and changed:
            session.commit()

    return scanned, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize existing system event descriptions.")
    parser.add_argument("--apply", action="store_true", help="Persist normalized descriptions to DB.")
    args = parser.parse_args()

    scanned, changed = normalize_events(apply_changes=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned={scanned}, would_change={changed}" if not args.apply else f"[{mode}] scanned={scanned}, changed={changed}")


if __name__ == "__main__":
    main()
