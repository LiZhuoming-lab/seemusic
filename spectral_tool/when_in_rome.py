from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi
import pandas as pd

WHEN_IN_ROME_REPO = "MarkGotham/When-in-Rome"
WHEN_IN_ROME_BRANCH = "master"
WHEN_IN_ROME_WEB_URL = f"https://github.com/{WHEN_IN_ROME_REPO}"
WHEN_IN_ROME_TREE_API_URL = (
    f"https://api.github.com/repos/{WHEN_IN_ROME_REPO}/git/trees/{WHEN_IN_ROME_BRANCH}?recursive=1"
)
WHEN_IN_ROME_SUPPORTED_EXTENSIONS = (".musicxml", ".xml", ".mxl", ".mid", ".midi", ".krn")


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "seemusic-when-in-rome-loader",
        },
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=30) as response:
        return json.load(response)


def _humanize_segment(value: str) -> str:
    return value.replace("_", " ").strip()


def _entry_from_tree_path(path: str) -> dict[str, str]:
    parts = Path(path).parts
    corpus_parts = list(parts[1:-1]) if len(parts) >= 2 else []
    category = corpus_parts[0] if len(corpus_parts) >= 1 else ""
    composer = corpus_parts[1] if len(corpus_parts) >= 2 else ""
    collection = corpus_parts[2] if len(corpus_parts) >= 3 else ""
    item_parts = corpus_parts[3:] if len(corpus_parts) >= 4 else []
    item = " / ".join(_humanize_segment(part) for part in item_parts) if item_parts else ""
    score_name = Path(path).name
    title_parts = [_humanize_segment(part) for part in corpus_parts[1:]]
    display_name = " / ".join(part for part in title_parts if part)
    raw_url = (
        f"https://raw.githubusercontent.com/{WHEN_IN_ROME_REPO}/{WHEN_IN_ROME_BRANCH}/"
        f"{urllib.parse.quote(path, safe='/')}"
    )
    return {
        "path": path,
        "category": category,
        "category_label": _humanize_segment(category),
        "composer": composer,
        "composer_label": _humanize_segment(composer),
        "collection": collection,
        "collection_label": _humanize_segment(collection),
        "item": item,
        "display_name": display_name or _humanize_segment(Path(path).stem),
        "score_name": score_name,
        "extension": Path(path).suffix.lower(),
        "raw_url": raw_url,
    }


def list_when_in_rome_scores() -> pd.DataFrame:
    tree_payload = _fetch_json(WHEN_IN_ROME_TREE_API_URL)
    rows = [
        _entry_from_tree_path(item["path"])
        for item in tree_payload.get("tree", [])
        if item.get("type") == "blob"
        and str(item.get("path", "")).startswith("Corpus/")
        and str(item.get("path", "")).lower().endswith(WHEN_IN_ROME_SUPPORTED_EXTENSIONS)
    ]
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    return catalog.sort_values(
        ["category_label", "composer_label", "collection_label", "display_name", "score_name"]
    ).reset_index(drop=True)


def download_when_in_rome_score(path: str) -> tuple[bytes, str]:
    entry = _entry_from_tree_path(path)
    request = urllib.request.Request(
        entry["raw_url"],
        headers={"User-Agent": "seemusic-when-in-rome-loader"},
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=60) as response:
        data = response.read()
    return data, entry["score_name"]
