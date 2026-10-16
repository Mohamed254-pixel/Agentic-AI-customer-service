"""Local product photos with source records and a neutral missing-photo state."""

from base64 import b64encode
from functools import lru_cache
from html import escape
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets/products"


@lru_cache(maxsize=1)
def photo_sources():
    return json.loads((ASSETS / "sources.json").read_text())


@lru_cache(maxsize=128)
def product_thumbnail(name):
    """Use an explicit name match, never an unrelated photo guessed by keywords."""
    record = photo_sources().get(name)
    if record is None:
        return '<span class="product-image photo-missing" role="img" aria-label="Photo unavailable" title="Photo unavailable">—</span>'
    data = (ASSETS / record["file"]).read_bytes()
    encoded = b64encode(data).decode()
    return (
        f'<a class="product-photo-link" href="{escape(record["source"], quote=True)}" '
        'target="_blank" rel="noopener noreferrer" title="Representative product photo · View source">'
        f'<img class="product-image" src="data:image/jpeg;base64,{encoded}" '
        f'alt="{escape(name, quote=True)} — representative product photo"/></a>'
    )
