"""
svg_utils.py
------------
SVG cleaning, validation, and rendering utilities for the data pipeline.

Cleaning pipeline (applied in order):
  1. Strip XML comments (<!-- ... -->)
  2. Strip <?xml?> processing instructions
  3. Strip <metadata>, <desc>, <title> blocks
  4. Extract <svg>...</svg> content (discard anything outside)
  5. Round decimal numbers to N decimal places
  6. Collapse whitespace
  7. Validate: is it valid XML?
  8. Length filter: discard if < min_length_chars characters
  9. Deduplicate by MD5 hash (managed externally via seen_hashes set)
"""

import re
import hashlib
from typing import Optional

from lxml import etree


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import)
# ---------------------------------------------------------------------------
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_PI = re.compile(r"<\?xml[^?]*\?>", re.IGNORECASE)
_RE_METADATA_BLOCKS = re.compile(
    r"<(?:metadata|desc|title)\b[^>]*>.*?</(?:metadata|desc|title)>",
    re.DOTALL | re.IGNORECASE,
)
_RE_SVG_CONTENT = re.compile(r"(<svg\b.*</svg>)", re.DOTALL | re.IGNORECASE)
_RE_FLOAT = re.compile(r"-?\d+\.\d+")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{2,}")


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def _strip_comments(svg: str) -> str:
    return _RE_COMMENT.sub("", svg)


def _strip_processing_instructions(svg: str) -> str:
    return _RE_PI.sub("", svg)


def _strip_metadata_blocks(svg: str) -> str:
    return _RE_METADATA_BLOCKS.sub("", svg)


def _extract_svg_root(svg: str) -> Optional[str]:
    """Return only the <svg>...</svg> portion, or None if not found."""
    m = _RE_SVG_CONTENT.search(svg)
    return m.group(1) if m else None


def _round_floats(svg: str, decimal_places: int = 1) -> str:
    """Round all floating-point numbers in the SVG string."""
    def _round_match(m: re.Match) -> str:
        val = float(m.group())
        rounded = round(val, decimal_places)
        # Format: suppress trailing zeros for integers (e.g. 2.0 → "2")
        if decimal_places == 0:
            return str(int(rounded))
        fmt = f"{rounded:.{decimal_places}f}"
        # Strip unnecessary trailing zeros: "1.10" → "1.1", "1.00" → "1"
        fmt = fmt.rstrip("0").rstrip(".")
        return fmt
    return _RE_FLOAT.sub(_round_match, svg)


def _collapse_whitespace(svg: str) -> str:
    svg = _RE_MULTI_SPACE.sub(" ", svg)
    svg = _RE_MULTI_NEWLINE.sub("\n", svg)
    return svg.strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_xml(svg: str) -> bool:
    """Return True if the string parses as valid XML via lxml (strict)."""
    try:
        etree.fromstring(svg.encode("utf-8"))
        return True
    except etree.XMLSyntaxError:
        return False


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_svg(
    raw_svg: str,
    decimal_places: int = 1,
    min_length_chars: int = 50,
    seen_hashes: Optional[set] = None,
) -> tuple[Optional[str], str]:
    """
    Clean a single raw SVG string through the full pipeline.

    Parameters
    ----------
    raw_svg : str
        The raw SVG string (may include XML declarations, comments, etc.)
    decimal_places : int
        Number of decimal places to round floats to.
    min_length_chars : int
        Minimum character length after cleaning; shorter SVGs are discarded.
    seen_hashes : set or None
        If provided, used for deduplication. Hashes of accepted SVGs are
        added to this set by the caller.

    Returns
    -------
    (cleaned_svg_or_None, reason_string)
        If cleaning succeeds, returns (cleaned_svg, "ok").
        If it fails at any step, returns (None, reason).
    """
    svg = raw_svg

    # Step 1: Strip comments
    svg = _strip_comments(svg)

    # Step 2: Strip processing instructions
    svg = _strip_processing_instructions(svg)

    # Step 3: Strip metadata blocks
    svg = _strip_metadata_blocks(svg)

    # Step 4: Extract <svg>...</svg>
    svg = _extract_svg_root(svg)
    if svg is None:
        return None, "no_svg_root"

    # Step 5: Round floats
    svg = _round_floats(svg, decimal_places=decimal_places)

    # Step 6: Collapse whitespace
    svg = _collapse_whitespace(svg)

    # Step 7: Validate XML
    if not is_valid_xml(svg):
        return None, "invalid_xml"

    # Step 8: Length filter
    if len(svg) < min_length_chars:
        return None, "too_short"

    # Step 9: Deduplication (caller manages the set; we just check)
    if seen_hashes is not None:
        h = md5_hash(svg)
        if h in seen_hashes:
            return None, "duplicate"
        seen_hashes.add(h)

    return svg, "ok"


# ---------------------------------------------------------------------------
# Batch cleaning
# ---------------------------------------------------------------------------

def clean_svg_batch(
    raw_svgs: list[str],
    decimal_places: int = 1,
    min_length_chars: int = 50,
    deduplicate: bool = True,
) -> tuple[list[str], dict]:
    """
    Clean a list of raw SVG strings.

    Returns
    -------
    (cleaned_svgs, stats_dict)
    """
    seen_hashes: Optional[set] = set() if deduplicate else None
    cleaned = []
    stats = {
        "total_input": len(raw_svgs),
        "no_svg_root": 0,
        "invalid_xml": 0,
        "too_short": 0,
        "duplicate": 0,
        "ok": 0,
    }

    for raw in raw_svgs:
        result, reason = clean_svg(
            raw,
            decimal_places=decimal_places,
            min_length_chars=min_length_chars,
            seen_hashes=seen_hashes,
        )
        if result is not None:
            cleaned.append(result)
            stats["ok"] += 1
        else:
            stats[reason] = stats.get(reason, 0) + 1

    return cleaned, stats


# ---------------------------------------------------------------------------
# Rendering (optional — requires cairosvg)
# ---------------------------------------------------------------------------

def render_svg_to_png(svg: str, output_size: int = 256) -> Optional[bytes]:
    """
    Render an SVG string to a PNG bytes object.
    Returns None if rendering fails (e.g., unsupported features).

    Requires: cairosvg
    """
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=output_size)
    except Exception:
        return None


def is_renderable(svg: str) -> bool:
    """Return True if the SVG can be rendered by cairosvg without error."""
    return render_svg_to_png(svg) is not None
