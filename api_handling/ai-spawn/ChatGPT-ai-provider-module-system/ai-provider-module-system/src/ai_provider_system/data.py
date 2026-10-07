import json


def serialize_data(data):
    """Serialize supported input without silently stringifying unknown objects."""
    if isinstance(data, str):
        return data
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    # Pandas is optional; avoid loading it for ordinary text/JSON inputs.
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        # split preserves columns, index, duplicate column names and row order.
        return data.to_json(orient="split", force_ascii=False, date_format="iso")
    raise TypeError("data must be text, a JSON-compatible dict/list, or a pandas DataFrame")


def split_text(text, limit):
    """Lossless character chunks, preferring whitespace boundaries."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    chunks = []
    while len(text) > limit:
        boundary = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        end = boundary + 1 if boundary >= limit // 2 else limit
        chunks.append(text[:end])
        text = text[end:]
    if text:
        chunks.append(text)
    return chunks
