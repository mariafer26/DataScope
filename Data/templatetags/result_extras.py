from __future__ import annotations
from typing import Any, Dict, List
from django import template

register = template.Library()

def _is_list_of_dicts(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)

def _is_list_of_lists(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, (list, tuple)) for x in v)

@register.filter
def is_tabular(v: Any) -> bool:
    return _is_list_of_dicts(v) or _is_list_of_lists(v)

@register.filter
def as_table(v: Any) -> Dict[str, List[Any]]:
    if _is_list_of_dicts(v):
        headers: List[str] = []
        seen = set()
        for row in v:
            for k in row.keys():
                if k not in seen:
                    seen.add(k); headers.append(str(k))
        rows = [[row.get(h, "") for h in headers] for row in v]
        return {"headers": headers, "rows": rows}
    if _is_list_of_lists(v):
        maxc = max((len(r) for r in v), default=0)
        headers = [f"Col{i+1}" for i in range(maxc)]
        rows = [list(r) + [""]*(maxc-len(r)) for r in v]
        return {"headers": headers, "rows": rows}
    return {"headers": ["Valor"], "rows": [[v]]}

@register.filter
def to_display_text(v: Any) -> str:
    import json
    if isinstance(v, str): 
        return v
    try:
        return json.dumps(v, ensure_ascii=False, indent=2)
    except Exception:
        return str(v)

