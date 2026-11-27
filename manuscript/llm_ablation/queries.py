# queries.py -- domain-tagged query sets for the correlation study.
# A query set maps domain name -> list of prompts, so we can identify which
# activations correlate with which domain. Sources:
#   "domains"        all built-in domains (config.DOMAIN_QUERY_SETS)
#   "<domain name>"  one built-in domain (needs --rank-by usage|gate_prob,
#                    correlation needs >= 2 domains for contrast)
#   path/to/file.txt [domain] header sections, one prompt per line
from typing import Dict, List

from config import DOMAIN_QUERY_SETS


def load_domains(query_set: str, max_queries: int = 16) -> Dict[str, List[str]]:
    """Return {domain: prompts} with at most `max_queries` prompts per domain."""
    if query_set == "domains":
        domains = {d: list(q) for d, q in DOMAIN_QUERY_SETS.items()}
    elif query_set in DOMAIN_QUERY_SETS:
        domains = {query_set: list(DOMAIN_QUERY_SETS[query_set])}
    else:
        domains = _parse_file(query_set)
    out = {d: qs[:max_queries] for d, qs in domains.items() if qs}
    if not out:
        raise ValueError(f"Query set '{query_set}' is empty.")
    return out


def _parse_file(path: str) -> Dict[str, List[str]]:
    domains: Dict[str, List[str]] = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                domains.setdefault(current, [])
            elif current is not None:
                domains[current].append(line)
            else:
                raise ValueError(f"{path}: prompt before any [domain] header: {line!r}")
    return domains


def flatten(domains: Dict[str, List[str]]) -> List[str]:
    return [q for qs in domains.values() for q in qs]
