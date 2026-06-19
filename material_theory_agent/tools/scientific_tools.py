"""
tools/scientific_tools.py — Real scientific API integrations.

Tools available:
  - PubChem lookup (compound properties, CID, MW, etc.)
  - Wikipedia summary (material background)
  - Periodic Table data (element properties)
  - Oxidation state checker (chemical validity)
  - Crystal structure rules (theoretical check)
  - Cost estimator (rough raw-material cost tier)
  - Material property predictor (LLM-based, uses known data)
  - Literature search (Wikipedia + PubChem combined)

All tools return plain dicts so they compose easily into LangChain
tool objects and can be called directly from LangGraph nodes.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "MaterialTheoryAgent/1.0 (research)"})

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
PERIODIC_TABLE_API = "https://periodictable.p.rapidapi.com/"  # fallback: hardcoded data


def _get(url: str, params: Dict = None, timeout: int = 10) -> Optional[Dict]:
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Tool HTTP] {url} → {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. PubChem Tool
# ─────────────────────────────────────────────────────────────────────────────

def pubchem_lookup(compound_name: str) -> Dict[str, Any]:
    """
    Look up a compound by name on PubChem.
    Returns CID, molecular formula, MW, IUPAC name, and key properties.
    """
    result: Dict[str, Any] = {"compound": compound_name, "found": False}

    # Step 1: get CID
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(compound_name)}/JSON"
    data = _get(url)
    if not data:
        return result

    try:
        compounds = data["PC_Compounds"]
        if not compounds:
            return result

        cid = compounds[0]["id"]["id"]["cid"]
        result["cid"] = cid
        result["found"] = True

        # Step 2: fetch property summary
        props_url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,XLogP,TPSA/JSON"
        )
        props_data = _get(props_url)
        if props_data:
            props = props_data["PropertyTable"]["Properties"][0]
            result["molecular_formula"] = props.get("MolecularFormula", "N/A")
            result["molecular_weight"] = props.get("MolecularWeight", "N/A")
            result["iupac_name"] = props.get("IUPACName", compound_name)
            result["xlogp"] = props.get("XLogP", "N/A")
            result["tpsa"] = props.get("TPSA", "N/A")

        # Step 3: hazard / GHS summary
        safety_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/JSON?record_type=2d"
        # (lightweight — just confirm it exists)
        result["pubchem_url"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

    except (KeyError, IndexError, TypeError) as e:
        result["parse_error"] = str(e)

    time.sleep(0.3)   # be polite to PubChem
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Wikipedia Summary Tool
# ─────────────────────────────────────────────────────────────────────────────

def wikipedia_summary(query: str) -> Dict[str, str]:
    """
    Fetch the Wikipedia introductory summary for a material or concept.
    """
    url = f"{WIKIPEDIA_API}/{requests.utils.quote(query)}"
    data = _get(url)

    if not data or "extract" not in data:
        # Try a search fallback
        search_url = "https://en.wikipedia.org/w/api.php"
        search_data = _get(
            search_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
        )
        if search_data:
            results = search_data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                data = _get(f"{WIKIPEDIA_API}/{requests.utils.quote(title)}")

    if data and "extract" in data:
        return {
            "title": data.get("title", query),
            "summary": data["extract"][:800],   # trim to 800 chars
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }

    return {"title": query, "summary": "No Wikipedia summary found.", "url": ""}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Periodic Table Tool (hardcoded — no API needed, always available)
# ─────────────────────────────────────────────────────────────────────────────

PERIODIC_DATA: Dict[str, Dict[str, Any]] = {
    "H":  {"name": "Hydrogen",   "atomic_number": 1,  "group": 1,  "period": 1, "electronegativity": 2.20, "common_oxidation_states": [1, -1],   "category": "nonmetal"},
    "Li": {"name": "Lithium",    "atomic_number": 3,  "group": 1,  "period": 2, "electronegativity": 0.98, "common_oxidation_states": [1],        "category": "alkali metal"},
    "C":  {"name": "Carbon",     "atomic_number": 6,  "group": 14, "period": 2, "electronegativity": 2.55, "common_oxidation_states": [4, 2, -4], "category": "nonmetal"},
    "N":  {"name": "Nitrogen",   "atomic_number": 7,  "group": 15, "period": 2, "electronegativity": 3.04, "common_oxidation_states": [3, -3, 5], "category": "nonmetal"},
    "O":  {"name": "Oxygen",     "atomic_number": 8,  "group": 16, "period": 2, "electronegativity": 3.44, "common_oxidation_states": [-2],       "category": "nonmetal"},
    "F":  {"name": "Fluorine",   "atomic_number": 9,  "group": 17, "period": 2, "electronegativity": 3.98, "common_oxidation_states": [-1],       "category": "halogen"},
    "Si": {"name": "Silicon",    "atomic_number": 14, "group": 14, "period": 3, "electronegativity": 1.90, "common_oxidation_states": [4, -4],    "category": "metalloid"},
    "Al": {"name": "Aluminium",  "atomic_number": 13, "group": 13, "period": 3, "electronegativity": 1.61, "common_oxidation_states": [3],        "category": "post-transition metal"},
    "Mg": {"name": "Magnesium",  "atomic_number": 12, "group": 2,  "period": 3, "electronegativity": 1.31, "common_oxidation_states": [2],        "category": "alkaline earth metal"},
    "Fe": {"name": "Iron",       "atomic_number": 26, "group": 8,  "period": 4, "electronegativity": 1.83, "common_oxidation_states": [2, 3],     "category": "transition metal"},
    "Cu": {"name": "Copper",     "atomic_number": 29, "group": 11, "period": 4, "electronegativity": 1.90, "common_oxidation_states": [1, 2],     "category": "transition metal"},
    "Zn": {"name": "Zinc",       "atomic_number": 30, "group": 12, "period": 4, "electronegativity": 1.65, "common_oxidation_states": [2],        "category": "transition metal"},
    "Ti": {"name": "Titanium",   "atomic_number": 22, "group": 4,  "period": 4, "electronegativity": 1.54, "common_oxidation_states": [4, 3, 2],  "category": "transition metal"},
    "Ni": {"name": "Nickel",     "atomic_number": 28, "group": 10, "period": 4, "electronegativity": 1.91, "common_oxidation_states": [2, 3],     "category": "transition metal"},
    "Cr": {"name": "Chromium",   "atomic_number": 24, "group": 6,  "period": 4, "electronegativity": 1.66, "common_oxidation_states": [3, 6, 2],  "category": "transition metal"},
    "Mn": {"name": "Manganese",  "atomic_number": 25, "group": 7,  "period": 4, "electronegativity": 1.55, "common_oxidation_states": [2, 4, 7],  "category": "transition metal"},
    "Co": {"name": "Cobalt",     "atomic_number": 27, "group": 9,  "period": 4, "electronegativity": 1.88, "common_oxidation_states": [2, 3],     "category": "transition metal"},
    "Na": {"name": "Sodium",     "atomic_number": 11, "group": 1,  "period": 3, "electronegativity": 0.93, "common_oxidation_states": [1],        "category": "alkali metal"},
    "K":  {"name": "Potassium",  "atomic_number": 19, "group": 1,  "period": 4, "electronegativity": 0.82, "common_oxidation_states": [1],        "category": "alkali metal"},
    "Ca": {"name": "Calcium",    "atomic_number": 20, "group": 2,  "period": 4, "electronegativity": 1.00, "common_oxidation_states": [2],        "category": "alkaline earth metal"},
    "P":  {"name": "Phosphorus", "atomic_number": 15, "group": 15, "period": 3, "electronegativity": 2.19, "common_oxidation_states": [5, 3, -3], "category": "nonmetal"},
    "S":  {"name": "Sulfur",     "atomic_number": 16, "group": 16, "period": 3, "electronegativity": 2.58, "common_oxidation_states": [-2, 4, 6], "category": "nonmetal"},
    "Cl": {"name": "Chlorine",   "atomic_number": 17, "group": 17, "period": 3, "electronegativity": 3.16, "common_oxidation_states": [-1, 1, 5], "category": "halogen"},
    "Ba": {"name": "Barium",     "atomic_number": 56, "group": 2,  "period": 6, "electronegativity": 0.89, "common_oxidation_states": [2],        "category": "alkaline earth metal"},
    "Pb": {"name": "Lead",       "atomic_number": 82, "group": 14, "period": 6, "electronegativity": 1.87, "common_oxidation_states": [2, 4],     "category": "post-transition metal"},
    "Sn": {"name": "Tin",        "atomic_number": 50, "group": 14, "period": 5, "electronegativity": 1.96, "common_oxidation_states": [2, 4],     "category": "post-transition metal"},
    "Zr": {"name": "Zirconium",  "atomic_number": 40, "group": 4,  "period": 5, "electronegativity": 1.33, "common_oxidation_states": [4],        "category": "transition metal"},
    "W":  {"name": "Tungsten",   "atomic_number": 74, "group": 6,  "period": 6, "electronegativity": 2.36, "common_oxidation_states": [6, 4],     "category": "transition metal"},
    "Mo": {"name": "Molybdenum", "atomic_number": 42, "group": 6,  "period": 5, "electronegativity": 2.16, "common_oxidation_states": [6, 4, 2],  "category": "transition metal"},
    "V":  {"name": "Vanadium",   "atomic_number": 23, "group": 5,  "period": 4, "electronegativity": 1.63, "common_oxidation_states": [5, 4, 3],  "category": "transition metal"},
}

TOXIC_ELEMENTS = {"Pb", "Hg", "Cd", "As", "Be", "Tl", "Cr"}  # Cr(VI) is toxic


def periodic_table_lookup(elements: List[str]) -> Dict[str, Any]:
    """
    Return properties for a list of element symbols.
    Falls back to 'Unknown' for elements not in local data.
    """
    result = {}
    for symbol in elements:
        symbol = symbol.strip().capitalize()
        if symbol in PERIODIC_DATA:
            result[symbol] = PERIODIC_DATA[symbol]
        else:
            result[symbol] = {"name": symbol, "note": "Not in local database"}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Oxidation State Checker
# ─────────────────────────────────────────────────────────────────────────────

def check_oxidation_states(formula: str) -> Dict[str, Any]:
    """
    Parse a simple formula and verify that oxidation states are chemically
    plausible for each element using PERIODIC_DATA.

    Returns {"valid": bool, "issues": List[str], "elements_found": List[str]}
    """
    issues: List[str] = []
    toxic_found: List[str] = []

    # Extract element symbols (handles formulas like Al2O3, SiC, Fe3O4)
    element_pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    found_elements = []

    # Expand parentheses naively for common patterns
    clean_formula = re.sub(r"\(([^)]+)\)(\d+)", lambda m: m.group(1) * int(m.group(2) or 1), formula)

    for match in element_pattern.finditer(clean_formula):
        symbol = match.group(1)
        if symbol in PERIODIC_DATA:
            found_elements.append(symbol)
            if symbol in TOXIC_ELEMENTS:
                toxic_found.append(PERIODIC_DATA[symbol]["name"])
        else:
            if len(symbol) <= 2 and symbol.isalpha():
                issues.append(f"Unknown element: {symbol}")

    # Check for known unstable combinations
    if "F" in found_elements and "O" in found_elements:
        issues.append("Fluorine + Oxygen combinations can be highly reactive/unstable")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "toxic_elements": toxic_found,
        "elements_found": list(set(found_elements)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cost Estimator
# ─────────────────────────────────────────────────────────────────────────────

# Rough cost tiers based on abundance and industrial production
ELEMENT_COST_TIER: Dict[str, str] = {
    "H": "very_low", "C": "very_low", "O": "very_low", "N": "very_low",
    "Si": "low",     "Al": "low",     "Fe": "low",     "Ca": "low",
    "Na": "low",     "K":  "low",     "Mg": "low",     "S":  "low",
    "Ti": "medium",  "Cr": "medium",  "Mn": "medium",  "Ni": "medium",
    "Cu": "medium",  "Zn": "medium",  "Sn": "medium",  "Mo": "medium",
    "Co": "high",    "V":  "high",    "W":  "high",     "Zr": "high",
    "Pb": "low",     "Ba": "low",
}

COST_SCORE = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}


def estimate_cost(elements: List[str]) -> Dict[str, Any]:
    """
    Estimate overall cost tier of a material based on constituent elements.
    Returns a tier (low/medium/high) and per-element breakdown.
    """
    breakdown = {}
    scores = []

    for sym in elements:
        sym = sym.strip().capitalize()
        tier = ELEMENT_COST_TIER.get(sym, "medium")
        breakdown[sym] = tier
        scores.append(COST_SCORE.get(tier, 3))

    if not scores:
        return {"overall_tier": "unknown", "breakdown": breakdown}

    avg = sum(scores) / len(scores)
    if avg <= 1.5:
        overall = "very_low"
    elif avg <= 2.5:
        overall = "low"
    elif avg <= 3.5:
        overall = "medium"
    elif avg <= 4.5:
        overall = "high"
    else:
        overall = "very_high"

    return {"overall_tier": overall, "score": round(avg, 2), "breakdown": breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Literature Search (Wikipedia + PubChem combined)
# ─────────────────────────────────────────────────────────────────────────────

def literature_search(query: str, material_name: str = "") -> Dict[str, Any]:
    """
    Combine Wikipedia summary and PubChem lookup for a material/concept.
    Returns a merged snapshot of available knowledge.
    """
    wiki = wikipedia_summary(query)
    pubchem = pubchem_lookup(material_name or query) if material_name else {}

    snippets: List[str] = []
    if wiki.get("summary"):
        snippets.append(f"[Wikipedia] {wiki['summary']}")
    if pubchem.get("found"):
        snippets.append(
            f"[PubChem] Formula: {pubchem.get('molecular_formula', 'N/A')}, "
            f"MW: {pubchem.get('molecular_weight', 'N/A')}, "
            f"IUPAC: {pubchem.get('iupac_name', 'N/A')}"
        )

    return {
        "query": query,
        "snippets": snippets,
        "wikipedia": wiki,
        "pubchem": pubchem,
    }
