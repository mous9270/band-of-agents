"""Deterministic tools for the Procurement & Sourcing Agent (Agent 3).

The strategic reasoning (which feedstocks matter, GO/NO-GO, the executive
narrative) is done by the LLM, but the *facts and the money* should not be
hallucinated:

* ``lookup_supply_chain`` plays the role of a procurement market-data feed. It
  returns a grounded, **reproducible** availability/price/lead-time profile for
  a material — from a small curated catalog of common feedstocks, with a
  deterministic hash-based fallback for anything not in the catalog (so the same
  material always yields the same numbers, like a real database would).

* ``compute_roi`` turns the manufacturing cost (from Agent 2) plus a target
  price and volume into reproducible ROI arithmetic.

Both return plain dicts so they can be registered as pydantic-ai tools in the
standalone pipeline *and* reused by the Band-facing agent.
"""

from __future__ import annotations

import hashlib

# Curated "market database": indicative supply profiles for common feedstocks.
# price_usd_per_kg, lead_time_weeks, suppliers, availability, note.
# Numbers are illustrative but realistic in relative terms (e.g. platinum is
# expensive and concentrated; silica sand is abundant and cheap).
_CATALOG: dict[str, dict] = {
    "platinum": {"price": 31000.0, "lead": 10, "suppliers": 4, "availability": "SCARCE",
                 "note": "Supply concentrated in South Africa/Russia; PGM price volatility."},
    "palladium": {"price": 38000.0, "lead": 10, "suppliers": 4, "availability": "SCARCE",
                  "note": "PGM, Russia-concentrated supply; substitution risk."},
    "silicon": {"price": 3.0, "lead": 4, "suppliers": 40, "availability": "ABUNDANT",
                "note": "Metallurgical-grade widely available; China-dominant refining."},
    "silica": {"price": 0.1, "lead": 2, "suppliers": 80, "availability": "ABUNDANT",
               "note": "Sand-derived, effectively unlimited supply."},
    "carbon": {"price": 1.5, "lead": 3, "suppliers": 60, "availability": "ABUNDANT",
               "note": "Graphite/carbon black broadly sourced."},
    "copper": {"price": 9.5, "lead": 6, "suppliers": 30, "availability": "AVAILABLE",
               "note": "Liquid commodity market; price tracks LME."},
    "aluminum": {"price": 2.6, "lead": 4, "suppliers": 35, "availability": "ABUNDANT",
                 "note": "Energy-intensive smelting; otherwise plentiful."},
    "lithium": {"price": 70.0, "lead": 16, "suppliers": 8, "availability": "CONSTRAINED",
                "note": "EV-driven demand; Australia/Chile concentrated."},
    "cobalt": {"price": 33.0, "lead": 14, "suppliers": 6, "availability": "CONSTRAINED",
               "note": "DRC-concentrated; ESG/sourcing scrutiny."},
    "nickel": {"price": 18.0, "lead": 8, "suppliers": 20, "availability": "AVAILABLE",
               "note": "Indonesia-led supply growth; class-1 vs class-2 split."},
    "gallium": {"price": 600.0, "lead": 20, "suppliers": 3, "availability": "SCARCE",
                "note": "China export controls; few alternative sources."},
    "germanium": {"price": 1400.0, "lead": 20, "suppliers": 3, "availability": "SCARCE",
                  "note": "Byproduct supply; export-control exposure."},
    "gold": {"price": 75000.0, "lead": 6, "suppliers": 15, "availability": "AVAILABLE",
             "note": "Deep market but high carrying cost."},
    "tungsten": {"price": 45.0, "lead": 14, "suppliers": 6, "availability": "CONSTRAINED",
                 "note": "China-dominant; strategic/critical mineral."},
    "titanium": {"price": 12.0, "lead": 10, "suppliers": 12, "availability": "AVAILABLE",
                 "note": "Sponge supply concentrated; aerospace demand."},
    "oxygen": {"price": 0.2, "lead": 1, "suppliers": 100, "availability": "ABUNDANT",
               "note": "Industrial gas, locally sourced."},
    "hydrogen": {"price": 6.0, "lead": 2, "suppliers": 25, "availability": "AVAILABLE",
                 "note": "Grey vs green price spread."},
    "platinum catalyst": {"price": 45000.0, "lead": 12, "suppliers": 3, "availability": "SCARCE",
                          "note": "Pt-loaded catalyst; PGM exposure plus fabrication."},
    "siloxane": {"price": 8.0, "lead": 6, "suppliers": 15, "availability": "AVAILABLE",
                 "note": "Silicone precursor; tied to silicon-metal supply."},
}

_AVAILABILITY_RISK = {
    "ABUNDANT": "LOW",
    "AVAILABLE": "LOW",
    "CONSTRAINED": "MEDIUM",
    "SCARCE": "HIGH",
    "UNAVAILABLE": "HIGH",
}


def _stable_unit(material: str) -> float:
    """Deterministic float in [0, 1) derived from the material name."""
    digest = hashlib.sha256(material.lower().strip().encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def lookup_supply_chain(material: str) -> dict:
    """Look up the supply-chain profile for one feedstock / raw material.

    Call this once for every key feedstock the material requires. Returns a
    grounded, reproducible profile: market price (USD/kg), lead time (weeks),
    number of qualified suppliers, an availability tier
    (ABUNDANT / AVAILABLE / CONSTRAINED / SCARCE / UNAVAILABLE), a supply-risk
    rating, and a sourcing note. Use the returned numbers verbatim — do not
    invent your own prices or lead times.
    """
    key = material.lower().strip()
    record = None
    # Exact match, then substring match against the curated catalog.
    if key in _CATALOG:
        record = _CATALOG[key]
    else:
        for name, rec in _CATALOG.items():
            if name in key or key in name:
                record = rec
                break

    if record is None:
        # Deterministic synthetic profile for unknown feedstocks.
        u = _stable_unit(material)
        price = round(2.0 + u * 500.0, 2)  # $2 .. $502 / kg
        lead = int(2 + u * 22)             # 2 .. 24 weeks
        suppliers = int(2 + (1 - u) * 38)  # 2 .. 40 suppliers
        if suppliers >= 25:
            availability = "ABUNDANT"
        elif suppliers >= 12:
            availability = "AVAILABLE"
        elif suppliers >= 5:
            availability = "CONSTRAINED"
        else:
            availability = "SCARCE"
        note = "Not in curated catalog; profile estimated from a deterministic model."
        record = {"price": price, "lead": lead, "suppliers": suppliers,
                  "availability": availability, "note": note}

    return {
        "material": material,
        "availability": record["availability"],
        "market_price_usd_per_kg": record["price"],
        "estimated_lead_time_weeks": record["lead"],
        "suppliers_identified": record["suppliers"],
        "supply_risk": _AVAILABILITY_RISK[record["availability"]],
        "notes": record["note"],
    }


def compute_roi(
    manufacturing_cost_usd_per_kg: float,
    target_selling_price_usd_per_kg: float,
    annual_production_volume_kg: float,
    upfront_investment_usd: float,
    horizon_years: int = 5,
) -> dict:
    """Compute grounded, reproducible product ROI from Agent 2's cost figure.

    Pass the manufacturing cost (USD/kg) from the VirtualLabReport, a target
    selling price, the planned annual volume (kg), and the upfront capital
    investment. Returns gross margin %, annual revenue/gross profit, payback
    period (years), and ROI % over the horizon. Use the returned numbers
    verbatim in the proposal — do not recompute them yourself.
    """
    price = float(target_selling_price_usd_per_kg)
    cost = float(manufacturing_cost_usd_per_kg)
    volume = float(annual_production_volume_kg)
    upfront = float(upfront_investment_usd)
    horizon = max(1, int(horizon_years))

    gross_margin_per_kg = price - cost
    gross_margin_percent = (gross_margin_per_kg / price * 100.0) if price > 0 else 0.0
    annual_revenue = price * volume
    annual_gross_profit = gross_margin_per_kg * volume

    if annual_gross_profit > 0:
        payback_period_years = round(upfront / annual_gross_profit, 2)
    else:
        payback_period_years = -1.0  # never pays back at this price/volume

    if upfront > 0:
        roi_percent = (annual_gross_profit * horizon - upfront) / upfront * 100.0
    else:
        roi_percent = float("inf") if annual_gross_profit > 0 else 0.0

    return {
        "manufacturing_cost_usd_per_kg": round(cost, 2),
        "target_selling_price_usd_per_kg": round(price, 2),
        "gross_margin_percent": round(gross_margin_percent, 1),
        "annual_production_volume_kg": volume,
        "annual_revenue_usd": round(annual_revenue, 2),
        "annual_gross_profit_usd": round(annual_gross_profit, 2),
        "upfront_investment_usd": round(upfront, 2),
        "payback_period_years": payback_period_years,
        "roi_percent": round(roi_percent, 1) if roi_percent != float("inf") else 1e9,
        "horizon_years": horizon,
    }
