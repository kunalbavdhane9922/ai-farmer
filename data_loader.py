"""
data_loader.py
──────────────
Loads and parses the local 'all_agent_data.json' Knowledge Bundle.

Responsibilities:
  1. Read & validate the JSON file.
  2. Extract each data section (ML Engine, Soil Health Card, Weather, Market/Policy).
  3. Return the Top-5 crops for downstream agent analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
#  File I/O
# ──────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / "all_agent_data.json"


def load_json(filepath: Path = DATA_FILE) -> dict:
    """Load and return the full JSON knowledge bundle.

    Args:
        filepath: Path to the JSON file (defaults to all_agent_data.json
                  in the same directory as this script).

    Returns:
        Parsed dictionary from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Structural validation and schema adaptation
    if "knowledge_bundle" not in data:
        print("[DataLoader] ⚠️ 'knowledge_bundle' not found. Adapting new schema format...")
        # Adapt new schema to the expected internal format
        kb = {}
        meta = data.get("Metadata", {})
        kb["metadata"] = {
            "region": meta.get("Location", "Unknown").split(", ")[-1] if ", " in meta.get("Location", "") else meta.get("Location", "Unknown"),
            "district": meta.get("Location", "Unknown").split(", ")[0],
            "season": "Kharif 2026",
            "soil_type": data.get("Soil_Health_Card", {}).get("soil_type", "Unknown"),
            "irrigation_source": "Unknown"
        }
        
        ml_data = data.get("ML_Engine_Data", [])
        top5 = []
        for c in ml_data:
            # Map "Rice" to "Paddy (IR-64)" to match CANONICAL_CROPS for consensus
            crop_name = "Paddy (IR-64)" if c.get("crop") == "Rice" else c.get("crop")
            top5.append({
                "rank": c.get("rank"),
                "crop_name": crop_name,
                "crop_code": crop_name.lower().replace(" ", "_"),
                "suitability_score": c.get("suitability", 0) / 100.0,
                "predicted_yield_tons_per_ha": 5.0, # Default
                "growing_period_days": 120, # Default
                "water_requirement_mm": 1000, # Default
                "confidence_interval": c.get("confidence", 0.0)
            })
        kb["ml_engine_crop_suitability"] = {"top_5_crops": top5}
        
        shc = data.get("Soil_Health_Card", {})
        micros = {}
        for k, v in shc.get("micronutrients", {}).items():
            micros[f"{k}_ppm"] = {
                "value": 1.0,
                "rating": v,
                "critical_limit": 1.0,
                "recommendation": f"Manage {k} according to {v} status."
            }
        macros = shc.get("macronutrients", {})
        kb["soil_health_card"] = {
            "macronutrients": {
                "nitrogen_kg_per_ha": {"value": 200, "rating": macros.get("N", "Medium")},
                "phosphorus_kg_per_ha": {"value": 20, "rating": macros.get("P", "Medium")},
                "potassium_kg_per_ha": {"value": 200, "rating": macros.get("K", "Medium")}
            },
            "micronutrients": micros
        }
        
        wf = data.get("Weather_API_7_Day_Forecast", {})
        kb["weather_forecast_2day"] = {
            "agro_advisory": {
                "sowing_window_status": "Open",
                "pest_alert": "None",
                "irrigation_advisory": "Standard"
            },
            "daily_forecast": [
                {
                    "day_label": "Today",
                    "date": "2026-04-25",
                    "temperature": {"max_celsius": wf.get("avg_temp_c", 30), "min_celsius": wf.get("avg_temp_c", 30) - 5},
                    "precipitation": {"expected_mm": wf.get("total_rainfall_forecast_mm", 0) / 7, "probability_percent": 20},
                    "condition_summary": f"Humidity: {wf.get('humidity_trend', 'Unknown')}"
                },
                {
                    "day_label": "Tomorrow",
                    "date": "2026-04-26",
                    "temperature": {"max_celsius": wf.get("avg_temp_c", 30), "min_celsius": wf.get("avg_temp_c", 30) - 5},
                    "precipitation": {"expected_mm": wf.get("total_rainfall_forecast_mm", 0) / 7, "probability_percent": 20},
                    "condition_summary": "Normal"
                }
            ]
        }
        
        mp = data.get("Market_Data_2026", [])
        prices = []
        for m in mp:
            crop_name = "Paddy (IR-64)" if m.get("crop") == "Rice" else m.get("crop")
            prices.append({
                "crop": crop_name,
                "modal_price_rs_per_quintal": m.get("pune_mandi_price"),
                "trend": m.get("price_trend")
            })
        
        pol = data.get("Policy_Data_2026", {})
        kb["market_policy_data_2026"] = {
            "current_mandi_prices": {"prices": prices},
            "minimum_support_price_msp": {("paddy" if k == "Rice" else k.lower()): v for k, v in pol.get("msp_2026", {}).items()},
            "export_demand_signals": {("paddy" if k == "Rice" else k.lower()): v for k, v in pol.get("export_status", {}).items()},
            "government_schemes_2026": pol.get("subsidies", [])
        }
        
        data = {"knowledge_bundle": kb}

    print(f"[DataLoader] ✅ Loaded knowledge bundle from '{filepath.name}'")
    return data


# ──────────────────────────────────────────────
#  Section Extractors
# ──────────────────────────────────────────────

def get_bundle(data: dict) -> dict:
    """Return the inner knowledge_bundle dict."""
    return data["knowledge_bundle"]


def get_metadata(bundle: dict) -> dict:
    """Extract metadata (region, season, soil type, etc.)."""
    return bundle.get("metadata", {})


def get_ml_engine_data(bundle: dict) -> dict:
    """Extract the ML Engine crop-suitability section."""
    return bundle.get("ml_engine_crop_suitability", {})


def get_soil_health_card(bundle: dict) -> dict:
    """Extract the Soil Health Card section (macro + micro nutrients)."""
    return bundle.get("soil_health_card", {})


def get_weather_forecast(bundle: dict) -> dict:
    """Extract the 2-day weather forecast section."""
    return bundle.get("weather_forecast_2day", {})


def get_market_policy_data(bundle: dict) -> dict:
    """Extract the 2026 Market & Policy data section."""
    return bundle.get("market_policy_data_2026", {})


# ──────────────────────────────────────────────
#  Top-5 Crop Extraction
# ──────────────────────────────────────────────

def get_top5_crops(bundle: dict) -> list[dict]:
    """Return the list of Top 5 crops from the ML Engine data.

    Each crop dict contains:
        rank, crop_name, crop_code, suitability_score,
        predicted_yield_tons_per_ha, growing_period_days,
        water_requirement_mm, confidence_interval

    Returns:
        List of 5 crop dictionaries sorted by rank.
    """
    ml = get_ml_engine_data(bundle)
    crops = ml.get("top_5_crops", [])
    if not crops:
        raise ValueError("No 'top_5_crops' found in ML Engine data.")
    return sorted(crops, key=lambda c: c["rank"])


def get_top5_crop_names(bundle: dict) -> list[str]:
    """Convenience: return just the crop names in rank order."""
    return [crop["crop_name"] for crop in get_top5_crops(bundle)]


# ──────────────────────────────────────────────
#  Micronutrient Helpers
# ──────────────────────────────────────────────

def get_micronutrients(bundle: dict) -> dict:
    """Return the micronutrients sub-section from the Soil Health Card."""
    shc = get_soil_health_card(bundle)
    return shc.get("micronutrients", {})


def get_deficient_micronutrients(bundle: dict) -> list[dict]:
    """Return a list of micronutrients rated 'Deficient' or 'Low'.

    Each entry contains:
        name, value, rating, critical_limit, recommendation
    """
    micros = get_micronutrients(bundle)
    deficient = []
    for name, info in micros.items():
        if info.get("rating", "").lower() in ("deficient", "low"):
            deficient.append({
                "name": name.replace("_ppm", "").replace("_", " ").title(),
                "value_ppm": info["value"],
                "rating": info["rating"],
                "critical_limit": info.get("critical_limit"),
                "recommendation": info.get("recommendation", "N/A"),
            })
    return deficient


# ──────────────────────────────────────────────
#  Weather Helpers
# ──────────────────────────────────────────────

def get_daily_forecasts(bundle: dict) -> list[dict]:
    """Return the list of daily forecast dicts."""
    weather = get_weather_forecast(bundle)
    return weather.get("daily_forecast", [])


def get_agro_advisory(bundle: dict) -> dict:
    """Return the agro-meteorological advisory."""
    weather = get_weather_forecast(bundle)
    return weather.get("agro_advisory", {})


# ──────────────────────────────────────────────
#  Market / Policy Helpers
# ──────────────────────────────────────────────

def get_msp_data(bundle: dict) -> dict:
    """Return Minimum Support Price data for all crops."""
    mp = get_market_policy_data(bundle)
    return mp.get("minimum_support_price_msp", {})


def get_mandi_prices(bundle: dict) -> list[dict]:
    """Return current mandi (market) prices snapshot."""
    mp = get_market_policy_data(bundle)
    return mp.get("current_mandi_prices", {}).get("prices", [])


def get_government_schemes(bundle: dict) -> list[dict]:
    """Return list of active government schemes."""
    mp = get_market_policy_data(bundle)
    return mp.get("government_schemes_2026", [])


def get_export_signals(bundle: dict) -> dict:
    """Return export demand signals per crop."""
    mp = get_market_policy_data(bundle)
    return mp.get("export_demand_signals", {})


# ──────────────────────────────────────────────
#  Aggregated Context Builder (for Agents)
# ──────────────────────────────────────────────

def build_crop_context(bundle: dict, crop: dict) -> dict:
    """Build a unified context dict for a single crop.

    Merges ML suitability, soil deficiencies, weather outlook,
    MSP / mandi price, and policy info into a single dict
    that an agent can consume.
    """
    crop_name = crop["crop_name"]
    crop_code = crop["crop_code"]

    # Soil deficiencies
    deficiencies = get_deficient_micronutrients(bundle)

    # Weather
    forecasts = get_daily_forecasts(bundle)
    advisory = get_agro_advisory(bundle)

    # Market
    msp = get_msp_data(bundle)
    mandi = get_mandi_prices(bundle)
    export = get_export_signals(bundle)

    # Find matching mandi price
    mandi_match = next(
        (p for p in mandi if crop_name.split(" (")[0].lower() in p["crop"].lower()),
        None,
    )

    # Find matching MSP
    msp_key = None
    for key in msp:
        if crop_code.split("_")[0].lower() in key.lower():
            msp_key = key
            break
    msp_info = msp.get(msp_key, {}) if msp_key else {}

    # Find export signal
    export_key = crop_name.split(" (")[0].lower().replace(" ", "_")
    export_signal = export.get(export_key, "No specific signal")

    return {
        "crop": crop,
        "soil_deficiencies": deficiencies,
        "weather_forecast": forecasts,
        "agro_advisory": advisory,
        "msp": msp_info,
        "mandi_price": mandi_match,
        "export_signal": export_signal,
        "government_schemes": get_government_schemes(bundle),
    }


# ──────────────────────────────────────────────
#  CLI Preview
# ──────────────────────────────────────────────

def print_summary(bundle: dict) -> None:
    """Print a human-readable summary of the loaded knowledge bundle."""
    meta = get_metadata(bundle)
    top5 = get_top5_crops(bundle)
    deficiencies = get_deficient_micronutrients(bundle)
    forecasts = get_daily_forecasts(bundle)
    mandi = get_mandi_prices(bundle)

    print("\n" + "═" * 60)
    print("  📦  KNOWLEDGE BUNDLE SUMMARY")
    print("═" * 60)

    # Metadata
    print(f"\n📍 Region     : {meta.get('region', 'N/A')}")
    print(f"📍 District   : {meta.get('district', 'N/A')}")
    print(f"🌾 Season     : {meta.get('season', 'N/A')}")
    print(f"🪨 Soil Type  : {meta.get('soil_type', 'N/A')}")
    print(f"💧 Irrigation : {meta.get('irrigation_source', 'N/A')}")

    # Top 5 Crops
    print(f"\n{'─' * 60}")
    print("  🏆  TOP 5 CROPS (ML Engine Suitability)")
    print(f"{'─' * 60}")
    print(f"  {'Rank':<6}{'Crop':<28}{'Score':<8}{'Yield (t/ha)'}")
    print(f"  {'─'*5:<6}{'─'*26:<28}{'─'*5:<8}{'─'*10}")
    for crop in top5:
        print(
            f"  {crop['rank']:<6}"
            f"{crop['crop_name']:<28}"
            f"{crop['suitability_score']:<8.2f}"
            f"{crop['predicted_yield_tons_per_ha']}"
        )

    # Soil Deficiencies
    print(f"\n{'─' * 60}")
    print("  ⚠️  SOIL MICRONUTRIENT DEFICIENCIES")
    print(f"{'─' * 60}")
    if deficiencies:
        for d in deficiencies:
            print(f"  • {d['name']}: {d['value_ppm']} ppm ({d['rating']}) → {d['recommendation']}")
    else:
        print("  ✅ No deficiencies detected.")

    # Weather
    print(f"\n{'─' * 60}")
    print("  🌤️  2-DAY WEATHER FORECAST")
    print(f"{'─' * 60}")
    for day in forecasts:
        print(
            f"  📅 {day['day_label']} ({day['date']}): "
            f"{day['temperature']['max_celsius']}°C / {day['temperature']['min_celsius']}°C  "
            f"Rain: {day['precipitation']['probability_percent']}% ({day['precipitation']['expected_mm']}mm)"
        )
        print(f"     {day['condition_summary']}")

    # Mandi Prices
    print(f"\n{'─' * 60}")
    print("  💰  MANDI PRICES (Thanjavur APMC)")
    print(f"{'─' * 60}")
    print(f"  {'Crop':<22}{'Price (₹/qtl)':<16}{'Trend'}")
    print(f"  {'─'*20:<22}{'─'*13:<16}{'─'*8}")
    for p in mandi:
        print(f"  {p['crop']:<22}{p['modal_price_rs_per_quintal']:<16}{p['trend']}")

    print(f"\n{'═' * 60}\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    try:
        raw_data = load_json()
        bundle = get_bundle(raw_data)
        print_summary(bundle)

        # Demonstrate top-5 extraction
        top5 = get_top5_crops(bundle)
        print("🌱 Top 5 crop names for agent analysis:")
        for name in get_top5_crop_names(bundle):
            print(f"   → {name}")
        print()

        # Demonstrate per-crop context building
        print("📋 Building unified context for each crop…")
        for crop in top5:
            ctx = build_crop_context(bundle, crop)
            mandi_str = (
                f"₹{ctx['mandi_price']['modal_price_rs_per_quintal']}/qtl ({ctx['mandi_price']['trend']})"
                if ctx["mandi_price"]
                else "N/A"
            )
            print(f"   [{crop['rank']}] {crop['crop_name']}: Mandi={mandi_str}, Export={ctx['export_signal'][:50]}…")
        print("\n✅ Data loader ready. All sections parsed successfully.\n")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
