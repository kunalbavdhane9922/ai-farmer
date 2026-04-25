import os

def main():
    file_path = "agents.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Exact string replacements for agent names
    replacements = {
        "Agronomist": "Farmer",
        "Market Strategist": "Trader",
        "Trend Forecaster": "Analyst",
        "Mkt Strategist": "Trader",
        "Mkt Strat": "Trader",
        # Lowercase variables
        "run_agronomist": "run_farmer",
        "run_market_strategist": "run_trader",
        "run_trend_forecaster": "run_analyst",
        "agro_result": "farmer_result",
        "market_result": "trader_result",
        "trend_result": "analyst_result",
        "final_agro": "final_farmer",
        "final_market": "final_trader",
        "current_agro": "current_farmer",
        "current_market": "current_trader",
        "new_agro": "new_farmer",
        "new_market": "new_trader",
        "old_agro": "old_farmer",
        "old_market": "old_trader",
        "agro_map": "farmer_map",
        "mkt_map": "trader_map",
        "trend_map": "analyst_map",
        "s_agronomy": "s_farmer",
        "s_market": "s_trader",
        "s_demand": "s_analyst",
        "agro_reason": "farmer_reason",
        "market_reason": "trader_reason",
        "trend_reason": "analyst_reason",
        "final_agronomist": "final_farmer",
        "final_market_strategist": "final_trader",
        "trend_forecaster_unchanged": "analyst_unchanged"
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully updated agents.py with new agent names.")

if __name__ == "__main__":
    main()
