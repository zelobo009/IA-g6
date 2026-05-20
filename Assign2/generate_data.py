"""
generate_data.py — Synthetic Airbnb data for Portugal (all 18 districts)

"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ─── Districts ────────────────────────────────────────────────────────────────
# base        : base nightly price (€)
# popularity  : demand multiplier
# type        : coastal / urban / historic / rural
# competition : fixed district-level listing density index (0–1)
LOCATIONS = {
    # Coastal
    "Aveiro":           {"base": 75,  "popularity": 1.15, "type": "coastal",  "competition": 0.48},
    "Faro":             {"base": 125, "popularity": 1.50, "type": "coastal",  "competition": 0.78},
    "Leiria":           {"base": 70,  "popularity": 1.10, "type": "coastal",  "competition": 0.40},
    "Setúbal":          {"base": 95,  "popularity": 1.25, "type": "coastal",  "competition": 0.60},
    "Viana do Castelo": {"base": 68,  "popularity": 1.08, "type": "coastal",  "competition": 0.38},
    # Urban / Metro
    "Braga":            {"base": 72,  "popularity": 1.15, "type": "urban",    "competition": 0.58},
    "Coimbra":          {"base": 74,  "popularity": 1.12, "type": "urban",    "competition": 0.55},
    "Lisboa":           {"base": 118, "popularity": 1.55, "type": "urban",    "competition": 0.92},
    "Porto":            {"base": 102, "popularity": 1.45, "type": "urban",    "competition": 0.85},
    "Santarém":         {"base": 60,  "popularity": 1.00, "type": "urban",    "competition": 0.35},
    # Historic / Cultural
    "Castelo Branco":   {"base": 55,  "popularity": 0.95, "type": "historic", "competition": 0.28},
    "Évora":            {"base": 82,  "popularity": 1.22, "type": "historic", "competition": 0.52},
    "Viseu":            {"base": 60,  "popularity": 1.02, "type": "historic", "competition": 0.32},
    # Rural / Interior
    "Beja":             {"base": 55,  "popularity": 0.95, "type": "rural",    "competition": 0.20},
    "Bragança":         {"base": 52,  "popularity": 0.90, "type": "rural",    "competition": 0.15},
    "Guarda":           {"base": 50,  "popularity": 0.88, "type": "rural",    "competition": 0.14},
    "Portalegre":       {"base": 53,  "popularity": 0.92, "type": "rural",    "competition": 0.18},
    "Vila Real":        {"base": 55,  "popularity": 0.95, "type": "rural",    "competition": 0.22},
    # Autonomous Regions
    "Madeira":          {"base": 105, "popularity": 1.35, "type": "urban",    "competition": 0.65},
    "Açores":           {"base": 88,  "popularity": 1.20, "type": "rural",    "competition": 0.30},
}

DISTRICT_TYPE_BOOST = {"coastal": 12, "urban": 8, "historic": 10, "rural": -8}

# ─── Events ───────────────────────────────────────────────────────────────────
EVENTS = ["none", "festival", "sports_event", "concert", "fair", "national_holiday"]

EVENT_OCC_BOOST = {
    "none":             0,
    "festival":        30,
    "sports_event":    25,
    "concert":         18,
    "fair":            12,
    "national_holiday":28,
}

EVENT_PRICE_BOOST = {   # additive €
    "none":              0,
    "festival":         38,
    "sports_event":     32,
    "concert":          22,
    "fair":             15,
    "national_holiday": 30,
}

EVENT_PROBS = [0.52, 0.12, 0.10, 0.10, 0.08, 0.08]

# ─── Portuguese National Holidays (month, day) ────────────────────────────────
NATIONAL_HOLIDAYS = {
    (1, 1),   # Ano Novo
    (4, 25),  # Dia da Liberdade
    (5, 1),   # Dia do Trabalhador
    (6, 10),  # Dia de Portugal
    (8, 15),  # Assunção de Nossa Senhora
    (10, 5),  # Implantação da República
    (11, 1),  # Todos os Santos
    (12, 1),  # Restauração da Independência
    (12, 8),  # Imaculada Conceição
    (12, 25), # Natal
    # Moveable — fixed approximations for synthetic data
    (2, 20),  # Carnaval
    (3, 29),  # Sexta-feira Santa
    (3, 31),  # Páscoa
    (6, 19),  # Corpo de Deus
}

# ─── Season & calendar boosts ────────────────────────────────────────────────
SEASON_BOOST = {
    1: 5,  2: 6,  3: 10, 4: 15,
    5: 20, 6: 26, 7: 32, 8: 30,
    9: 22, 10: 14, 11: 7, 12: 16,
}

DOW_BOOST = {0: -10, 1: -13, 2: -10, 3: -7, 4: 8, 5: 24, 6: 20}

# ─── Property types ───────────────────────────────────────────────────────────
PROPERTY_TYPES = ["Apartment", "House", "Room", "Shared_House"]
PROPERTY_TYPE_PROBS = [0.45, 0.25, 0.20, 0.10]

PROPERTY_BASE_MULT = {
    "Apartment":   1.00,
    "House":       1.55,
    "Room":        0.48,
    "Shared_House":0.38,
}

# Probability that a property type has parking / pool
PARKING_PROB = {"Apartment": 0.35, "House": 0.70, "Room": 0.15, "Shared_House": 0.25}
POOL_PROB    = {"Apartment": 0.10, "House": 0.45, "Room": 0.02, "Shared_House": 0.08}


# ─── Generator ────────────────────────────────────────────────────────────────

def generate(n: int = 6000, extra_rows=None) -> pd.DataFrame:
    """
    Generate n synthetic rows of Airbnb-style listing data for Portugal.

    Parameters
    ----------
    n          : number of synthetic rows to generate
    extra_rows : list of dicts with real observations to append

    Returns
    -------
    pd.DataFrame with columns:
        month, day_of_week, is_weekend, is_holiday, event, lead_days,
        location, district_type, competition,
        rooms, property_type, has_parking, has_pool,
        review_score,
        occupancy, price_eur
    """
    rng = np.random.default_rng(42)

    # ── Temporal ──────────────────────────────────────────────────────────────
    months     = rng.integers(1, 13, n)
    days       = rng.integers(1, 29, n)   # max 28 — always valid
    dow        = rng.integers(0, 7, n)
    is_weekend = (dow >= 5).astype(int)

    # ── Events + holiday flag ─────────────────────────────────────────────────
    events = rng.choice(EVENTS, n, p=EVENT_PROBS)
    is_holiday = np.array(
        [(int(m), int(d)) in NATIONAL_HOLIDAYS for m, d in zip(months, days)],
        dtype=int,
    )
    for i in range(n):
        if is_holiday[i] and events[i] == "none":
            events[i] = "national_holiday"

    # ── Location & district characteristics ───────────────────────────────────
    loc_keys    = list(LOCATIONS.keys())
    locs        = rng.choice(loc_keys, n)
    dist_types  = np.array([LOCATIONS[l]["type"]        for l in locs])
    competition = np.array([LOCATIONS[l]["competition"] for l in locs])

    # ── Listing attributes ────────────────────────────────────────────────────
    rooms      = rng.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.28, 0.30, 0.18, 0.12])
    prop_types = rng.choice(PROPERTY_TYPES, n, p=PROPERTY_TYPE_PROBS)
    lead       = rng.integers(1, 91, n)

    # Review score (1–5)
    review_score = np.clip(rng.normal(4.2, 0.5, n), 1.0, 5.0).round(2)

    # Parking & pool — probabilities vary by property type
    has_parking = np.array(
        [int(rng.random() < PARKING_PROB[p]) for p in prop_types]
    )
    has_pool = np.array(
        [int(rng.random() < POOL_PROB[p]) for p in prop_types]
    )

    # ── Occupancy model ───────────────────────────────────────────────────────
    season_b      = np.array([SEASON_BOOST[int(m)] for m in months])
    dow_b         = np.array([DOW_BOOST[int(d)]    for d in dow])
    event_occ_b   = np.array([EVENT_OCC_BOOST[e]   for e in events])
    dtype_b       = np.array([DISTRICT_TYPE_BOOST[t] for t in dist_types])
    lead_b        = -0.28 * lead + np.where(lead <= 3, 10, 0)

    review_b      = (review_score - 3.5) * 8
    parking_occ_b = has_parking * 3
    pool_occ_b    = has_pool    * 5
    competition_b = -competition * 12   # higher district competition → lower your share
    weekend_b     = is_weekend  * 4
    holiday_b     = is_holiday  * 10

    occupancy = (
        50
        + season_b + dow_b + event_occ_b + dtype_b
        + lead_b
        + review_b
        + parking_occ_b + pool_occ_b
        + competition_b
        + weekend_b + holiday_b
        + rng.normal(0, 6, n)
    ).clip(5, 99)

    booking_probability = occupancy / 100

    occupied = np.array([
        int(rng.random() < p)
        for p in booking_probability
    ])

    # ── Price model ───────────────────────────────────────────────────────────
    loc_base  = np.array([LOCATIONS[l]["base"]       for l in locs])
    loc_pop   = np.array([LOCATIONS[l]["popularity"] for l in locs])
    event_p   = np.array([EVENT_PRICE_BOOST[e]       for e in events])
    prop_mult = np.array([PROPERTY_BASE_MULT[p]      for p in prop_types])

    occ_mult        = 0.60 + (occupancy / 100) * 1.05
    review_price_b  = (review_score - 3.5) * 12
    parking_price_b = has_parking * 6
    pool_price_b    = has_pool    * 18
    room_b          = (rooms - 1) * 14

    price = (
        loc_base * occ_mult * loc_pop * prop_mult
        + event_p
        + review_price_b
        + parking_price_b + pool_price_b
        + room_b
        + rng.normal(0, 8, n)
    ).clip(15, 2500).round(0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "month":         months.astype(int),
        "day_of_week":   dow.astype(int),
        "is_weekend":    is_weekend,
        "is_holiday":    is_holiday,
        "event":         events,
        "lead_days":     lead.astype(int),
        "location":      locs,
        "district_type": dist_types,
        "competition":   competition,
        "rooms":         rooms.astype(int),
        "property_type": prop_types,
        "has_parking":   has_parking,
        "has_pool":      has_pool,
        "review_score":  review_score,
        "occupancy":     occupied,
        "price_eur":     price,
    })

    if extra_rows is not None and len(extra_rows) > 0:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


    return df


# ─── Exports consumed by app.py ───────────────────────────────────────────────

if __name__ == "__main__":
    df = generate(6000)
    df.to_csv("airbnb_data.csv", index=False)
    print(f"Saved {len(df)} rows to airbnb_data.csv")

__all__ = ["generate", "LOCATIONS", "EVENTS", "NATIONAL_HOLIDAYS", "PROPERTY_TYPES"]