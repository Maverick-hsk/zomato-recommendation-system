"""Generate a sample Zomato-style dataset for local development."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

CITIES = [
    ("New Delhi", ["Connaught Place", "Hauz Khas", "Karol Bagh", "Saket", "Dwarka"]),
    ("Bangalore", ["Indiranagar", "Koramangala", "BTM", "Whitefield", "MG Road"]),
    ("Mumbai", ["Bandra", "Andheri", "Colaba", "Powai", "Juhu"]),
    ("Hyderabad", ["Banjara Hills", "Hitech City", "Gachibowli", "Jubilee Hills"]),
    ("Pune", ["Koregaon Park", "Hinjewadi", "Camp", "Kothrud"]),
]

CUISINE_POOL = [
    "North Indian", "Chinese", "Italian", "South Indian", "Fast Food",
    "Mughlai", "Continental", "Thai", "Mexican", "Bakery", "Cafe",
    "Desserts", "Seafood", "Biryani", "Pizza", "BBQ", "Japanese",
]

RESTAURANT_PREFIXES = [
    "Spice", "Royal", "Urban", "Golden", "The", "Cafe", "Bistro",
    "Kitchen", "House of", "Delhi", "Mumbai", "Bangalore",
]

RESTAURANT_SUFFIXES = [
    "Garden", "Palace", "Express", "Corner", "Hub", "Kitchen",
    "Grill", "Lounge", "Diner", "Bites", "Treats", "Flavors",
]

RATING_TEXT = {
    (4.5, 5.1): "Excellent",
    (4.0, 4.5): "Very Good",
    (3.5, 4.0): "Good",
    (3.0, 3.5): "Average",
    (0.0, 3.0): "Poor",
}


def _rating_label(rating: float) -> str:
    for (low, high), label in RATING_TEXT.items():
        if low <= rating < high:
            return label
    return "Not rated"


def generate_sample_dataset(n_restaurants: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic restaurant records matching Zomato Kaggle schema."""
    random.seed(seed)
    rows = []

    for i in range(1, n_restaurants + 1):
        city, localities = random.choice(CITIES)
        locality = random.choice(localities)
        n_cuisines = random.randint(1, 3)
        cuisines = ", ".join(random.sample(CUISINE_POOL, n_cuisines))
        price_range = random.choices([1, 2, 3, 4], weights=[15, 45, 30, 10])[0]
        rating = round(random.uniform(2.5, 4.8), 1)
        votes = random.randint(10, 5000)

        name = f"{random.choice(RESTAURANT_PREFIXES)} {random.choice(RESTAURANT_SUFFIXES)} {i}"

        rows.append(
            {
                "Restaurant ID": i,
                "Restaurant Name": name,
                "Country Code": 1,
                "City": city,
                "Address": f"{locality}, {city}",
                "Locality": locality,
                "Locality Verbose": f"{locality}, {city}, India",
                "Longitude": round(random.uniform(72.8, 77.5), 4),
                "Latitude": round(random.uniform(12.9, 28.7), 4),
                "Cuisines": cuisines,
                "Average Cost for two": price_range * 400 + random.randint(0, 300),
                "Currency": "INR",
                "Has Table booking": random.choice(["Yes", "No"]),
                "Has Online delivery": random.choice(["Yes", "No"]),
                "Is delivering now": random.choice(["Yes", "No"]),
                "Switch to order menu": random.choice(["Yes", "No"]),
                "Price range": price_range,
                "Aggregate rating": rating,
                "Rating color": "Green" if rating >= 3.5 else "Yellow",
                "Rating text": _rating_label(rating),
                "Votes": votes,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    output = Path(__file__).resolve().parent.parent / "data" / "sample_zomato.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset()
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} restaurants -> {output}")


if __name__ == "__main__":
    main()
