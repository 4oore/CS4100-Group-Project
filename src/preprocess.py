from pathlib import Path
import pandas as pd
import argparse


_SRC_DIR = Path(__file__).parent
_DATA_DIR = _SRC_DIR.parent / "data"

FINAL_COLS = [
    "food_id",
    "name",
    "category",
    "source",
    "basis_g",
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Simulated Annealing Meal Planner")

    # Data
    parser.add_argument(
        "--usda_path",
        type=str,
        default=str(_DATA_DIR / "FoodData_Central_foundation_food_csv_2025-12-18"),
        help="path to USDA data folder",
    )
    parser.add_argument(
        "--off_path",
        type=str,
        default=str(_DATA_DIR / "en.openfoodfacts.org.products.csv"),
        help="path to OFF data TSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_DATA_DIR / "processed"),
        help="dir path to processed foods CSV",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=10000,
        help="max total number of foods to process",
    )

    parser.add_argument(
        "--max_per_category",
        type=int,
        default=1000,
        help="max number of foods to process per category",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()


def ingest_OOF(off_path):
    OFF_COLS = [
        "code",
        "product_name",
        "categories_en",
        "energy_100g",
        "fat_100g",
        "carbohydrates_100g",
        "proteins_100g",
        "sugars_100g",
        "fiber_100g",
        "sodium_100g",
        "salt_100g",
    ]

    def col(name):
        return pd.to_numeric(df[name], errors="coerce")

    chunks = []
    for chunk in pd.read_csv(
        off_path,
        sep="\t",
        usecols=lambda c: c in set(OFF_COLS),
        chunksize=50000,
        low_memory=False,
    ):
        df = chunk.rename(
            columns={
                "code": "food_id",
                "product_name": "name",
                "categories_en": "category",
            }
        )
        df["source"] = "off"
        df["basis_g"] = 100.0

        df["energy_kcal"] = col("energy_100g") / 4.184
        df["protein_g"] = col("proteins_100g")
        df["fat_g"] = col("fat_100g")
        df["carbs_g"] = col("carbohydrates_100g")
        df["sugar_g"] = col("sugars_100g")
        df["fiber_g"] = col("fiber_100g")
        df["sodium_mg"] = (
            col("sodium_100g").mul(1000).fillna(col("salt_100g") / 2.5 * 1000)
        )

        # Drop rows with no name or macros
        df = df[df["name"].notna() & (df["name"].str.strip() != "")]
        df = df.dropna(subset=["energy_kcal", "protein_g", "fat_g", "carbs_g"])

        chunks.append(df[FINAL_COLS])

    off = pd.concat(chunks, ignore_index=True)
    return off


def ingest_USDA(usda_path):
    # Load food list
    food = pd.read_csv(
        usda_path / "food.csv",
        usecols=["fdc_id", "data_type", "description", "food_category_id"],
        dtype=str,
    )

    # Load categories
    cats = pd.read_csv(
        usda_path / "food_category.csv", usecols=["id", "description"]
    ).rename(columns={"id": "food_category_id", "description": "category"})
    cats["food_category_id"] = cats["food_category_id"].astype(str)

    # Load & filter nutrients
    NUTRIENT_SPECS = [
        ("Energy", "KCAL", "energy_kcal"),
        ("Protein", "G", "protein_g"),
        ("Total lipid (fat)", "G", "fat_g"),
        ("Carbohydrate, by difference", "G", "carbs_g"),
        ("Fiber, total dietary", "G", "fiber_g"),
        ("Sugars, Total", "G", "sugar_g"),
        ("Sodium, Na", "MG", "sodium_mg"),
    ]

    specs_df = pd.DataFrame(NUTRIENT_SPECS, columns=["name", "unit_name", "key"])
    nutrients = pd.read_csv(
        usda_path / "nutrient.csv", usecols=["id", "name", "unit_name"]
    ).merge(specs_df, on=["name", "unit_name"])

    # Load food-nutrient amounts and pivot to wide
    fn = (
        pd.read_csv(
            usda_path / "food_nutrient.csv",
            usecols=["fdc_id", "nutrient_id", "amount"],
            dtype={"fdc_id": str, "amount": "float64"},
        )
        .merge(nutrients[["id", "key"]], left_on="nutrient_id", right_on="id")
        .groupby(["fdc_id", "key"])["amount"]
        .mean()
        .unstack("key")
        .reset_index()
    )

    # Assemble
    usda = (
        food.merge(fn, on="fdc_id", how="left")
        .merge(cats, on="food_category_id", how="left")
        .rename(columns={"fdc_id": "food_id", "description": "name"})
    )
    usda["source"] = "usda"
    usda["basis_g"] = 100.0

    return usda[FINAL_COLS]


def clean(foods):
    # Drop rows any nutritional value doesn't exist
    foods = foods.dropna(
        subset=[
            "energy_kcal",
            "protein_g",
            "fat_g",
            "carbs_g",
            "fiber_g",
            "sugar_g",
            "sodium_mg",
        ]
    )

    # Drop rows where name doesn't exist
    foods = foods[foods["name"].notna() & (foods["name"].astype(str).str.strip() != "")]

    # Drop dublicates
    foods = foods.drop_duplicates(subset=["source", "food_id"]).reset_index(drop=True)

    return foods


if __name__ == "__main__":
    args = _parse_args()

    # Make sure out dir exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    off_df = ingest_OOF(args.off_path)
    usda_df = ingest_USDA(Path(args.usda_path))

    print("USDA data shape:", usda_df.shape)
    print("OFF data shape:", off_df.shape)

    # Combine both dataframes
    combined_df = pd.concat([usda_df, off_df], ignore_index=True)

    # Shuffle the combined df
    combined_df = combined_df.sample(frac=1, random_state=args.seed).reset_index(
        drop=True
    )

    print("Combined data shape before cleaning:", combined_df.shape)

    # Clean df
    cleaned_df = clean(combined_df)

    print("Combined data shape after cleaning:", cleaned_df.shape)

    # Limit total and per category
    limited_df = (
        cleaned_df.groupby("category")
        .head(args.max_per_category)
        .reset_index(drop=True)
        .head(args.max_total)
        .reset_index(drop=True)
    )

    print("Final data shape after limiting:", limited_df.shape)

    # Save processed data
    out_path = out_dir / "foods.csv"
    limited_df.to_csv(out_path, index=False)
