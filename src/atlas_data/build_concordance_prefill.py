from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


###
# CONFIG
###

DEFAULT_CONCORDANCE_PATH = Path("data/atlas/concordance/hs92_to_eora26_manual.csv")
DEFAULT_OUTPUT_PATH = Path("data/atlas/concordance/hs92_to_eora26_prefilled.csv")
DEFAULT_REVIEW_PATH = Path("data/atlas/concordance/hs92_to_eora26_review.csv")


###
# EORA26 SECTOR LABELS
###

EORA_SECTORS = {
    "Agriculture",
    "Fishing",
    "Mining and Quarrying",
    "Food & Beverages",
    "Textiles and Wearing Apparel",
    "Wood and Paper",
    "Petroleum, Chemical and Non-Metallic Mineral Products",
    "Metal Products",
    "Electrical and Machinery",
    "Transport Equipment",
    "Other Manufacturing",
    "Recycling",
    "Electricity, Gas and Water",
    "Construction",
    "Maintenance and Repair",
    "Wholesale Trade",
    "Retail Trade",
    "Hotels and Restraurants",
    "Transport",
    "Post and Telecommunications",
    "Finacial Intermediation and Business Activities",
    "Public Administration",
    "Education, Health and Other Services",
    "Private Households",
    "Others",
    "Re-export & Re-import",
}


###
# HELPERS
###

def hs2(code: str) -> int | None:
    code = str(code)

    if not code[:2].isdigit():
        return None

    return int(code[:2])


def is_empty(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def validate_sector(sector: str) -> None:
    if sector not in EORA_SECTORS:
        raise ValueError(f"Unknown Eora sector label: {sector}")


###
# MAPPING RULES
###

def map_service_code(code: str) -> tuple[str, str]:
    code = str(code).strip()

    service_mapping = {
        "transport": "Transport",
        "ict": "Post and Telecommunications",
        "financial": "Finacial Intermediation and Business Activities",
        "travel": "Hotels and Restraurants",
        "unspecified": "Others",
        "XXXX": "Re-export & Re-import",
    }

    if code in service_mapping:
        sector = service_mapping[code]
        validate_sector(sector)
        return sector, "service_manual_rule"

    return "", ""


def map_hs4_code(code: str, name: str) -> tuple[str, str]:
    chapter = hs2(code)

    if chapter is None:
        return "", ""

    name_lower = str(name).lower()

    # 01-05: live animals and animal products
    if 1 <= chapter <= 3:
        return "Agriculture", "hs_chapter_rule"

    if chapter == 4:
        return "Food & Beverages", "hs_chapter_rule"

    if chapter == 5:
        return "Agriculture", "hs_chapter_rule"

    # 06-14: vegetable products
    if 6 <= chapter <= 14:
        return "Agriculture", "hs_chapter_rule"

    # 15-24: fats, prepared food, beverages, tobacco
    if 15 <= chapter <= 24:
        return "Food & Beverages", "hs_chapter_rule"

    # 25-27: minerals and fuels
    if 25 <= chapter <= 27:
        return "Mining and Quarrying", "hs_chapter_rule"

    # 28-40: chemicals, plastics, rubber
    if 28 <= chapter <= 40:
        return "Petroleum, Chemical and Non-Metallic Mineral Products", "hs_chapter_rule"

    # 41-43: hides, skins, leather
    if 41 <= chapter <= 43:
        return "Textiles and Wearing Apparel", "hs_chapter_rule"

    # 44-49: wood, paper, printed matter
    if 44 <= chapter <= 49:
        return "Wood and Paper", "hs_chapter_rule"

    # 50-63: textiles and apparel
    if 50 <= chapter <= 63:
        return "Textiles and Wearing Apparel", "hs_chapter_rule"

    # 64-67: footwear, headgear, feathers
    if 64 <= chapter <= 67:
        return "Textiles and Wearing Apparel", "hs_chapter_rule"

    # 68-71: stone, cement, glass, precious stones/metals
    if 68 <= chapter <= 71:
        return "Petroleum, Chemical and Non-Metallic Mineral Products", "hs_chapter_rule"

    # 72-83: base metals and articles of base metal
    if 72 <= chapter <= 83:
        return "Metal Products", "hs_chapter_rule"

    # 84-85: machinery, electrical machinery
    if 84 <= chapter <= 85:
        return "Electrical and Machinery", "hs_chapter_rule"

    # 86-89: transport equipment
    if 86 <= chapter <= 89:
        return "Transport Equipment", "hs_chapter_rule"

    # 90-92: optical, medical, precision, musical instruments
    if 90 <= chapter <= 92:
        return "Electrical and Machinery", "hs_chapter_rule"

    # 93: arms and ammunition
    if chapter == 93:
        return "Other Manufacturing", "hs_chapter_rule"

    # 94-96: furniture, toys, miscellaneous manufacturing
    if 94 <= chapter <= 96:
        return "Other Manufacturing", "hs_chapter_rule"

    # 97: works of art, antiques
    if chapter == 97:
        return "Others", "hs_chapter_rule"

    # 99 and other residual product codes
    if chapter >= 98:
        return "Re-export & Re-import", "hs_chapter_rule"

    # Text fallback for fish-related products if needed
    if "fish" in name_lower or "crustacean" in name_lower or "mollusc" in name_lower:
        return "Fishing", "name_rule"

    return "", ""


def infer_mapping(row: pd.Series) -> tuple[str, str]:
    code = str(row["code"])
    name = str(row["nameEn"])

    if bool(row.get("is_service_like", False)):
        return map_service_code(code)

    if bool(row.get("is_hs4_code", False)):
        return map_hs4_code(code, name)

    return "", ""


###
# PREFILL
###

def prefill_concordance(
    concordance_path: Path,
    output_path: Path,
    review_path: Path,
    force_existing: bool = False,
) -> pd.DataFrame:
    if not concordance_path.exists():
        raise FileNotFoundError(f"Concordance skeleton not found: {concordance_path}")

    df = pd.read_csv(concordance_path, dtype=str).fillna("")

    bool_cols = ["is_hs4_code", "is_service_like", "greenProduct", "naturalResource"]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: str(x).strip().lower() in ["true", "1", "yes"]
        )

    required_cols = [
        "productIdRaw",
        "code",
        "nameEn",
        "is_hs4_code",
        "is_service_like",
        "eora26_sector",
        "mapping_status",
        "mapping_method",
        "notes",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required concordance columns: {missing_cols}")

    df = df.copy()

    assigned = 0
    preserved = 0

    for idx, row in df.iterrows():
        existing_sector = row.get("eora26_sector", "")

        if not force_existing and not is_empty(existing_sector):
            preserved += 1
            continue

        sector, method = infer_mapping(row)

        if sector:
            df.at[idx, "eora26_sector"] = sector
            df.at[idx, "mapping_status"] = "prefilled"
            df.at[idx, "mapping_method"] = method
            assigned += 1
        else:
            df.at[idx, "mapping_status"] = "needs_review"
            df.at[idx, "mapping_method"] = ""

    review = df[df["mapping_status"].isin(["needs_review", "unmapped"])].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    logging.info("Rows: %s", len(df))
    logging.info("Assigned by prefill: %s", assigned)
    logging.info("Preserved existing mappings: %s", preserved)
    logging.info("Needs review: %s", len(review))
    logging.info("Saved prefilled concordance -> %s", output_path)
    logging.info("Saved review file -> %s", review_path)

    return df


###
# CLI
###

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concordance-path",
        default=str(DEFAULT_CONCORDANCE_PATH),
    )

    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--review-path",
        default=str(DEFAULT_REVIEW_PATH),
    )

    parser.add_argument(
        "--force-existing",
        action="store_true",
        help="Overwrite existing eora26_sector assignments.",
    )

    return parser.parse_args()


###
# MAIN
###

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    prefill_concordance(
        concordance_path=Path(args.concordance_path),
        output_path=Path(args.output_path),
        review_path=Path(args.review_path),
        force_existing=args.force_existing,
    )


if __name__ == "__main__":
    main()