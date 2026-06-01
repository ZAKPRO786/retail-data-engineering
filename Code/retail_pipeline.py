import pandas as pd
import numpy as np
import logging
from pathlib import Path

# =====================================
# Setup
# =====================================

Path("logs").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Pipeline Started")

# =====================================
# Load Data
# =====================================

retail1 = pd.read_csv("data/raw/retail_data1.csv")
retail2 = pd.read_csv("data/raw/retail_data2.csv")
product = pd.read_csv("data/reference/product_details.csv")

print(f"Retail1 Records : {len(retail1)}")
print(f"Retail2 Records : {len(retail2)}")

# =====================================
# Merge Data
# =====================================

df = pd.concat(
    [retail1, retail2],
    ignore_index=True
)

print(f"Total Records : {len(df)}")

# =====================================
# Remove Duplicates
# =====================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

duplicates_removed = before - after

print(f"Duplicates Removed : {duplicates_removed}")

logging.info(f"Duplicates Removed : {duplicates_removed}")

# =====================================
# Standardize Product Names
# =====================================

df["product_name"] = (
    df["product_name"]
    .astype(str)
    .str.strip()
    .str.title()
)

# =====================================
# Standardize Categories
# =====================================

df["category"] = (
    df["category"]
    .astype(str)
    .str.strip()
)

category_map = {
    "ELEC": "Electronics",
    "electronics": "Electronics",

    "FURN": "Furniture",
    "furniture": "Furniture",

    "HOME": "Home Appliances",
    "home appliances": "Home Appliances",
    "HOME APPLIANCES": "Home Appliances",

    "CLOTH": "Clothing",
    "clothing": "Clothing",
    "CLOTHING": "Clothing"
}
df["category"] = df["category"].replace(category_map)

# =====================================
# Product Reference Maps
# =====================================

product_name_map = dict(
    zip(product["product_id"],
        product["product_name"])
)

product_category_map = dict(
    zip(product["product_id"],
        product["category"])
)

product_price_map = dict(
    zip(product["product_id"],
        product["price"])
)

# =====================================
# Missing Value Handling
# =====================================

df["product_name"] = (
    df["product_name"]
    .replace("Nan", np.nan)
    .fillna(
        df["product_id"]
        .map(product_name_map)
    )
)

df["category"] = (
    df["category"]
    .replace("Nan", np.nan)
    .fillna(
        df["product_id"]
        .map(product_category_map)
    )
)

df["price"] = (
    df["price"]
    .fillna(
        df["product_id"]
        .map(product_price_map)
    )
)

# =====================================
# Date Parsing
# =====================================

def parse_date(x):

    if pd.isna(x):
        return pd.NaT

    x = str(x).strip()

    try:
        return pd.to_datetime(
            x,
            format="%d/%m/%Y"
        )
    except:
        pass

    try:
        return pd.to_datetime(
            x,
            format="%m-%d-%Y"
        )
    except:
        pass

    return pd.NaT


df["transaction_date"] = (
    df["transaction_date"]
    .apply(parse_date)
)

# =====================================
# Invalid Quantity Handling
# =====================================

df = df[df["quantity"] > 0]

# =====================================
# PII Masking
# =====================================

def mask_email(email):

    try:
        email = str(email)

        if "@" not in email:
            return email

        name, domain = email.split("@")

        return f"{name[0]}***@{domain}"

    except:
        return np.nan


def mask_phone(phone):

    try:
        phone = str(phone)

        return "*" * (len(phone) - 4) + phone[-4:]

    except:
        return np.nan


df["masked_email"] = (
    df["email"]
    .apply(mask_email)
)

df["masked_phone"] = (
    df["phone"]
    .apply(mask_phone)
)

# =====================================
# Revenue Calculation
# =====================================

df["revenue"] = (
    df["price"]
    * df["quantity"]
    * (1 - df["discount"])
)

# =====================================
# Date Features
# =====================================

df["year"] = (
    df["transaction_date"]
    .dt.year
)

df["month"] = (
    df["transaction_date"]
    .dt.month_name()
)

# =====================================
# Join Product Master
# =====================================

df = df.merge(
    product,
    on="product_id",
    how="left",
    suffixes=("", "_master")
)

# =====================================
# Data Quality Report
# =====================================

quality_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage":
        round(
            df.isnull().mean() * 100,
            2
        ).values
})

quality_report.to_csv(
    "data/processed/data_quality_report.csv",
    index=False
)

# =====================================
# KPI Tables
# =====================================

revenue_by_category = (
    df.groupby("category")["revenue"]
    .sum()
    .reset_index()
)

revenue_by_city = (
    df.groupby("city")["revenue"]
    .sum()
    .reset_index()
)

top_products = (
    df.groupby("product_name")["revenue"]
    .sum()
    .reset_index()
    .sort_values(
        by="revenue",
        ascending=False
    )
)

revenue_by_category.to_csv(
    "data/processed/revenue_by_category.csv",
    index=False
)

revenue_by_city.to_csv(
    "data/processed/revenue_by_city.csv",
    index=False
)

top_products.to_csv(
    "data/processed/top_products.csv",
    index=False
)

# =====================================
# Save Final Outputs
# =====================================

df.to_csv(
    "data/processed/cleaned_retail_data.csv",
    index=False
)

df.to_parquet(
    "data/processed/cleaned_retail_data.parquet",
    index=False
)

# =====================================
# KPI Summary
# =====================================

print("\nKPI SUMMARY")
print("-" * 40)

print(
    f"Total Revenue: {round(df['revenue'].sum(),2)}"
)

print(
    f"Total Orders: {df['transaction_id'].nunique()}"
)

print(
    f"Total Customers: {df['customer_id'].nunique()}"
)

print(
    f"Average Order Value: {round(df['revenue'].mean(),2)}"
)

print(
    f"Missing Dates: {df['transaction_date'].isnull().sum()}"
)

logging.info("Pipeline Completed Successfully")

print("\nPipeline Completed Successfully")