"""
Restaurant Revenue & Demand Analytics
======================================
IBM SkillsBuild Final Project

A complete end-to-end analytics pipeline + Streamlit dashboard for
restaurant sales data.

Run with:
    streamlit run restaurant_demand_analytics.py

Or run the pipeline only (no UI):
    python restaurant_demand_analytics.py --pipeline-only
"""

import sys
import warnings
import os

warnings.filterwarnings("ignore")

# ── Core libraries ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Machine Learning ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH = "restaurant_sales_data.csv"   # ← change path here if needed

PALETTE = {
    "primary":   "#3b82d4",
    "secondary": "#7c5cd8",
    "success":   "#22c55e",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "muted":     "#6b7280",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the CSV and return a raw DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please set CSV_PATH to the correct location."
        )
    df = pd.read_csv(path)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def audit_data(df: pd.DataFrame) -> dict:
    """
    Return a dictionary summarising data quality metrics.
    Does NOT modify the DataFrame.
    """
    report = {}
    report["shape"]          = df.shape
    report["missing_counts"] = df.isnull().sum().to_dict()
    report["total_missing"]  = int(df.isnull().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Duplicate transaction keys (date + restaurant + item + meal)
    dup_keys = df.duplicated(
        subset=["date", "restaurant_id", "menu_item_name", "meal_type"],
        keep=False
    )
    report["duplicate_keys"] = int(dup_keys.sum())

    report["zero_quantity"]  = int((df["quantity_sold"].astype(float) == 0).sum())
    report["negative_qty"]   = int((df["quantity_sold"].astype(float) < 0).sum())

    report["unique_restaurants"] = int(df["restaurant_id"].nunique())
    report["unique_items"]       = int(df["menu_item_name"].nunique())
    report["unique_rest_types"]  = df["restaurant_type"].unique().tolist()
    report["unique_meal_types"]  = df["meal_type"].unique().tolist()
    report["unique_weather"]     = df["weather_condition"].unique().tolist()

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data cleaning steps and return a cleaned copy.
    Retain zero-quantity rows as instructed.
    """
    df = df.copy()

    # 3a. Parse date
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # 3b. Ensure boolean columns are proper Python bools.
    # pandas reads TRUE/FALSE CSV values directly as bool dtype, so .map() with
    # string keys produces all-NaN. We handle both cases (string and bool input).
    for col in ["has_promotion", "special_event"]:
        if df[col].dtype == object:
            # Stored as strings (e.g. 'TRUE' / 'FALSE')
            df[col] = df[col].str.upper().map({"TRUE": True, "FALSE": False}).astype(bool)
        else:
            # Already bool or numeric — just cast
            df[col] = df[col].astype(bool)

    # 3c. Cast numeric columns to float/int
    for col in ["typical_ingredient_cost", "observed_market_price", "actual_selling_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quantity_sold"]   = pd.to_numeric(df["quantity_sold"],   errors="coerce").fillna(0).astype(int)
    df["restaurant_id"]   = pd.to_numeric(df["restaurant_id"],   errors="coerce").astype(int)

    # 3d. Strip whitespace from text columns
    text_cols = ["restaurant_type", "menu_item_name", "meal_type", "weather_condition"]
    for col in text_cols:
        df[col] = df[col].str.strip()

    # 3e. Sort by date for time-series work
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

DAY_NAMES = {
    0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
    4: "Fri", 5: "Sat", 6: "Sun",
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analysis and modelling."""
    df = df.copy()

    # ── Revenue & margin ─────────────────────────────────────────────────────
    df["revenue"]                = df["actual_selling_price"] * df["quantity_sold"]
    df["estimated_cost"]         = df["typical_ingredient_cost"] * df["quantity_sold"]
    df["estimated_gross_margin"] = df["revenue"] - df["estimated_cost"]

    # Avoid division by zero when revenue = 0 (zero-qty rows)
    df["margin_percentage"] = np.where(
        df["revenue"] > 0,
        df["estimated_gross_margin"] / df["revenue"] * 100,
        np.nan
    )

    # ── Price features ────────────────────────────────────────────────────────
    df["price_vs_market"] = df["actual_selling_price"] - df["observed_market_price"]
    df["markup_ratio"]    = df["actual_selling_price"] / df["typical_ingredient_cost"]

    # ── Date features ─────────────────────────────────────────────────────────
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.month.map(MONTH_NAMES)
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["day_name"]    = df["date"].dt.dayofweek.map(DAY_NAMES)
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KPI CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_kpis(df: pd.DataFrame) -> dict:
    """Return a dictionary of top-level KPI values."""
    kpis = {}
    kpis["total_revenue"]          = df["revenue"].sum()
    kpis["total_quantity"]         = int(df["quantity_sold"].sum())
    kpis["total_gross_margin"]     = df["estimated_gross_margin"].sum()
    kpis["avg_selling_price"]      = df["actual_selling_price"].mean()
    kpis["avg_margin_pct"]         = df.loc[df["revenue"] > 0, "margin_percentage"].mean()
    kpis["n_restaurants"]          = int(df["restaurant_id"].nunique())
    kpis["n_menu_items"]           = int(df["menu_item_name"].nunique())
    kpis["n_transactions"]         = len(df)
    kpis["zero_qty_rows"]          = int((df["quantity_sold"] == 0).sum())
    kpis["promo_rows"]             = int(df["has_promotion"].sum())
    kpis["special_event_rows"]     = int(df["special_event"].sum())
    return kpis


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EDA HELPERS (pure computation, no Streamlit)
# ═══════════════════════════════════════════════════════════════════════════════

def eda_revenue_by_item(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("menu_item_name", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity_sold", "sum"),
            avg_price=("actual_selling_price", "mean"),
            avg_margin_pct=("margin_percentage", "mean"),
            transaction_count=("revenue", "count"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return g


def eda_revenue_by_type(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("restaurant_type", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity_sold", "sum"),
            avg_margin_pct=("margin_percentage", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return g


def eda_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["year", "month", "month_name"], as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity_sold", "sum"),
        )
        .sort_values(["year", "month"])
    )
    return g


def eda_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("date", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity_sold", "sum"),
        )
        .sort_values("date")
    )
    return g


def eda_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["day_of_week", "day_name"], as_index=False)
        .agg(
            avg_quantity=("quantity_sold", "mean"),
            total_revenue=("revenue", "sum"),
        )
        .sort_values("day_of_week")
    )
    return g


def eda_meal_type(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("meal_type", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            avg_quantity=("quantity_sold", "mean"),
            total_quantity=("quantity_sold", "sum"),
        )
    )
    return g


def eda_promo_effect(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("has_promotion", as_index=False)
        .agg(
            avg_quantity=("quantity_sold", "mean"),
            total_revenue=("revenue", "sum"),
            count=("quantity_sold", "count"),
        )
    )
    # Use apply() so the label map works regardless of the column dtype
    g["label"] = g["has_promotion"].apply(
        lambda v: "Promotion" if bool(v) else "No Promotion"
    )
    return g


def eda_event_effect(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("special_event", as_index=False)
        .agg(
            avg_quantity=("quantity_sold", "mean"),
            total_revenue=("revenue", "sum"),
            count=("quantity_sold", "count"),
        )
    )
    g["label"] = g["special_event"].apply(
        lambda v: "Special Event" if bool(v) else "Normal Day"
    )
    return g


def eda_weather_effect(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("weather_condition", as_index=False)
        .agg(
            avg_quantity=("quantity_sold", "mean"),
            total_revenue=("revenue", "sum"),
            count=("quantity_sold", "count"),
        )
    )
    return g


def eda_margin_by_item(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.loc[df["revenue"] > 0]
        .groupby("menu_item_name", as_index=False)
        .agg(
            avg_margin_pct=("margin_percentage", "mean"),
            total_margin=("estimated_gross_margin", "sum"),
        )
        .sort_values("avg_margin_pct", ascending=False)
    )
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION HELPERS  (return matplotlib Figure objects)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9fafb")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_revenue_trend(df: pd.DataFrame):
    monthly = eda_monthly_trend(df)
    fig, ax = _fig(10, 3.5)
    ax.plot(
        range(len(monthly)),
        monthly["total_revenue"],
        color=PALETTE["primary"], linewidth=2.5, marker="o", markersize=4,
    )
    ax.fill_between(
        range(len(monthly)),
        monthly["total_revenue"],
        alpha=0.15, color=PALETTE["primary"],
    )
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_name"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"RM {x:,.0f}"))
    ax.set_title("Monthly Revenue Trend", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (RM)")
    plt.tight_layout()
    return fig


def plot_demand_trend(df: pd.DataFrame):
    daily = eda_daily_trend(df)
    # Resample to weekly for readability
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = daily.set_index("date").resample("W")["total_quantity"].sum().reset_index()
    fig, ax = _fig(10, 3.5)
    ax.plot(
        weekly["date"], weekly["total_quantity"],
        color=PALETTE["secondary"], linewidth=2, marker=".", markersize=3,
    )
    ax.fill_between(weekly["date"], weekly["total_quantity"], alpha=0.12, color=PALETTE["secondary"])
    ax.set_title("Weekly Demand Trend (Total Units Sold)", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Units Sold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_top_items_revenue(df: pd.DataFrame, top_n: int = 10):
    item_df = eda_revenue_by_item(df).head(top_n)
    fig, ax = _fig(8, 4)
    bars = ax.barh(
        item_df["menu_item_name"][::-1],
        item_df["total_revenue"][::-1],
        color=PALETTE["primary"],
    )
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"RM {x/1e6:.1f}M" if x >= 1e6 else f"RM {x:,.0f}"))
    ax.set_title(f"Top {top_n} Menu Items by Revenue", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Total Revenue (RM)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                f"RM {w:,.0f}", va="center", fontsize=8, color=PALETTE["muted"])
    plt.tight_layout()
    return fig


def plot_day_of_week(df: pd.DataFrame):
    dow = eda_day_of_week(df)
    fig, ax = _fig(7, 3.5)
    colors = [PALETTE["warning"] if d >= 5 else PALETTE["primary"] for d in dow["day_of_week"]]
    ax.bar(dow["day_name"], dow["avg_quantity"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Average Demand by Day of Week", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Avg Units Sold")
    ax.set_xlabel("Day")
    plt.tight_layout()
    return fig


def plot_meal_type(df: pd.DataFrame):
    meal = eda_meal_type(df)
    fig, ax = _fig(6, 3.5)
    bars = ax.bar(meal["meal_type"], meal["avg_quantity"], color=PALETTE["secondary"], edgecolor="white")
    ax.set_title("Average Demand by Meal Type", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Avg Units Sold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_promo_effect(df: pd.DataFrame):
    promo = eda_promo_effect(df)
    fig, ax = _fig(5, 3.5)
    colors = [PALETTE["success"], PALETTE["muted"]]
    ax.bar(promo["label"], promo["avg_quantity"], color=colors, edgecolor="white")
    ax.set_title("Avg Demand: Promotion vs No Promotion", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Avg Units Sold")
    for i, row in promo.iterrows():
        ax.text(i, row["avg_quantity"] + 1, f"{row['avg_quantity']:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    return fig


def plot_event_effect(df: pd.DataFrame):
    ev = eda_event_effect(df)
    fig, ax = _fig(5, 3.5)
    colors = [PALETTE["warning"], PALETTE["muted"]]
    ax.bar(ev["label"], ev["avg_quantity"], color=colors, edgecolor="white")
    ax.set_title("Avg Demand: Special Event vs Normal Day", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Avg Units Sold")
    for i, row in ev.iterrows():
        ax.text(i, row["avg_quantity"] + 1, f"{row['avg_quantity']:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    return fig


def plot_weather_effect(df: pd.DataFrame):
    w = eda_weather_effect(df)
    fig, ax = _fig(6, 3.5)
    palette = [PALETTE["warning"], PALETTE["muted"], PALETTE["primary"]]
    ax.bar(w["weather_condition"], w["avg_quantity"], color=palette, edgecolor="white")
    ax.set_title("Avg Demand by Weather Condition", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Avg Units Sold")
    for i, row in w.iterrows():
        ax.text(i, row["avg_quantity"] + 1, f"{row['avg_quantity']:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    return fig


def plot_revenue_by_type(df: pd.DataFrame):
    rt = eda_revenue_by_type(df)
    fig, ax = _fig(7, 3.5)
    bars = ax.bar(rt["restaurant_type"], rt["total_revenue"],
                  color=sns.color_palette("Blues_d", len(rt)), edgecolor="white")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"RM {x/1e6:.1f}M"))
    ax.set_title("Total Revenue by Restaurant Type", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (RM)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def plot_margin_by_item(df: pd.DataFrame):
    margin = eda_margin_by_item(df)
    fig, ax = _fig(8, 4)
    colors = [PALETTE["success"] if m >= 60 else PALETTE["warning"] if m >= 40 else PALETTE["danger"]
              for m in margin["avg_margin_pct"]]
    ax.barh(margin["menu_item_name"][::-1], margin["avg_margin_pct"][::-1], color=colors[::-1])
    ax.set_title("Average Gross Margin % by Menu Item", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Avg Gross Margin (%)")
    ax.axvline(margin["avg_margin_pct"].mean(), color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_price_vs_quantity(df: pd.DataFrame):
    sample = df[df["quantity_sold"] > 0].sample(min(2000, len(df)), random_state=42)
    fig, ax = _fig(7, 4)
    sc = ax.scatter(
        sample["actual_selling_price"],
        sample["quantity_sold"],
        c=sample["estimated_gross_margin"],
        cmap="RdYlGn",
        alpha=0.45, s=12,
    )
    plt.colorbar(sc, ax=ax, label="Gross Margin (RM)")
    ax.set_title("Selling Price vs Quantity Sold\n(colour = Gross Margin)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Actual Selling Price (RM)")
    ax.set_ylabel("Quantity Sold")
    plt.tight_layout()
    return fig


def plot_market_vs_actual(df: pd.DataFrame):
    sample = df[df["quantity_sold"] > 0].sample(min(2000, len(df)), random_state=42)
    fig, ax = _fig(7, 4)
    ax.scatter(sample["observed_market_price"], sample["actual_selling_price"],
               alpha=0.3, s=10, color=PALETTE["primary"])
    lims = [0, max(sample["observed_market_price"].max(), sample["actual_selling_price"].max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Price = Market")
    ax.set_title("Market Price vs Actual Selling Price", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Observed Market Price (RM)")
    ax.set_ylabel("Actual Selling Price (RM)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted Demand"):
    fig, ax = _fig(7, 5)
    ax.scatter(y_test, y_pred, alpha=0.35, s=12, color=PALETTE["primary"])
    lims = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect Prediction")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Actual Quantity Sold")
    ax.set_ylabel("Predicted Quantity Sold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_ml_features(df: pd.DataFrame) -> tuple:
    """
    Encode features and return the time-aware train/test split.
    Uses time-aware split: last 20% of dates as test set.
    Avoids target leakage: no revenue/margin columns in features.
    """
    df_ml = df.copy()

    # ── Encode categoricals ──────────────────────────────────────────────────
    cat_cols = ["restaurant_type", "menu_item_name", "meal_type", "weather_condition"]
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col + "_enc"] = le.fit_transform(df_ml[col])

    # ── Feature set (NO revenue/margin → no leakage) ─────────────────────────
    feature_cols = [
        "restaurant_id",
        "restaurant_type_enc",
        "menu_item_name_enc",
        "meal_type_enc",
        "weather_condition_enc",
        "actual_selling_price",
        "observed_market_price",
        "typical_ingredient_cost",
        "has_promotion",
        "special_event",
        "month",
        "day_of_week",
        "week",
        "is_weekend",
    ]

    X = df_ml[feature_cols].copy()
    X["has_promotion"] = X["has_promotion"].astype(int)
    X["special_event"] = X["special_event"].astype(int)
    X["is_weekend"]    = X["is_weekend"].astype(int)

    y = df_ml["quantity_sold"]

    # ── Time-aware split: train on first 80% of sorted dates ─────────────────
    split_idx = int(len(df_ml) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the training split."""
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return evaluation metrics for the trained model."""
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    return {
        "MAE":  mae,
        "RMSE": rmse,
        "R2":   r2,
        "y_pred": y_pred,
        "y_test": y_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DYNAMIC BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> list:
    """
    Generate plain-language insights from the actual (filtered) data.
    Returns a list of insight strings.
    """
    insights = []

    # ── Top revenue item ─────────────────────────────────────────────────────
    item_rev = eda_revenue_by_item(df)
    if not item_rev.empty:
        top = item_rev.iloc[0]
        insights.append(
            f"**Top Revenue Item:** {top['menu_item_name']} generated "
            f"RM {top['total_revenue']:,.0f} in total revenue "
            f"({top['total_quantity']:,} units sold)."
        )

    # ── Best margin item ──────────────────────────────────────────────────────
    margin_df = eda_margin_by_item(df)
    if not margin_df.empty:
        best_margin = margin_df.iloc[0]
        insights.append(
            f"**Highest Margin Item:** {best_margin['menu_item_name']} has the best "
            f"average gross margin at {best_margin['avg_margin_pct']:.1f}%."
        )

    # ── Lowest revenue item ───────────────────────────────────────────────────
    if len(item_rev) > 1:
        low = item_rev.iloc[-1]
        insights.append(
            f"**Lowest Revenue Item:** {low['menu_item_name']} has the lowest revenue "
            f"(RM {low['total_revenue']:,.0f}). Consider promotion or menu placement changes."
        )

    # ── Promotion lift ────────────────────────────────────────────────────────
    # Compute directly from df to avoid groupby bool-index issues
    promo_avg   = df.loc[df["has_promotion"],  "quantity_sold"].mean()
    nopromo_avg = df.loc[~df["has_promotion"], "quantity_sold"].mean()
    if pd.notna(promo_avg) and pd.notna(nopromo_avg) and nopromo_avg > 0:
        lift = ((promo_avg - nopromo_avg) / nopromo_avg) * 100
        direction = "higher" if lift >= 0 else "lower"
        insights.append(
            f"**Promotion Effect:** Promoted items sell {abs(lift):.1f}% {direction} "
            f"on average ({promo_avg:.1f} vs {nopromo_avg:.1f} units/row)."
        )

    # ── Special event lift ────────────────────────────────────────────────────
    ev_avg   = df.loc[df["special_event"],  "quantity_sold"].mean()
    noev_avg = df.loc[~df["special_event"], "quantity_sold"].mean()
    if pd.notna(ev_avg) and pd.notna(noev_avg) and noev_avg > 0:
        lift = ((ev_avg - noev_avg) / noev_avg) * 100
        direction = "higher" if lift >= 0 else "lower"
        insights.append(
            f"**Special Event Effect:** Special event days see demand that is "
            f"{abs(lift):.1f}% {direction} than normal days "
            f"({ev_avg:.1f} vs {noev_avg:.1f} units/row)."
        )

    # ── Best weather ──────────────────────────────────────────────────────────
    weather = eda_weather_effect(df)
    if not weather.empty:
        best_w = weather.sort_values("avg_quantity", ascending=False).iloc[0]
        insights.append(
            f"**Best Weather for Sales:** '{best_w['weather_condition']}' weather correlates "
            f"with the highest average demand ({best_w['avg_quantity']:.1f} units/row)."
        )

    # ── Best restaurant type ──────────────────────────────────────────────────
    rt = eda_revenue_by_type(df)
    if not rt.empty:
        top_type = rt.iloc[0]
        insights.append(
            f"**Top Restaurant Type by Revenue:** {top_type['restaurant_type']} "
            f"leads with RM {top_type['total_revenue']:,.0f} in total revenue."
        )

    # ── Weekend vs weekday ────────────────────────────────────────────────────
    wkd = df.loc[~df["is_weekend"], "quantity_sold"].mean()
    wke = df.loc[ df["is_weekend"], "quantity_sold"].mean()
    if pd.notna(wkd) and pd.notna(wke) and wkd > 0:
        diff = ((wke - wkd) / wkd * 100)
        direction = "higher" if diff >= 0 else "lower"
        insights.append(
            f"**Weekday vs Weekend:** Weekend demand is {abs(diff):.1f}% {direction} "
            f"than weekdays on average ({wke:.1f} vs {wkd:.1f} units/row)."
        )

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PIPELINE RUNNER  (for --pipeline-only mode)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Run the full analytics pipeline and print results to console."""
    print("\n" + "=" * 60)
    print("  Restaurant Revenue & Demand Analytics - Pipeline")
    print("=" * 60)

    print("\n[1] Loading data ...")
    raw = load_data()
    print(f"    Loaded {len(raw):,} rows x {len(raw.columns)} columns")

    print("\n[2] Auditing data ...")
    audit = audit_data(raw)
    print(f"    Missing values   : {audit['total_missing']}")
    print(f"    Duplicate rows   : {audit['duplicate_rows']}")
    print(f"    Duplicate keys   : {audit['duplicate_keys']}")
    print(f"    Zero-qty rows    : {audit['zero_quantity']}")
    print(f"    Unique restaurants: {audit['unique_restaurants']}")
    print(f"    Unique menu items : {audit['unique_items']}")

    print("\n[3] Cleaning data ...")
    df = clean_data(raw)

    print("\n[4] Engineering features ...")
    df = engineer_features(df)

    print("\n[5] KPIs ...")
    kpis = calculate_kpis(df)
    print(f"    Total Revenue        : RM {kpis['total_revenue']:>14,.2f}")
    print(f"    Total Quantity Sold  : {kpis['total_quantity']:>15,}")
    print(f"    Avg Selling Price    : RM {kpis['avg_selling_price']:>14,.2f}")
    print(f"    Avg Gross Margin %   : {kpis['avg_margin_pct']:>14.1f}%")
    print(f"    Total Gross Margin   : RM {kpis['total_gross_margin']:>14,.2f}")

    print("\n[6] Training ML model (this may take ~30 s) ...")
    X_train, X_test, y_train, y_test = prepare_ml_features(df)
    model  = train_model(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    print(f"    MAE  : {result['MAE']:.2f}")
    print(f"    RMSE : {result['RMSE']:.2f}")
    print(f"    R2   : {result['R2']:.4f}")

    print("\n[7] Business Insights ...")
    insights = generate_insights(df)
    for i, ins in enumerate(insights, 1):
        # Strip markdown bold markers for console output
        clean = ins.replace("**", "")
        print(f"    {i}. {clean}")

    print("\nPipeline complete.\n")
    return df, model, result


# ═══════════════════════════════════════════════════════════════════════════════
# 11. STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_prepare() -> pd.DataFrame:
    """Cached: load → clean → engineer features."""
    raw = load_data()
    df  = clean_data(raw)
    df  = engineer_features(df)
    return df


@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame):
    """Cached: train model once and reuse."""
    X_train, X_test, y_train, y_test = prepare_ml_features(df)
    model  = train_model(X_train, y_train)
    result = evaluate_model(model, X_test, y_test)
    return model, result


def kpi_card(label: str, value: str, delta: str = ""):
    """Render a compact KPI metric card."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def sidebar_filters(df: pd.DataFrame):
    """Render sidebar filters and return filtered DataFrame."""
    st.sidebar.header("🔎 Filters")

    # Date range
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    # Restaurant type
    rest_types = ["All"] + sorted(df["restaurant_type"].unique().tolist())
    sel_type   = st.sidebar.multiselect(
        "Restaurant Type",
        options=rest_types[1:],
        default=[],
        placeholder="All types",
    )

    # Menu item
    items = sorted(df["menu_item_name"].unique().tolist())
    sel_items = st.sidebar.multiselect(
        "Menu Item",
        options=items,
        default=[],
        placeholder="All items",
    )

    # Meal type
    meal_types = sorted(df["meal_type"].unique().tolist())
    sel_meals  = st.sidebar.multiselect(
        "Meal Type",
        options=meal_types,
        default=[],
        placeholder="All meal types",
    )

    # Weather
    weathers = sorted(df["weather_condition"].unique().tolist())
    sel_weather = st.sidebar.multiselect(
        "Weather",
        options=weathers,
        default=[],
        placeholder="All weather",
    )

    # ── Apply filters ──────────────────────────────────────────────────────
    filtered = df.copy()
    filtered = filtered[
        (filtered["date"].dt.date >= start_d) &
        (filtered["date"].dt.date <= end_d)
    ]
    if sel_type:
        filtered = filtered[filtered["restaurant_type"].isin(sel_type)]
    if sel_items:
        filtered = filtered[filtered["menu_item_name"].isin(sel_items)]
    if sel_meals:
        filtered = filtered[filtered["meal_type"].isin(sel_meals)]
    if sel_weather:
        filtered = filtered[filtered["weather_condition"].isin(sel_weather)]
    row_count = len(filtered)
    st.sidebar.markdown(f"**{row_count:,} rows** match the current filters.")
    if row_count == 0:
        st.sidebar.warning("No data matches these filters. Adjust selections above.")

    return filtered


def page_executive_overview(df: pd.DataFrame):
    """PAGE 1 — Executive Overview"""
    st.header("📊 Executive Overview")
    st.caption("High-level KPIs and trends across the selected time window and filters.")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    kpis = calculate_kpis(df)

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Revenue",    f"RM {kpis['total_revenue']:,.0f}")
    with c2: kpi_card("Units Sold",       f"{kpis['total_quantity']:,}")
    with c3: kpi_card("Gross Margin",     f"RM {kpis['total_gross_margin']:,.0f}")
    with c4: kpi_card("Avg Sell Price",   f"RM {kpis['avg_selling_price']:.2f}")
    with c5: kpi_card("Restaurants",      f"{kpis['n_restaurants']}")
    with c6: kpi_card("Menu Items",       f"{kpis['n_menu_items']}")

    st.markdown("---")

    # ── Trends ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Revenue Trend")
        st.pyplot(plot_revenue_trend(df))
        st.caption("Monthly total revenue across all restaurants and items in the current filter selection.")
    with col2:
        st.subheader("📦 Demand Trend")
        st.pyplot(plot_demand_trend(df))
        st.caption("Weekly total units sold. Peaks may coincide with promotions or special events.")

    # ── Top items ─────────────────────────────────────────────────────────────
    st.subheader("🏆 Top Menu Items by Revenue")
    top_n = st.slider("Show top N items", 3, 14, 10, key="top_n_exec")
    st.pyplot(plot_top_items_revenue(df, top_n))

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("📋 Item Summary Table")
    item_df = eda_revenue_by_item(df)
    item_df.columns = ["Menu Item", "Total Revenue (RM)", "Total Units", "Avg Price (RM)", "Avg Margin %", "Transactions"]
    item_df["Total Revenue (RM)"] = item_df["Total Revenue (RM)"].round(2)
    item_df["Avg Price (RM)"]     = item_df["Avg Price (RM)"].round(2)
    item_df["Avg Margin %"]       = item_df["Avg Margin %"].round(1)
    st.dataframe(item_df.set_index("Menu Item"), width="stretch")


def page_demand_operations(df: pd.DataFrame, model_result: dict):
    """PAGE 2 — Demand & Operations"""
    st.header("⚙️ Demand & Operations")
    st.caption("Understand what drives demand: timing, promotions, events, and weather.")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # ── Demand drivers grid ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Day-of-Week Demand")
        st.pyplot(plot_day_of_week(df))
        st.caption("Average units sold per transaction record by weekday. Orange bars = weekends.")
    with col2:
        st.subheader("🍽️ Meal-Type Demand")
        st.pyplot(plot_meal_type(df))
        st.caption("Average demand across Breakfast, Lunch, and Dinner segments.")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.subheader("🎟️ Promotions")
        st.pyplot(plot_promo_effect(df))
    with col4:
        st.subheader("🎉 Special Events")
        st.pyplot(plot_event_effect(df))
    with col5:
        st.subheader("🌤️ Weather")
        st.pyplot(plot_weather_effect(df))

    st.caption(
        "Note: these are observational correlations, not causal claims. "
        "High demand on special events may also overlap with promotion activity."
    )

    st.markdown("---")

    # ── ML results ────────────────────────────────────────────────────────────
    st.subheader("🤖 Machine Learning — Demand Prediction")

    y_test = model_result["y_test"]
    y_pred = model_result["y_pred"]

    m1, m2, m3 = st.columns(3)
    with m1: kpi_card("MAE",  f"{model_result['MAE']:.2f} units")
    with m2: kpi_card("RMSE", f"{model_result['RMSE']:.2f} units")
    with m3: kpi_card("R²",   f"{model_result['R2']:.3f}")

    st.pyplot(plot_actual_vs_predicted(y_test, y_pred))
    st.caption(
        "Each point is a test-set transaction. The red dashed line is perfect prediction. "
        "A tighter cluster around the line = better model accuracy. "
        f"The model explains ~{model_result['R2']*100:.1f}% of demand variance "
        f"with an average error of {model_result['MAE']:.1f} units."
    )

    with st.expander("ℹ️ About the model"):
        st.markdown(
            """
            - **Algorithm:** Random Forest Regressor (150 trees, max depth 12)
            - **Target:** `quantity_sold`
            - **Split:** Time-aware — first 80 % of dates for training, last 20 % for testing
            - **Features used:** restaurant type, menu item, meal type, weather, prices,
              promotion flag, special event flag, month, day of week, week number, is_weekend
            - **Leakage prevention:** Revenue, cost, and margin columns are excluded from features
              because they are derived from quantity_sold itself.
            """
        )


def page_product_profitability(df: pd.DataFrame):
    """PAGE 3 — Product & Profitability"""
    st.header("💰 Product & Profitability")
    st.caption("Analyse margins, pricing, and revenue contribution by item and restaurant type.")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # ── Revenue by restaurant type ─────────────────────────────────────────
    st.subheader("🏬 Revenue by Restaurant Type")
    st.pyplot(plot_revenue_by_type(df))

    # ── Margin by menu item ────────────────────────────────────────────────
    st.subheader("📊 Gross Margin % by Menu Item")
    st.pyplot(plot_margin_by_item(df))
    st.caption(
        "Green = margin ≥ 60 %, Yellow = 40–60 %, Red = below 40 %. "
        "Higher margin items should be prioritised for upselling."
    )

    # ── Price vs quantity & market comparison ──────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💲 Price vs Quantity")
        st.pyplot(plot_price_vs_quantity(df))
        st.caption("Shows whether higher-priced items sell fewer units (demand elasticity signal).")
    with col2:
        st.subheader("📉 Market Price vs Actual Price")
        st.pyplot(plot_market_vs_actual(df))
        st.caption(
            "Points above the red dashed line = restaurant prices higher than market. "
            "Points below = priced below market (check for missing promotion flags)."
        )

    # ── Low-performing items ────────────────────────────────────────────────
    st.subheader("⚠️ Low-Performing Items")
    item_df = eda_revenue_by_item(df)
    threshold = item_df["total_revenue"].quantile(0.25)
    low = item_df[item_df["total_revenue"] <= threshold].copy()
    low.columns = ["Menu Item", "Total Revenue (RM)", "Total Units", "Avg Price (RM)", "Avg Margin %", "Transactions"]
    low["Total Revenue (RM)"] = low["Total Revenue (RM)"].round(2)
    low["Avg Margin %"]       = low["Avg Margin %"].round(1)
    st.dataframe(low.set_index("Menu Item"), width="stretch")
    st.caption(f"Items in the bottom 25th percentile of revenue (below RM {threshold:,.0f}).")

    # ── Full profitability table ────────────────────────────────────────────
    with st.expander("📋 Full Item Profitability Table"):
        full = eda_margin_by_item(df).copy()
        full.columns = ["Menu Item", "Avg Margin %", "Total Gross Margin (RM)"]
        full["Avg Margin %"]              = full["Avg Margin %"].round(1)
        full["Total Gross Margin (RM)"]   = full["Total Gross Margin (RM)"].round(2)
        st.dataframe(full.set_index("Menu Item"), width="stretch")


def page_insights(df: pd.DataFrame, kpis: dict):
    """Business Insights & Recommendations section."""
    st.header("💡 Business Insights & Recommendations")
    st.caption(
        "All insights below are generated dynamically from the actual data "
        "matching your current filters — no hard-coded findings."
    )

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    insights = generate_insights(df)
    for ins in insights:
        st.info(ins)

    # ── Recommendation bullets ─────────────────────────────────────────────
    st.markdown("### 📌 Recommended Actions")
    item_df = eda_revenue_by_item(df)
    margin_df = eda_margin_by_item(df)

    if not item_df.empty and not margin_df.empty:
        top_item   = item_df.iloc[0]["menu_item_name"]
        worst_item = item_df.iloc[-1]["menu_item_name"]
        best_margin_item = margin_df.iloc[0]["menu_item_name"]

        promo_yes = df.loc[ df["has_promotion"], "quantity_sold"].mean()
        promo_no  = df.loc[~df["has_promotion"], "quantity_sold"].mean()
        promo_works = (pd.notna(promo_yes) and pd.notna(promo_no) and promo_yes > promo_no)

        st.markdown(
            f"""
1. **Double down on {top_item}** — the top revenue generator. Ensure consistent ingredient supply
   and consider premium or value variants to capture more price segments.

2. **Promote {best_margin_item}** more aggressively — it has the highest gross margin percentage.
   Upselling this item boosts profitability without increasing volume.

3. **Investigate {worst_item}** — lowest revenue in the current selection.
   Consider whether it belongs on the menu, needs re-pricing, or needs better placement.

4. **{"Run targeted promotions" if promo_works else "Review promotion strategy"}** —
   {"the data shows promotions are associated with higher average demand. Expand promo days for low-demand periods." if promo_works else "promoted items do not show higher average demand in this selection. Evaluate whether promotions are reaching the right customers."}

5. **Zero-quantity rows** ({kpis['zero_qty_rows']:,} records) represent days with no sales.
   Analyse these to identify slow periods and target operational improvements there.
"""
        )


def run_dashboard():
    """Entry point for the Streamlit dashboard."""

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Restaurant Revenue & Demand Analytics",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        [data-testid="metric-container"] {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 14px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🍽️ Restaurant Revenue & Demand Analytics")
    st.markdown(
        "**IBM SkillsBuild Final Project** &nbsp;|&nbsp; "
        "Interactive analytics dashboard for restaurant sales data."
    )
    st.markdown("---")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading and preparing data..."):
        try:
            df_full = load_and_prepare()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    # ── Train model (cached) ──────────────────────────────────────────────────
    with st.spinner("Training demand prediction model (first run only)..."):
        model, model_result = get_model(df_full)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    df_filtered = sidebar_filters(df_full)
    kpis = calculate_kpis(df_filtered)

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Overview",
        "⚙️ Demand & Operations",
        "💰 Product & Profitability",
        "💡 Insights & Recommendations",
    ])

    with tab1:
        page_executive_overview(df_filtered)

    with tab2:
        page_demand_operations(df_filtered, model_result)

    with tab3:
        page_product_profitability(df_filtered)

    with tab4:
        page_insights(df_filtered, kpis)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Dataset: {len(df_full):,} rows · "
        f"{df_full['date'].min().strftime('%d %b %Y')} – "
        f"{df_full['date'].max().strftime('%d %b %Y')} · "
        "IBM SkillsBuild Final Project"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--pipeline-only" in sys.argv:
        run_pipeline()
    else:
        # When launched via `streamlit run`, Streamlit calls the script top-to-bottom;
        # the presence of st calls outside __main__ is fine, but we guard the entry
        # point so that python restaurant_demand_analytics.py still works.
        run_dashboard()
else:
    # Called by `streamlit run restaurant_demand_analytics.py`
    run_dashboard()
