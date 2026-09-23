# Restaurant Revenue & Demand Analytics

**IBM SkillsBuild Final Project**

An end-to-end analytics pipeline and interactive Streamlit dashboard built on a real restaurant sales dataset. The project covers data cleaning, feature engineering, KPI calculations, exploratory data analysis, machine learning demand prediction, and dynamic business insights — all in a single Python file.

---

## Project Structure

```
ibm/
├── restaurant_demand_analytics.py   ← Main Python file (pipeline + dashboard)
├── restaurant_sales_data.csv        ← Dataset (10,000 rows × 13 columns)
├── requirements.txt                 ← Python dependencies
└── README.md                        ← This file
```

---

## Dataset

| Property | Value |
|---|---|
| Rows | 10,000 |
| Columns | 13 |
| Date range | 1 Jan 2024 – 1 Jan 2025 |
| Restaurants | 50 |
| Menu items | 14 |
| Missing values | 0 |

**Columns:** `date, restaurant_id, restaurant_type, menu_item_name, meal_type, key_ingredients_tags, typical_ingredient_cost, observed_market_price, actual_selling_price, quantity_sold, has_promotion, special_event, weather_condition`

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run restaurant_demand_analytics.py
```

Open your browser at **http://localhost:8501**

### 3. Run the pipeline only (console output, no UI)

```bash
python restaurant_demand_analytics.py --pipeline-only
```

---

## Dashboard Pages

| Page | Contents |
|---|---|
| **Executive Overview** | Total revenue, units sold, gross margin, avg price, revenue & demand trends, top items table |
| **Demand & Operations** | Day-of-week demand, meal-type demand, promotion effect, special event effect, weather effect, ML actual vs predicted |
| **Product & Profitability** | Revenue by restaurant type, gross margin % by item, price vs quantity, market price vs actual price, low-performing items |
| **Insights & Recommendations** | Dynamically generated insights and action items from the filtered data |

All pages respond to the **sidebar filters**: date range, restaurant type, menu item, meal type, and weather.

---

## Pipeline Stages

1. **Data Loading** — reads `restaurant_sales_data.csv`
2. **Data Quality Audit** — counts missing values, duplicates, zero-qty rows
3. **Data Cleaning** — parses dates, casts booleans and numerics, strips whitespace
4. **Feature Engineering** — revenue, estimated cost, gross margin, margin %, date features (year/month/day/week/day_of_week/is_weekend), price vs market, markup ratio
5. **KPI Calculations** — total revenue, total quantity, gross margin, avg price, avg margin %
6. **EDA** — revenue/demand by item, restaurant type, meal type, day-of-week, month, weather, promotions, special events
7. **Visualisations** — 14 matplotlib/seaborn charts
8. **Machine Learning** — RandomForestRegressor, time-aware 80/20 split, no target leakage
9. **Model Evaluation** — MAE, RMSE, R², actual vs predicted scatter plot
10. **Dynamic Business Insights** — generated from real filtered data at runtime

---

## Machine Learning Details

- **Algorithm:** `RandomForestRegressor` (150 trees, max depth 12)
- **Target:** `quantity_sold`
- **Train/test split:** Time-aware (first 80 % of dates for training, last 20 % for testing)
- **Features:** restaurant type, menu item, meal type, weather, prices, promotion, special event, month, day of week, week number, is_weekend
- **Leakage prevention:** Revenue, cost, and margin columns excluded from features
- **Typical performance:** MAE ≈ 78 units, R² ≈ 0.76

---

## Configuring the CSV Path

If your CSV is in a different location, edit line 61 of `restaurant_demand_analytics.py`:

```python
CSV_PATH = "restaurant_sales_data.csv"   # ← change this
```

---

## Notes

- Zero-quantity rows (502) are retained as instructed; they represent listed items with no sales on a given day.
- All insights are generated dynamically — no hard-coded analytical conclusions.
- Observational correlations (weather, promotions, events) are clearly labelled as correlations, not causal claims.
