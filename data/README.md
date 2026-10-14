# Data

| File | Description |
|---|---|
| `raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original Telco Customer Churn sample dataset: 7,043 customers, 21 columns. Unmodified. |
| `processed/telco_customers_clean.csv` | Cleaned file produced by `src/clean_data.py`: 7,043 rows, 28 columns. |

## Data dictionary (cleaned file)

| Column | Meaning |
|---|---|
| `customer_id` | Unique customer identifier |
| `gender`, `senior_citizen`, `partner`, `dependents` | Demographics (senior, partner, dependents coded 1/0) |
| `tenure_months` | Months as a customer. For churned customers, months before leaving. For active customers, months so far |
| `phone_service`, `multiple_lines` | Phone products (1/0) |
| `internet_service` | DSL, Fiber optic or No |
| `online_security`, `online_backup`, `device_protection`, `tech_support` | Protective add-ons (1/0) |
| `streaming_tv`, `streaming_movies` | Streaming add-ons (1/0) |
| `contract` | Month-to-month, One year or Two year |
| `paperless_billing` | 1/0 |
| `payment_method` | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `monthly_charges`, `total_charges` | Current monthly bill and cumulative billing, in dollars |
| `churn` | 1 if the customer left in the latest period, else 0 |
| `total_charges_missing` | 1 for the 11 new customers whose blank total charges were set to 0 |
| `has_internet`, `has_phone` | Product flags derived from the service columns |
| `protective_services` | Count (0 to 4) of security, backup, device protection and tech support |
| `autopay` | 1 if payment method is automatic |
| `plan` | Contract and internet service combined, for example `Month-to-month / Fiber optic` |
| `tenure_band` | 0-3, 4-6, 7-12, 13-24, 25-48 or 49-72 months |

## Not available in this dataset

Signup date, region and stated churn reasons. Analyses that need them are not included.

## Source

Telco Customer Churn sample dataset (IBM sample data, widely mirrored on Kaggle). It is a sample dataset, so figures describe this dataset and not any specific company.
