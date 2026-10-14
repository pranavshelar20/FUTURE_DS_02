# FUTURE_DS_02: Customer Retention and Churn Analysis

An analysis of 7,043 telecom subscription customers: where churn happens, what keeps customers, how long they stay, and what a subscription business should do about it. The result is an interactive dashboard plus a reproducible Python pipeline.

**Live dashboard:** https://pranavshelar20.github.io/FUTURE_DS_02/
(enable GitHub Pages: Settings, Pages, Deploy from branch `main`, folder `/docs`)

![Dashboard overview](images/01_overview.png)

## Headline findings

- **Churn is a first-year, month-to-month, fiber problem.** 26.5% of customers churned (1,869 of 7,043), taking $139K of monthly revenue (30.5% of the base). 55% of churners left within 12 months.
- **Contract type is the biggest divide.** Churn is 42.7% on month-to-month, 11.3% on one-year and 2.8% on two-year plans. A two-year contract cuts the odds of churn by 74% after controlling for other factors.
- **Support services protect.** Online security and tech support each cut the odds of churn by about a third, independent of contract, tenure and price. Fiber customers with neither churn at 55%, against 24% with at least one.
- **Payment and billing matter.** Electronic check payers churn at 45% against 15% to 17% for autopay. Paperless billing and senior customers also run high.
- **Lifetime gap.** Over five years a month-to-month customer stays about 33 months against 57 for a one-year customer, roughly $1,500 less revenue per customer.
- **Risk stacks.** Customers with none of five risk factors churn at 2.5%. Those with all five churn at 74%.
- **One segment dominates.** Month-to-month fiber customers without security or tech support are 22% of customers and 49% of churners.

## What the dashboard covers

| Section | Contents |
|---|---|
| Churn patterns | Monthly churn by tenure, churn by contract, service, payment, billing, age; contract by tenure heatmap |
| Retention drivers | Adjusted odds ratios with bootstrap intervals; protective-service effect; factors that do not matter |
| Customer lifetime | Kaplan-Meier retention curves by segment; five-year expected revenue per customer |
| Cohorts | Retention at 6, 12, 24, 36 months by plan cohort; first-year funnel; age mix of the active base |
| Churn reasons | Inferred drivers and how risk compounds across factors |
| Where churn sits | Highest-impact segments; interactive churn-rate lookup for any customer profile |
| Recommendations | Five plays with targets, success metrics, effort, estimated impact and a 12-week sequence |
| Method | Data-preparation log, coverage table, caveats |

| Cohorts | Churn reasons |
|---|---|
| ![Cohorts](images/02_cohorts.png) | ![Churn reasons](images/03_churn_reasons.png) |

## Recommendations in brief

1. Move month-to-month customers onto 12-month plans (price lock plus a free security trial).
2. Bundle security and tech support into fiber plans.
3. Get electronic check payers onto autopay.
4. Fix the first 90 days with scheduled touchpoints, and run a fiber diagnostic.
5. Build a churn-risk score and give the save team a short list.

Plays 1 to 4 are sized at roughly 47 to 224 churners avoided each (a half-gap case and a full-gap case, not additive). Details are in the dashboard.

## Repository structure

```
FUTURE_DS_02/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md                  data dictionary and source
│   ├── raw/                       original dataset, unmodified
│   └── processed/                 cleaned dataset
├── src/
│   ├── clean_data.py              step 1: cleaning and derived fields
│   ├── analysis.py                step 2: survival, drivers, cohorts, scenarios, model
│   ├── build_dashboard.py         step 3: builds the dashboard page
│   └── templates/dashboard_template.html
├── outputs/                       JSON results and cleaning log
├── docs/index.html                the dashboard (served by GitHub Pages)
└── images/                        screenshots used in this README
```

## Reproduce it

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/FUTURE_DS_02.git
cd FUTURE_DS_02
pip install -r requirements.txt
python src/clean_data.py
python src/analysis.py
python src/build_dashboard.py
```

Then open `docs/index.html` in a browser. Web fonts load from Google Fonts when online and fall back to system fonts offline.

## Method

- **Cleaning.** No duplicates or nulls. 11 blank total-charges values (all new customers with zero tenure) were set to 0 and flagged. Yes/No columns were recoded to 1/0, and fields such as plan, protective-service count and autopay were derived. Full log in `outputs/cleaning_log.json` and in the dashboard.
- **Retention curves.** Kaplan-Meier survival, which handles customers who are still active. Curves stop where fewer than 40 customers remain at risk.
- **Lifetime value.** Average months retained within 60 months times average monthly bill. Revenue only, no margin or acquisition cost.
- **Drivers.** Logistic regression on tenure, price, services, contract, payment, billing and demographics, with 95% intervals from 200 bootstrap resamples.
- **Risk model.** Gradient boosting on a 75/25 split: 0.85 AUC; the top 20% of scores hold 49% of churners.
- **Impact estimates.** Observed churn gaps applied to a share of each segment, shown at half the gap and at the full gap.

## Limitations

- One snapshot with no dates, so trends are by customer age, not calendar month. Signup-month and regional cohorts are not possible.
- No stated churn reasons. The reasons shown are inferred from traits churners share and should be confirmed with exit surveys.
- Gaps between groups are observational, not proven causes. Every play should be tested against a holdout group.
- The 0.85 AUC is probably optimistic because tenure partly encodes the outcome in a single snapshot. Validate on forward-looking data.
- It is a sample dataset. Rerun the pipeline on production data before committing budget.

## Data source and license

Data: Telco Customer Churn sample dataset (see `data/README.md`). Code: MIT License.
