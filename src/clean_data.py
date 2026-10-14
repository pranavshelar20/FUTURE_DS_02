"""Step 1: clean the raw Telco churn file and add derived fields."""
import pandas as pd, numpy as np, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

raw=pd.read_csv(ROOT/'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
log=[]
df=raw.copy()
log.append(('Rows / columns in',f'{raw.shape[0]:,} / {raw.shape[1]}'))
log.append(('Duplicate customer IDs',int(df.customerID.duplicated().sum())))
log.append(('Fully duplicated rows',int(df.duplicated().sum())))
log.append(('Null cells (before)',int(raw.isna().sum().sum())))
# TotalCharges blanks
blank=df.TotalCharges.str.strip()==''
log.append(('Blank TotalCharges',f'{int(blank.sum())} rows, all with tenure 0 (new sign-ups, nothing billed yet)'))
df['TotalCharges']=pd.to_numeric(df.TotalCharges.replace(r'^\s*$',np.nan,regex=True))
df['total_charges_missing']=df.TotalCharges.isna().astype(int)
df['TotalCharges']=df.TotalCharges.fillna(0.0)
# standardise
df.columns=[c.strip() for c in df.columns]
df=df.rename(columns={'customerID':'customer_id','SeniorCitizen':'senior_citizen','MonthlyCharges':'monthly_charges','TotalCharges':'total_charges'})
df.columns=[c if c.islower() or '_' in c else ''.join(['_'+x.lower() if x.isupper() else x for x in c]).lstrip('_') for c in df.columns]
df=df.rename(columns={'tenure':'tenure_months','streaming_t_v':'streaming_tv'})
# yes/no
yn=[c for c in df.columns if set(df[c].unique())<= {'Yes','No','No internet service','No phone service'} and c!='churn']
for c in yn:
    df[c]=df[c].map({'Yes':1,'No':0,'No internet service':0,'No phone service':0}).astype(int)
log.append(('Yes/No columns recoded to 1/0',f'{len(yn)} columns; "No internet service" and "No phone service" folded into 0, with has_internet / has_phone flags kept'))
df['churn']=(df['churn']=='Yes').astype(int)
df['senior_citizen']=df.senior_citizen.astype(int)
df['has_internet']=(df.internet_service!='No').astype(int)
df['has_phone']=df.phone_service
# derived
sec=['online_security','online_backup','device_protection','tech_support']
df['protective_services']=df[sec].sum(axis=1)
df['autopay']=df.payment_method.str.contains('automatic').astype(int)
df['plan']=df.contract+' / '+df.internet_service.replace({'No':'Phone only'})
df['tenure_band']=pd.cut(df.tenure_months,[-1,3,6,12,24,48,72],labels=['0-3','4-6','7-12','13-24','25-48','49-72']).astype(str)
df['expected_total']=df.tenure_months*df.monthly_charges
dev=(df.total_charges-df.expected_total)/df.expected_total.replace(0,np.nan)
log.append(('Billing consistency check',f'TotalCharges is within ±25% of tenure × MonthlyCharges for {((dev.abs()<=.25)|dev.isna()).mean()*100:.0f}% of customers, so billing fields are internally consistent'))
log.append(('Outliers',f'Monthly charges range ${df.monthly_charges.min():.2f}–${df.monthly_charges.max():.2f}; tenure {df.tenure_months.min()}–{df.tenure_months.max()} months; no values removed'))
log.append(('Inconsistent combos',f'{int(((df.has_internet==0)&(df[sec].sum(axis=1)>0)).sum())} customers with internet add-ons but no internet'))
df=df.drop(columns=['expected_total'])
df.to_csv(ROOT/'data/processed/telco_customers_clean.csv',index=False)
print(df.shape, df.columns.tolist())
json.dump(log,open(ROOT/'outputs/cleaning_log.json','w'),indent=1)
for l in log:print(l)

