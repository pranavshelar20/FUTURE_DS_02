"""Step 2: retention analysis. Survival curves, hazard, drivers, scenarios, cohorts."""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df=pd.read_csv(ROOT/'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges']=pd.to_numeric(df.TotalCharges,errors='coerce').fillna(0)
df['c']=(df.Churn=='Yes').astype(int)
N=len(df); CH=int(df.c.sum()); base=df.c.mean()
D={}
D['kpi']=dict(n=N,churned=CH,rate=base,mrr=float(df.MonthlyCharges.sum()),mrr_lost=float(df[df.c==1].MonthlyCharges.sum()),
  arpu_churn=float(df[df.c==1].MonthlyCharges.mean()),arpu_stay=float(df[df.c==0].MonthlyCharges.mean()),
  med_churn_tenure=float(df[df.c==1].tenure.median()),share12=float((df[df.c==1].tenure<=12).mean()),share6=float((df[df.c==1].tenure<=6).mean()),
  share3=float((df[df.c==1].tenure<=3).mean()))

def km(d,tmax=72,minrisk=40):
    S=1;out=[1.0];risk=[len(d)]
    for t in range(1,tmax+1):
        ar=(d.tenure>=t).sum(); ev=((d.tenure==t)&(d.c==1)).sum()
        if ar>0: S*=1-ev/ar
        out.append(S);risk.append(int(ar))
    cut=max([t for t in range(tmax+1) if risk[t]>=minrisk])
    return [round(float(v),4) for v in out[:cut+1]], cut
def rmst(vals,T=60):
    v=vals[:T+1]
    return float(sum(v[:-1])) if len(v)>T else None

inet=df[df.InternetService!='No']
groups={
 'contract':[('Month-to-month',df[df.Contract=='Month-to-month'],'s1'),('One year',df[df.Contract=='One year'],'s2'),('Two year',df[df.Contract=='Two year'],'s3')],
 'internet':[('Fiber optic',df[df.InternetService=='Fiber optic'],'s1'),('DSL',df[df.InternetService=='DSL'],'s4'),('Phone only',df[df.InternetService=='No'],'s3')],
 'payment':[('Electronic check',df[df.PaymentMethod=='Electronic check'],'s1'),('Mailed check',df[df.PaymentMethod=='Mailed check'],'s2'),('Bank transfer (auto)',df[df.PaymentMethod=='Bank transfer (automatic)'],'s4'),('Credit card (auto)',df[df.PaymentMethod=='Credit card (automatic)'],'s3')],
 'support':[('No tech support',inet[inet.TechSupport=='No'],'s1'),('Has tech support',inet[inet.TechSupport=='Yes'],'s3')],
 'senior':[('Senior',df[df.SeniorCitizen==1],'s1'),('Non-senior',df[df.SeniorCitizen==0],'s4')],
}
D['km']={}
D['rmst']=[]
for k,lst in groups.items():
    D['km'][k]=[]
    for name,g,col in lst:
        s,cut=km(g)
        D['km'][k].append(dict(name=name,color=col,n=len(g),churn=float(g.c.mean()),s=s,cut=cut))
        if k in('contract','internet'):
            r=rmst(s,60); arpu=float(g.MonthlyCharges.mean())
            D['rmst'].append(dict(group=k,name=name,n=len(g),months=round(r,1),arpu=round(arpu,2),rev=round(r*arpu),churn=float(g.c.mean()),mean_tenure=float(g.tenure.mean())))
# hazard
b=[0,3,6,12,24,36,48,60,72];hz=[]
for lo,hi in zip(b[:-1],b[1:]):
    sub=df[df.tenure>lo]; expo=float(np.clip(np.minimum(sub.tenure,hi)-lo,0,None).sum())
    ev=int(((df.c==1)&(df.tenure>lo)&(df.tenure<=hi)).sum())
    hz.append(dict(label=f'{lo+1}–{hi}',events=ev,rate=ev/expo*100))
D['hazard']=hz
# rates
def rt(col,order=None,labels=None):
    g=df.groupby(col).c.agg(['size','mean'])
    rows=[]
    for k in (order or g.index):
        rows.append(dict(label=(labels or {}).get(k,k),n=int(g.loc[k,'size']),rate=float(g.loc[k,'mean'])))
    return rows
D['rates']={
 'contract':rt('Contract',['Month-to-month','One year','Two year']),
 'internet':rt('InternetService',['Fiber optic','DSL','No'],{'No':'Phone only (no internet)'}),
 'payment':rt('PaymentMethod',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']),
 'billing':rt('PaperlessBilling',['Yes','No'],{'Yes':'Paperless billing','No':'Paper billing'}),
 'senior':rt('SeniorCitizen',[1,0],{1:'Senior citizen',0:'Not senior'}),
}
# no-effect
_i=df[df.InternetService!='No']; _st=((_i.StreamingTV=='Yes')|(_i.StreamingMovies=='Yes'))
D['noeffect']=[
 dict(label='Female',n=int((df.gender=='Female').sum()),rate=float(df[df.gender=='Female'].c.mean())),
 dict(label='Male',n=int((df.gender=='Male').sum()),rate=float(df[df.gender=='Male'].c.mean())),
 dict(label='Has phone service',n=int((df.PhoneService=='Yes').sum()),rate=float(df[df.PhoneService=='Yes'].c.mean())),
 dict(label='No phone service',n=int((df.PhoneService=='No').sum()),rate=float(df[df.PhoneService=='No'].c.mean())),
 dict(label='Streams TV or movies (internet customers)',n=int(_st.sum()),rate=float(_i[_st].c.mean())),
 dict(label='No streaming (internet customers)',n=int((~_st).sum()),rate=float(_i[~_st].c.mean())),
]
# heat
bands=[(1,6),(7,12),(13,24),(25,48),(49,72)]
heat=[]
for ct in ['Month-to-month','One year','Two year']:
    row=[]
    for lo,hi in bands:
        g=df[(df.Contract==ct)&(df.tenure>=lo)&(df.tenure<=hi)]
        row.append(dict(n=len(g),rate=float(g.c.mean()) if len(g) else None))
    heat.append(dict(contract=ct,cells=row))
D['heat']=dict(bands=[f'{a}–{b} mo' for a,b in bands],rows=heat)
# addons
sec=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport']
inet=df[df.InternetService!='No'].copy(); inet['nsec']=(inet[sec]=='Yes').sum(axis=1)
D['addons']=[dict(k=int(k),n=int(len(g)),rate=float(g.c.mean())) for k,g in inet.groupby('nsec')]
# odds ratios
d=df.copy()
d['sec']=(d.OnlineSecurity=='Yes').astype(int); d['ts']=(d.TechSupport=='Yes').astype(int)
d['fiber']=(d.InternetService=='Fiber optic').astype(int); d['dsl']=(d.InternetService=='DSL').astype(int)
d['y1']=(d.Contract=='One year').astype(int); d['y2']=(d.Contract=='Two year').astype(int)
d['echeck']=(d.PaymentMethod=='Electronic check').astype(int); d['paperless']=(d.PaperlessBilling=='Yes').astype(int)
d['senior']=d.SeniorCitizen; d['partner']=(d.Partner=='Yes').astype(int); d['dep']=(d.Dependents=='Yes').astype(int)
X=d[['tenure','MonthlyCharges','sec','ts','fiber','dsl','y1','y2','echeck','paperless','senior','partner','dep']].copy()
X['tenure']=X.tenure/12; X['MonthlyCharges']=X.MonthlyCharges/10
fit=lambda a,b_:LogisticRegression(penalty=None,max_iter=5000).fit(a,b_)
bo=np.exp(fit(X,d.c).coef_[0]); rng=np.random.default_rng(1); bs=[]
for i in range(200):
    idx=rng.integers(0,len(d),len(d)); bs.append(np.exp(fit(X.iloc[idx],d.c.iloc[idx]).coef_[0]))
bs=np.array(bs)
names={'tenure':'Each extra year of tenure','MonthlyCharges':'Each +$10 in monthly bill','sec':'Has online security','ts':'Has tech support','fiber':'Fiber optic (vs phone-only)','dsl':'DSL (vs phone-only)','y1':'One-year contract (vs monthly)','y2':'Two-year contract (vs monthly)','echeck':'Pays by electronic check','paperless':'Paperless billing','senior':'Senior citizen','partner':'Has partner','dep':'Has dependents'}
orr=[dict(label=names[c],or_=float(bo[i]),lo=float(np.percentile(bs[:,i],2.5)),hi=float(np.percentile(bs[:,i],97.5))) for i,c in enumerate(X.columns)]
D['or']=sorted(orr,key=lambda r:r['or_'])
# segments
mm=df.Contract=='Month-to-month'; fb=df.InternetService=='Fiber optic'
segs=[
 ('Month-to-month + fiber, no security or tech support',mm&fb&(df.OnlineSecurity=='No')&(df.TechSupport=='No')),
 ('Fiber without tech support',fb&(df.TechSupport=='No')),
 ('Month-to-month, first 12 months',mm&(df.tenure<=12)),
 ('Month-to-month paying by electronic check',mm&(df.PaymentMethod=='Electronic check')),
 ('Senior citizens',df.SeniorCitizen==1),
]
D['segments']=[]
for nme,mk in segs:
    s=df[mk]
    D['segments'].append(dict(name=nme,n=int(len(s)),rate=float(s.c.mean()),churners=int(s.c.sum()),share=float(s.c.sum()/CH),arr=float(s[s.c==1].MonthlyCharges.sum()*12)))
# scenarios
m2m=df[mm]; A=0
for svc,g in m2m.groupby('InternetService'):
    r1=df[(df.Contract=='One year')&(df.InternetService==svc)].c.mean(); A+=len(g)*.20*(g.c.mean()-r1)
f0=df[fb&(df.TechSupport=='No')&(df.OnlineSecurity=='No')]; f1=df[fb&((df.TechSupport=='Yes')|(df.OnlineSecurity=='Yes'))]
Bv=len(f0)*.25*(f0.c.mean()-f1.c.mean())
e=df[mm&(df.PaymentMethod=='Electronic check')]; au=df[mm&df.PaymentMethod.str.contains('automatic')]
Cv=len(e)*.25*(e.c.mean()-au.c.mean())
first3=int(((df.c==1)&(df.tenure<=3)).sum())
arpu=D['kpi']['arpu_churn']
sc=[
 dict(name='Move 20% of month-to-month customers to a 1-year plan',full=A,low=A/2,gap=f"{m2m.c.mean()*100:.0f}% vs {df[df.Contract=='One year'].c.mean()*100:.0f}% churn"),
 dict(name='Bundle security or tech support for 25% of unprotected fiber customers',full=Bv,low=Bv/2,gap=f"{f0.c.mean()*100:.0f}% vs {f1.c.mean()*100:.0f}% churn"),
 dict(name='Move 25% of month-to-month e-check payers to autopay',full=Cv,low=Cv/2,gap=f"{e.c.mean()*100:.0f}% vs {au.c.mean()*100:.0f}% churn"),
 dict(name='Cut first-90-day churn by 10–20%',full=first3*.2,low=first3*.1,gap=f"{first3:,} churners leave in months 1–3"),
]
for s in sc:
    s['arr_low']=s['low']*arpu*12; s['arr_full']=s['full']*arpu*12
D['scen']=sc
# model
Xm=pd.get_dummies(df.drop(columns=['customerID','Churn','c']),drop_first=True)
Xtr,Xte,ytr,yte=train_test_split(Xm,df.c,test_size=.25,random_state=42,stratify=df.c)
gb=GradientBoostingClassifier(random_state=42).fit(Xtr,ytr)
p=gb.predict_proba(Xte)[:,1]
auc=roc_auc_score(yte,p)
o=np.argsort(-p); k20=int(len(p)*.2); k10=int(len(p)*.1)
yy=yte.values[o]
D['model']=dict(auc=float(auc),top20_rate=float(yy[:k20].mean()),top20_capture=float(yy[:k20].sum()/yy.sum()),top10_rate=float(yy[:k10].mean()),top10_capture=float(yy[:k10].sum()/yy.sum()),test_n=int(len(p)))
# lookup
def sup(r):
    return 'na' if r.InternetService=='No' else ('yes' if r.TechSupport=='Yes' else 'no')
df['sup']=df.apply(sup,axis=1)
lk={}
for (ct,isv,pm,sp),g in df.groupby(['Contract','InternetService','PaymentMethod','sup']):
    lk[f'{ct}|{isv}|{pm}|{sp}']=[int(len(g)),round(float(g.c.mean()),4)]
D['lookup']=lk
json.dump(D,open(ROOT/'outputs/results_core.json','w'))


# ===== Step 2b: cohorts and inferred drivers (uses the cleaned data) =====
df=pd.read_csv(ROOT/'data/processed/telco_customers_clean.csv')
log=json.load(open(ROOT/'outputs/cleaning_log.json'))
# ---- cohorts by plan
def km2(d,tmax=72):
    S=1;out=[1.0]
    for t in range(1,tmax+1):
        ar=(d.tenure_months>=t).sum();ev=((d.tenure_months==t)&(d.churn==1)).sum()
        if ar>0:S*=1-ev/ar
        out.append(S)
    return out
def ret_at(d,m):
    if (d.tenure_months>=m).sum()<30: return None
    return round(float(km2(d,m)[m]),4)
coh=[]
for (ct,isv),g in df.groupby(['contract','internet_service']):
    coh.append(dict(plan=f"{ct} / {'Phone only' if isv=='No' else isv}",n=len(g),churn=float(g.churn.mean()),arpu=float(g.monthly_charges.mean()),
        r6=ret_at(g,6),r12=ret_at(g,12),r24=ret_at(g,24),r36=ret_at(g,36),churners=int(g.churn.sum()),arr=float(g[g.churn==1].monthly_charges.sum()*12)))
coh.sort(key=lambda r:-r['churners'])
# vintage mix of active customers by tenure band
act=df[df.churn==0]
vint=[dict(band=b,active=int((act.tenure_band==b).sum())) for b in ['0-3','4-6','7-12','13-24','25-48','49-72']]
# first-year funnel by contract: share still active at 3/6/12 months
fun=[]
for ct,g in df.groupby('contract'):
    fun.append(dict(contract=ct,r3=ret_at(g,3),r6=ret_at(g,6),r12=ret_at(g,12)))
# ---- inferred drivers
flags={
 'On a month-to-month contract':df.contract=='Month-to-month',
 'Paying by electronic check':df.payment_method=='Electronic check',
 'Fiber optic service':df.internet_service=='Fiber optic',
 'No tech support or online security (internet customers)':(df.has_internet==1)&(df.tech_support==0)&(df.online_security==0),
 'In first 12 months of tenure':df.tenure_months<=12,
 'Paperless billing':df.paperless_billing==1,
 'Bill above $80 a month':df.monthly_charges>80,
 'Senior citizen':df.senior_citizen==1,
}
ch=df[df.churn==1];st=df[df.churn==0]
drv=[]
for k,m in flags.items():
    pc=float(m[df.churn==1].mean());ps=float(m[df.churn==0].mean());rr=float(df[m].churn.mean()/df[~m].churn.mean())
    paf=pc*(1-1/rr) if rr>1 else 0
    drv.append(dict(label=k,churners=pc,retained=ps,rr=rr,paf=paf,rate=float(df[m].churn.mean())))
drv.sort(key=lambda r:-r['paf'])
# number of risk flags among churners
core=['On a month-to-month contract','Paying by electronic check','Fiber optic service','No tech support or online security (internet customers)','In first 12 months of tenure']
cnt=sum(flags[k].astype(int) for k in core)
rf=[dict(k=int(k),n=int((cnt==k).sum()),rate=float(df[cnt==k].churn.mean()),churners=int(df[cnt==k].churn.sum())) for k in range(6)]
out=dict(cohorts=coh,vintage=vint,funnel=fun,drivers=drv,riskflags=rf,clean=log)

json.dump(out,open(ROOT/'outputs/results_cohorts.json','w'),default=float)
print('Analysis complete: outputs/results_core.json, outputs/results_cohorts.json')
