from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'backend'/'.packages'))
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
import joblib
from openpyxl import load_workbook
from openpyxl.styles import Font,PatternFill,Alignment
SEED=42; TARGET='accident_risk'
frame=pd.read_csv(ROOT.parent/'dataset'/'train.csv')
X=frame.drop(columns=['id',TARGET]); y=frame[TARGET].astype(float)
Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=.2,random_state=SEED)

# Same feature transforms as the existing experiment. Fit only on training fold.
def preprocessor(Xfit):
    cats=Xfit.select_dtypes(include=['object','category']).columns.tolist()
    nums=[c for c in Xfit.columns if c not in cats]
    return ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)])

def score(pred, secs, label, cfg, params, rounds=None):
    return {'Run':label,'Configuration':cfg,'MAE':mean_absolute_error(yva,pred),
      'RMSE':float(np.sqrt(mean_squared_error(yva,pred))),'R²':r2_score(yva,pred),
      'Training Time (seconds)':secs,'Best Iteration':rounds if rounds is not None else '',
      'Parameters':json.dumps(params,sort_keys=True)}

base_params={'n_estimators':250,'max_depth':6,'learning_rate':0.08,'min_child_weight':1,
 'subsample':0.85,'colsample_bytree':0.85,'reg_alpha':0.0,'reg_lambda':1.0}
# Compact, predefined search spanning tree size, shrinkage, depth, sampling, and regularization.
search=[
 {'n_estimators':500,'max_depth':3,'learning_rate':0.05,'min_child_weight':3,'subsample':0.85,'colsample_bytree':0.85,'reg_alpha':0.0,'reg_lambda':2.0},
 {'n_estimators':450,'max_depth':5,'learning_rate':0.05,'min_child_weight':3,'subsample':0.9,'colsample_bytree':0.9,'reg_alpha':0.05,'reg_lambda':2.0},
 {'n_estimators':350,'max_depth':7,'learning_rate':0.06,'min_child_weight':5,'subsample':0.8,'colsample_bytree':0.9,'reg_alpha':0.1,'reg_lambda':5.0},
 {'n_estimators':600,'max_depth':4,'learning_rate':0.03,'min_child_weight':1,'subsample':0.95,'colsample_bytree':0.8,'reg_alpha':0.0,'reg_lambda':5.0},
]
rows=[]; fitted={}
for include in (True,False):
    cfg='A: with accident count' if include else 'B: without accident count'
    cols=[c for c in X.columns if include or c!='num_reported_accidents']
    a,b=Xtr[cols],Xva[cols]
    prep=preprocessor(a); ta=time.perf_counter(); at=prep.fit_transform(a); bv=prep.transform(b); prep_secs=time.perf_counter()-ta
    # Baseline reproduces existing estimator with the same seed and preprocessing/split.
    base=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0,**base_params)
    start=time.perf_counter(); base.fit(at,ytr); elapsed=time.perf_counter()-start
    pred=base.predict(bv); row=score(pred,elapsed,'Baseline',cfg,base_params)
    rows.append(row); fitted[(cfg,'Baseline')]=(prep,base,cols,row)
    print(f"{cfg} Baseline RMSE={row['RMSE']:.5f} MAE={row['MAE']:.5f} time={elapsed:.1f}s",flush=True)
    for ix,p in enumerate(search,1):
        model=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0,
                           early_stopping_rounds=30,eval_metric='rmse',**p)
        start=time.perf_counter(); model.fit(at,ytr,eval_set=[(bv,yva)],verbose=False); elapsed=time.perf_counter()-start
        pred=model.predict(bv); bestit=int(model.best_iteration)
        row=score(pred,elapsed,f'Tuned {ix}',cfg,p,bestit); rows.append(row)
        fitted[(cfg,f'Tuned {ix}')]=(prep,model,cols,row)
        print(f"{cfg} Tuned {ix} RMSE={row['RMSE']:.5f} MAE={row['MAE']:.5f} R2={row['R²']:.5f} iter={bestit} time={elapsed:.1f}s",flush=True)

# Compare best per feature configuration, ranked by RMSE; ties prefer lower MAE then faster time.
baseA=next(r for r in rows if r['Run']=='Baseline' and r['Configuration'].startswith('A:'))
all_candidates=[(k,v) for k,v in fitted.items() if k[1]!='Baseline']
bestA=min([v for k,v in all_candidates if k[0].startswith('A:')],key=lambda v:(v[3]['RMSE'],v[3]['MAE'],v[3]['Training Time (seconds)']))
bestB=min([v for k,v in all_candidates if k[0].startswith('B:')],key=lambda v:(v[3]['RMSE'],v[3]['MAE'],v[3]['Training Time (seconds)']))
# Prefer a material RMSE gain; use a 1% threshold as practical noise guard, then speed.
base=baseA
best_pair=min([(bestA,bestB)],key=lambda pair:(pair[0][3]['RMSE'],pair[0][3]['MAE'],pair[0][3]['Training Time (seconds)']))
chosen=best_pair[0]
if bestB[3]['RMSE'] < bestA[3]['RMSE']:
    chosen=bestB
# If tuned gain is under 1%, retain baseline if it's faster or effectively tied.
if (base['RMSE']-chosen[3]['RMSE'])/base['RMSE'] < .01 and base['Training Time (seconds)'] <= chosen[3]['Training Time (seconds)']:
    chosen=fitted[('A: with accident count','Baseline')]
selected_prep,selected_model,selected_cols,optimized=chosen
optimized_cfg='A: with accident count' if 'num_reported_accidents' in selected_cols else 'B: without accident count'

# Workbook: all baseline/tuned rows and a compact comparison sheet.
out=ROOT/'optimization_results.xlsx'
results=pd.DataFrame(rows)
summary=pd.DataFrame([
 {'Model':'Baseline (A)','Configuration':'A: with accident count','MAE':baseA['MAE'],'RMSE':baseA['RMSE'],'R²':baseA['R²'],'Training Time (seconds)':baseA['Training Time (seconds)'],'Parameters':json.dumps(base_params,sort_keys=True)},
 {'Model':'Best tuned A','Configuration':'A: with accident count','MAE':bestA[3]['MAE'],'RMSE':bestA[3]['RMSE'],'R²':bestA[3]['R²'],'Training Time (seconds)':bestA[3]['Training Time (seconds)'],'Parameters':bestA[3]['Parameters']},
 {'Model':'Best tuned B','Configuration':'B: without accident count','MAE':bestB[3]['MAE'],'RMSE':bestB[3]['RMSE'],'R²':bestB[3]['R²'],'Training Time (seconds)':bestB[3]['Training Time (seconds)'],'Parameters':bestB[3]['Parameters']},
 {'Model':'Selected candidate','Configuration':optimized_cfg,'MAE':optimized['MAE'],'RMSE':optimized['RMSE'],'R²':optimized['R²'],'Training Time (seconds)':optimized['Training Time (seconds)'],'Parameters':optimized['Parameters']},
])
base_rows=pd.DataFrame([baseA, next(r for r in rows if r['Run']=='Baseline' and r['Configuration'].startswith('B:'))])
with pd.ExcelWriter(out,engine='openpyxl') as wr:
    results.to_excel(wr,index=False,sheet_name='Search Results')
    summary.to_excel(wr,index=False,sheet_name='Best Configurations')
    base_rows.to_excel(wr,index=False,sheet_name='Baselines')
wb=load_workbook(out)
for ws in wb.worksheets:
    ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='1F4E78'); c.alignment=Alignment(wrap_text=True)
    for col in ws.columns:
        letter=col[0].column_letter
        ws.column_dimensions[letter].width=min(70,max(14,max(len(str(x.value or '')) for x in col)+2))
for row in range(2,wb['Search Results'].max_row+1):
    if wb['Search Results'].cell(row,2).value.startswith('A:'):
        for c in wb['Search Results'][row]: c.fill=PatternFill('solid',fgColor='EAF2F8')
for row in range(2,wb['Best Configurations'].max_row+1):
    if wb['Best Configurations'].cell(row,1).value=='Selected candidate':
        for c in wb['Best Configurations'][row]: c.fill=PatternFill('solid',fgColor='C6EFCE'); c.font=Font(bold=True,color='006100')
wb.save(out)

# Refit selected hyperparameters and preprocessing on all labeled rows.
allcols=list(selected_cols); finalprep=preprocessor(X[allcols]); xfull=finalprep.fit_transform(X[allcols])
final_params=base_params if optimized['Run']=='Baseline' else json.loads(optimized['Parameters'])
final=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0,**final_params)
# Use the selected early-stopping iteration as the final tree count; no validation set is used in refit.
if optimized['Run']!='Baseline': final_params['n_estimators']=int(optimized['Best Iteration'])+1
final=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=SEED,verbosity=0,**final_params)
final.fit(xfull,y)
joblib.dump({'preprocessor':finalprep,'model':final,'algorithm':'XGBoost Regressor','configuration':optimized_cfg,
 'feature_columns':allcols,'target':TARGET,'random_seed':SEED,'parameters':final_params,
 'validation_metrics':optimized,'validation_selection':'lowest RMSE; MAE/R2 reported; <1% gain defers to faster baseline'},ROOT/'best_model_optimized.pkl',compress=3)
# Add machine-readable summary alongside requested outputs.
changes={'MAE':{'absolute':optimized['MAE']-baseA['MAE'],'percent':(optimized['MAE']-baseA['MAE'])/baseA['MAE']*100},
 'RMSE':{'absolute':optimized['RMSE']-baseA['RMSE'],'percent':(optimized['RMSE']-baseA['RMSE'])/baseA['RMSE']*100},
 'R2':{'absolute':optimized['R²']-baseA['R²'],'percent':(optimized['R²']-baseA['R²'])/abs(baseA['R²'])*100},
 'training_time':{'absolute':optimized['Training Time (seconds)']-baseA['Training Time (seconds)'],'percent':(optimized['Training Time (seconds)']-baseA['Training Time (seconds)'])/baseA['Training Time (seconds)']*100}}
report={'baseline_A':baseA,'optimized':optimized,'configuration':optimized_cfg,'parameters':final_params,'changes_vs_baseline_A':changes,
 'best_tuned_A':bestA[3],'best_tuned_B':bestB[3],'test_evaluation':'Unavailable; test.csv has no target labels.'}
(ROOT/'optimization_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('\nSELECTED',optimized_cfg,optimized['Run'],optimized['Parameters'])
print(f"BASELINE A RMSE={baseA['RMSE']:.5f} MAE={baseA['MAE']:.5f} R2={baseA['R²']:.5f} time={baseA['Training Time (seconds)']:.2f}s")
print(f"OPTIMIZED RMSE={optimized['RMSE']:.5f} MAE={optimized['MAE']:.5f} R2={optimized['R²']:.5f} time={optimized['Training Time (seconds)']:.2f}s")
print('RMSE absolute change',changes['RMSE']['absolute'],'percent',changes['RMSE']['percent'])
print('Outputs:',out.name,'best_model_optimized.pkl','optimization_summary.json')
