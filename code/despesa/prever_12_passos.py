"""
Previsão de 12 passos à frente de cada série de despesa, com o melhor modelo por
série entre os avaliados (individuais, locais e globais, incluindo o Chronos). O
vencedor é lido da comparacao_unificada (menor RMSE de teste) e só a previsão
futura dele é gerada.

Saídas:
  results/previsoes_12_passos_despesa.csv / .xlsx (tabela)
  results/forecasts_despesa/forecast_<cod>_<mod>.png + _grade_completa.png
"""
from __future__ import annotations
import sys, ast, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (CAMINHO_PARQUET, HORIZONTE, MESES_PREV, N_LAGS_GLOBAL,
                    RESULTS_DIR, RANDOM_STATE, GRID_SVR)
from utils import (carregar_dados, construir_series, mapear_nomes,
                   filtrar_series, preparar_serie, reverter_shift, dividir_serie)
from utils.acf_analysis import analisar_serie
from utils.windowing import criar_janela
from utils.metrics import calcular_metricas
from models.arima_model import treinar_avaliar_arima, prever_arima
from models.lr_model    import treinar_avaliar_lr,    prever_lr
from models.svr_model   import treinar_avaliar_svr,   prever_svr
from models.mlp_model   import treinar_avaliar_mlp,   prever_mlp
from models.lgbm_global import (construir_dataset_global, treinar_lgbm_global,
                                prever_rolling_lgbm, construir_features_serie, _col_names)
from models.global_sklearn import (treinar_global_ridge, treinar_global_rf,
                                   treinar_global_mlp, prever_rolling_global_sklearn)
from models.chronos_global import carregar_pipeline, prever_rolling_chronos
from clustering.pooling import (construir_dataset_pooled, prever_rolling_pooled,
                                normalizar, desnormalizar)
from statsmodels.tsa.arima.model import ARIMA

OUT = RESULTS_DIR / "forecasts_despesa"; OUT.mkdir(parents=True, exist_ok=True)
fut_pidx  = pd.PeriodIndex([pd.Period(m, freq="M") for m in MESES_PREV])
fut_dates = fut_pidx.to_timestamp()

# melhor modelo por serie (da unificada)
unif = sorted(glob.glob(str(RESULTS_DIR / "comparacao_unificada_*.csv")))[-1]
udf  = pd.read_csv(unif); udf["Codigo"] = udf["Codigo"].astype(str)
local_csv = sorted(glob.glob(str(RESULTS_DIR / "comparacao_local_*.csv")))[-1]
ldf = pd.read_csv(local_csv); ldf["Codigo"] = ldf["Codigo"].astype(str)
cluster_de = dict(zip(ldf["Codigo"], ldf["Cluster_ID"]))

RMSE_COL = {"Ind":"Ind_RMSE","Local_Ridge":"Pooled_Ridge_RMSE","Local_SVR":"Pooled_SVR_RMSE",
 "Local_MLP":"Pooled_MLP_RMSE","Glob_Ridge":"Global_Ridge_RMSE","Glob_RF":"Global_RF_RMSE",
 "Glob_MLP":"Global_MLP_RMSE","Glob_LGBM":"Global_LGBM_RMSE","Glob_Chronos":"Global_Chronos_RMSE"}
RMSE_COL = {k:v for k,v in RMSE_COL.items() if v in udf.columns}
def melhor_modelo(row):
    cand = {k: row[v] for k,v in RMSE_COL.items() if pd.notna(row[v])}
    k = min(cand, key=cand.get)
    return ("Ind:"+str(row["Ind_Melhor_Modelo"])) if k=="Ind" else k

# prepara series e treina modelos globais uma vez
df = carregar_dados(CAMINHO_PARQUET)
nomes = mapear_nomes(df)
series, _ = filtrar_series(construir_series(df))
codes = set(udf["Codigo"])
prep, series_data = {}, {}
for cod, s in series.items():
    if str(cod) not in codes: continue
    sp, grupo, shift = preparar_serie(s)
    tv, tsv, tidx, tesidx, n_hold, _ = dividir_serie(sp)
    prep[str(cod)] = dict(sp=sp, shift=shift, tv=tv, tsv=tsv, tidx=tidx, tesidx=tesidx, n_hold=n_hold)
    series_data[str(cod)] = (tv, tidx)

print("Treinando modelos globais (LightGBM, Ridge, RF, MLP, Chronos)...")
Xg, yg, norm_params, label_enc = construir_dataset_global(series_data, N_LAGS_GLOBAL)
mod_lgbm, _  = treinar_lgbm_global(Xg, yg)
mod_ridge, _ = treinar_global_ridge(Xg, yg)
mod_rf, _    = treinar_global_rf(Xg, yg)
mod_mlp, _   = treinar_global_mlp(Xg, yg)
chronos = carregar_pipeline()
print("Globais treinados.\n")
GLOB_SK = {"Glob_Ridge":(mod_ridge,False), "Glob_RF":(mod_rf,True), "Glob_MLP":(mod_mlp,False)}

def _nlags(p):
    acf = analisar_serie(p["tv"])
    return acf["n_lags"] if not acf["e_ruido_branco"] else p["n_hold"]

def _df(Xs, nl):
    d = pd.DataFrame(Xs, columns=_col_names(nl)); d["series_id"] = d["series_id"].astype(int); return d

# handlers: retornam (fit_train, test_pred, future) na escala serie_pronta
def h_individual(nome, p):
    tv,tsv,sp = p["tv"],p["tsv"],p["sp"]; nl=_nlags(p); todos=sp.values.astype(float)
    fit=np.full(len(tv),np.nan)
    if nome=="ARIMA":
        r=treinar_avaliar_arima(tv,tsv); m=ARIMA(tv,order=(r["p"],r["d"],r["q"])).fit()
        fv=np.asarray(m.fittedvalues,float); fit[:len(fv)]=fv; fit[:max(r["d"],1)]=np.nan
        return fit, np.asarray(r["preds_teste"],float), np.asarray(prever_arima(todos,r["p"],r["d"],r["q"],HORIZONTE),float)
    if nome=="LR":  r=treinar_avaliar_lr(tv,tsv,nl,alpha=None); fut=prever_lr(todos,nl,r["alpha"],HORIZONTE)
    elif nome=="SVR": r=treinar_avaliar_svr(tv,tsv,nl); fut=prever_svr(todos,nl,ast.literal_eval(r["params"]),HORIZONTE)
    else: r=treinar_avaliar_mlp(tv,tsv,nl); fut=prever_mlp(todos,nl,ast.literal_eval(r["params"]),HORIZONTE)
    X,_=criar_janela(tv,nl)
    if len(X): fit[nl:]=r["scaler_y"].inverse_transform(r["modelo"].predict(r["scaler_X"].transform(X)).reshape(-1,1)).ravel()
    return fit, np.asarray(r["preds_teste"],float), np.asarray(fut,float)

def _global_fit(cod, p, nl, predict_norm):
    mean,std=norm_params[cod]; sid=int(label_enc.transform([cod])[0])
    fit=np.full(len(p["tv"]),np.nan)
    Xs,_=construir_features_serie(normalizar(p["tv"],mean,std),p["tidx"],nl,sid)
    if len(Xs):
        yv=desnormalizar(np.asarray(predict_norm(Xs),float),mean,std)
        fit[nl:nl+len(yv)]=yv
    return fit,mean,std,sid

def h_glob_lgbm(cod,p):
    nl=N_LAGS_GLOBAL
    fit,mean,std,sid=_global_fit(cod,p,nl, lambda Xs: mod_lgbm.predict(_df(Xs,nl)))
    tp=np.asarray(prever_rolling_lgbm(mod_lgbm,p["tv"],p["tsv"],p["tidx"],p["tesidx"],nl,mean,std,sid),float)
    todos=p["sp"].values.astype(float); tidx=p["tidx"].append(p["tesidx"])
    fut=np.asarray(prever_rolling_lgbm(mod_lgbm,todos,np.zeros(HORIZONTE),tidx,fut_pidx,nl,mean,std,sid),float)
    return fit,tp,fut

def h_glob_sklearn(nome,cod,p):
    modelo,usa_array=GLOB_SK[nome]; nl=N_LAGS_GLOBAL; cols=_col_names(nl)
    fit,mean,std,sid=_global_fit(cod,p,nl, lambda Xs: modelo.predict(Xs if usa_array else _df(Xs,nl)))
    tp=np.asarray(prever_rolling_global_sklearn(modelo,p["tv"],p["tsv"],p["tesidx"],nl,mean,std,sid,cols,usa_array),float)
    todos=p["sp"].values.astype(float)
    fut=np.asarray(prever_rolling_global_sklearn(modelo,todos,np.zeros(HORIZONTE),fut_pidx,nl,mean,std,sid,cols,usa_array),float)
    return fit,tp,fut

def h_chronos(cod,p):
    nl=N_LAGS_GLOBAL
    tp=np.asarray(prever_rolling_chronos(chronos,p["tv"],p["tsv"],context_length=nl),float)
    todos=p["sp"].values.astype(float)
    fut=np.asarray(prever_rolling_chronos(chronos,todos,np.zeros(HORIZONTE),context_length=nl),float)
    return np.full(len(p["tv"]),np.nan),tp,fut

def h_local(nome,cod,p):
    nl=_nlags(p); cid=cluster_de.get(cod)
    membros=[c for c in cluster_de if cluster_de[c]==cid and c in series_data]
    treinos={c:series_data[c][0] for c in membros}
    Xp,yp,npar,scaler=construir_dataset_pooled(membros,treinos,nl)
    if nome=="Local_Ridge": modelo=Ridge(alpha=1.0).fit(Xp,yp)
    elif nome=="Local_SVR":
        ns=min(3,max(2,len(Xp)//30))
        gs=GridSearchCV(SVR(),GRID_SVR,cv=TimeSeriesSplit(n_splits=ns),scoring="neg_root_mean_squared_error")
        gs.fit(Xp,yp); modelo=gs.best_estimator_
    else: modelo=MLPRegressor(hidden_layer_sizes=(50,),max_iter=1000,random_state=RANDOM_STATE).fit(Xp,yp)
    mean,std=npar[cod]
    tp=np.asarray(prever_rolling_pooled(modelo,scaler,p["tv"],p["tsv"],nl,mean,std),float)
    todos=p["sp"].values.astype(float)
    fut=np.asarray(prever_rolling_pooled(modelo,scaler,todos,np.zeros(HORIZONTE),nl,mean,std),float)
    return np.full(len(p["tv"]),np.nan),tp,fut

def fmt(x,_):
    a=abs(x)
    return f"{x/1e6:.1f}M" if a>=1e6 else (f"{x/1e3:.0f}k" if a>=1e3 else f"{x:.0f}")

LBL={"Glob_LGBM":"LightGBM global","Glob_RF":"RF global","Glob_MLP":"MLP global",
     "Glob_Ridge":"Ridge global","Glob_Chronos":"Chronos","Local_Ridge":"Ridge local",
     "Local_SVR":"SVR local","Local_MLP":"MLP local"}

linhas=[]; paineis=[]
for _, row in udf.sort_values("Codigo").iterrows():
    cod=row["Codigo"]; p=prep.get(cod)
    if p is None: continue
    mod=melhor_modelo(row); nome=nomes.get(cod,cod)
    lbl=LBL.get(mod, mod.replace("Ind:","")+" (ind.)")
    try:
        if mod.startswith("Ind:"): fit,tp,fut=h_individual(mod.split(":")[1],p)
        elif mod=="Glob_LGBM":     fit,tp,fut=h_glob_lgbm(cod,p)
        elif mod in GLOB_SK:       fit,tp,fut=h_glob_sklearn(mod,cod,p)
        elif mod=="Glob_Chronos":  fit,tp,fut=h_chronos(cod,p)
        elif mod.startswith("Local_"): fit,tp,fut=h_local(mod,cod,p)
        else: raise ValueError(mod)
        sh=p["shift"]
        obs=reverter_shift(p["sp"].values.astype(float),sh)
        fit_o=reverter_shift(fit,sh); tp_o=reverter_shift(tp,sh); fut_o=reverter_shift(fut,sh)
        mape=calcular_metricas(p["tsv"],tp)["MAPE"]
        td=p["tidx"].to_timestamp(); ed=p["tesidx"].to_timestamp(); all_d=td.append(ed)
        linha={"Codigo":cod,"Nome":str(nome)[:70],"Modelo":lbl,
               "MAPE_holdout": ">999" if mape>=1000 else round(mape,1)}
        for m,v in zip(MESES_PREV,fut_o): linha[m]=round(float(v),2)
        linhas.append(linha)
        fig,ax=plt.subplots(figsize=(8.2,3.0))
        ax.plot(all_d,obs,color="black",lw=1.4,label="Observado",zorder=3)
        ax.plot(td,fit_o,color="#1f77b4",lw=1.0,ls="--",label="Ajuste (treino)")
        ax.plot(ed,tp_o,color="#ff7f0e",lw=1.8,marker="o",ms=3,label="Previsão (teste)")
        ax.plot(fut_dates,fut_o,color="#d62728",lw=1.8,marker="s",ms=3,label="Previsão (12 passos)")
        ax.axvline(ed[0],color="gray",ls=":",alpha=0.7); ax.axvline(fut_dates[0],color="gray",ls=":",alpha=0.7)
        ax.set_title(f"{cod} — {str(nome)[:54]}  [{lbl}]",fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt)); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6,ncol=2,loc="best"); ax.grid(alpha=0.25,ls="--"); fig.tight_layout()
        tag=lbl.replace(' ','').replace('.','').replace('(','').replace(')','')
        fig.savefig(OUT/f"forecast_{cod}_{tag}.png",dpi=140,bbox_inches="tight"); plt.close(fig)
        paineis.append((cod,lbl,all_d,obs,td,fit_o,ed,tp_o,fut_o))
        print(f"[OK] {cod}  {lbl}")
    except Exception as e:
        print(f"[FALHA] {cod} {mod}: {e}")

n=len(paineis); nc=3; nr=int(np.ceil(n/nc))
fig,axes=plt.subplots(nr,nc,figsize=(nc*5,nr*2.6)); axes=np.atleast_1d(axes).ravel()
for ax,(cod,lbl,all_d,obs,td,fit_o,ed,tp_o,fut_o) in zip(axes,paineis):
    ax.plot(all_d,obs,color="black",lw=1.1); ax.plot(td,fit_o,color="#1f77b4",lw=0.9,ls="--")
    ax.plot(ed,tp_o,color="#ff7f0e",lw=1.4); ax.plot(fut_dates,fut_o,color="#d62728",lw=1.4)
    ax.axvline(ed[0],color="gray",ls=":",alpha=0.6); ax.axvline(fut_dates[0],color="gray",ls=":",alpha=0.6)
    ax.set_title(f"{cod} [{lbl}]",fontsize=8); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.tick_params(labelsize=6); ax.grid(alpha=0.25,ls="--")
for ax in axes[n:]: ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/"_grade_completa.png",dpi=130,bbox_inches="tight"); plt.close(fig)

out=pd.DataFrame(linhas)
out.to_csv(RESULTS_DIR/"previsoes_12_passos_despesa.csv",index=False,encoding="utf-8-sig")
with pd.ExcelWriter(RESULTS_DIR/"previsoes_12_passos_despesa.xlsx",engine="openpyxl") as w:
    out.to_excel(w,index=False,sheet_name="Previsoes_12m")
print(f"\n{len(out)} series. Distribuicao de modelos:")
print(out["Modelo"].value_counts().to_string())
