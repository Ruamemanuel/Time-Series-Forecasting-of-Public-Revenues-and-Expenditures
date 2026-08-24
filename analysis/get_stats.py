import pandas as pd, numpy as np, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from testes_estatisticos import calcular_normas, RECEITA, DESPESA

normas_r = calcular_normas(RECEITA['parquet'], RECEITA['col_valor'], RECEITA['unificada'])
normas_d = calcular_normas(DESPESA['parquet'], DESPESA['col_valor'], DESPESA['unificada'])

def get_n(cod, normas):
    return normas.get(cod) or normas.get(str(cod)) or (normas.get(int(str(cod))) if str(cod).isdigit() else None)

# ── Individuais via metricas ──────────────────────────────────────────────────
def ind_stats(met_path, normas, n_series):
    df = pd.read_csv(met_path)
    res = {}
    for modelo, grp in df.groupby('Modelo'):
        nrmses, mapes, wins = [], [], 0
        for _, row in grp.iterrows():
            n = get_n(row['Codigo'], normas)
            if n and n > 0 and not pd.isna(row['RMSE']):
                nrmses.append(row['RMSE'] / n)
            if 'MAPE' in row and not pd.isna(row['MAPE']):
                mapes.append(row['MAPE'])
            if row.get('Selecionado', False):
                wins += 1
        if nrmses:
            res[modelo] = dict(nrmse=round(np.median(nrmses),4),
                               mape=round(np.median(mapes),1) if mapes else None,
                               n=len(nrmses), wins=wins, n_series=n_series)
    return res

# ── Local/Global via unificada ────────────────────────────────────────────────
def multi_stats(uni_path, normas, n_series):
    uni = pd.read_csv(uni_path)
    cols = {
        'Local Ridge':  ('Pooled_Ridge_RMSE',   'Pooled_Ridge_MAPE'),
        'Local SVR':    ('Pooled_SVR_RMSE',      'Pooled_SVR_MAPE'),
        'Local MLP':    ('Pooled_MLP_RMSE',      'Pooled_MLP_MAPE'),
        'Local LGBM':   ('LGBM_Cluster_RMSE',    'LGBM_Cluster_MAPE'),
        'Local Melhor': ('Pooled_Melhor_RMSE',   None),
        'Glob Ridge':   ('Global_Ridge_RMSE',    'Global_Ridge_MAPE'),
        'Glob RF':      ('Global_RF_RMSE',       'Global_RF_MAPE'),
        'Glob MLP':     ('Global_MLP_RMSE',      'Global_MLP_MAPE'),
        'Glob LGBM':    ('Global_LGBM_RMSE',     'Global_LGBM_MAPE'),
        'Glob Melhor':  ('Global_Melhor_RMSE',   None),
        'Ind Melhor':   ('Ind_RMSE',             'Ind_MAPE'),
    }
    res = {}
    for label, (rcol, mcol) in cols.items():
        if rcol not in uni.columns:
            continue
        nrmses, mapes = [], []
        for _, row in uni.iterrows():
            n = get_n(row['Codigo'], normas)
            if n and n > 0 and not pd.isna(row[rcol]):
                nrmses.append(row[rcol] / n)
            if mcol and mcol in uni.columns and not pd.isna(row.get(mcol, float('nan'))):
                mapes.append(row[mcol])
        if nrmses:
            res[label] = dict(nrmse=round(np.median(nrmses),4),
                              mape=round(np.median(mapes),1) if mapes else None,
                              n=len(nrmses))
    # Vitórias intra
    for label, (rcol, _) in cols.items():
        if label in ('Local Melhor','Glob Melhor','Ind Melhor') or rcol not in uni.columns:
            continue
        paradigm = label.split()[0]
        paradigm_cols = [v[0] for k,v in cols.items() if k.startswith(paradigm) and 'Melhor' not in k and v[0] in uni.columns]
        mat = uni[paradigm_cols].values.astype(float)
        best = np.nanargmin(mat, axis=1)
        idx = paradigm_cols.index(rcol)
        wins = int((best == idx).sum())
        if label in res:
            res[label]['wins'] = wins
            res[label]['n_series'] = n_series
    # Vitórias inter (Melhor paradigma)
    inter_cols = {'Ind': 'Ind_RMSE', 'Local': 'Pooled_Melhor_RMSE', 'Glob': 'Global_Melhor_RMSE'}
    avail = {k: v for k,v in inter_cols.items() if v in uni.columns}
    if avail:
        mat = uni[[v for v in avail.values()]].values.astype(float)
        best = np.nanargmin(mat, axis=1)
        for i, (paradigm, _) in enumerate(avail.items()):
            key = paradigm + ' Melhor'
            if key in res:
                res[key]['wins'] = int((best == i).sum())
                res[key]['n_series'] = n_series
    return res

print("=== RECEITA ===")
r_ind = ind_stats(RECEITA['metricas'], normas_r, 133)
r_multi = multi_stats(RECEITA['unificada'], normas_r, 133)
for m, v in sorted(r_ind.items()):
    print(f"IND  {m:10s}  nRMSE={v['nrmse']:.4f}  MAPE={v['mape']}%  n={v['n']}  wins={v['wins']}/{v['n_series']}")
for m, v in r_multi.items():
    wins_str = f"wins={v.get('wins','?')}/{v.get('n_series','?')}" if 'wins' in v else ''
    print(f"MULT {m:15s}  nRMSE={v['nrmse']:.4f}  MAPE={v['mape']}%  n={v['n']}  {wins_str}")

print()
print("=== DESPESA ===")
d_ind = ind_stats(DESPESA['metricas'], normas_d, 19)
d_multi = multi_stats(DESPESA['unificada'], normas_d, 19)
for m, v in sorted(d_ind.items()):
    print(f"IND  {m:10s}  nRMSE={v['nrmse']:.4f}  MAPE={v['mape']}%  n={v['n']}  wins={v['wins']}/{v['n_series']}")
for m, v in d_multi.items():
    wins_str = f"wins={v.get('wins','?')}/{v.get('n_series','?')}" if 'wins' in v else ''
    print(f"MULT {m:15s}  nRMSE={v['nrmse']:.4f}  MAPE={v['mape']}%  n={v['n']}  {wins_str}")
