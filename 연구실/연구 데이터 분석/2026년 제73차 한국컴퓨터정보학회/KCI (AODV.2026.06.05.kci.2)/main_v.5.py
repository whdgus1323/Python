#!/usr/bin/env python
# coding: utf-8

# # main_v.4 random_base1 offline-best-threshold learning
# 
# This notebook does **not** try to predict the raw random threshold that was sampled online.
# 
# Instead, it builds an offline supervision label from the completed `random_base1` logs:
# 
# 1. keep only online-available input state features
# 2. evaluate threshold candidates using offline future metrics and run-level PDR
# 3. choose the better threshold pair for each state bin
# 4. train a model to predict that offline best threshold pair
# 
# Runs used while simulation is still in progress: `4, 10, 14, 15, 16`.

# In[1]:


from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

experiment_name = 'main_v_5_random_base1_offline_best_state_split'
random_base_root = Path(r'C:\Users\Choe JongHyeon\Desktop\map_v2\random_base1')
run_ids = [4, 10, 14, 15, 16]
seed_name = 'seed_1'

random_state = 42
chunk_size = 250_000
max_rows_per_run = 60_000

neighbor_norm = 20.0
hop_norm = 10.0
future_window_sec = 2
state_cbr_bin_size = 5.0
state_neighbor_bin_size = 5.0
hop_bin_max = 6
min_pair_samples_per_state = 10

selected_features = [
    'localCbrNorm',
    'neighborNorm',
    'hopNorm',
    'isDirectRoute',
]

display(pd.DataFrame({
    'experiment_name': [experiment_name],
    'random_base_root': [str(random_base_root)],
    'run_ids': [run_ids],
    'max_rows_per_run': [max_rows_per_run],
    'future_window_sec': [future_window_sec],
    'state_cbr_bin_size': [state_cbr_bin_size],
    'state_neighbor_bin_size': [state_neighbor_bin_size],
    'selected_features': [selected_features],
}))


# In[2]:


pdr_pattern = re.compile(r'PDR\s*\(%\):\s*([0-9.]+)')


def parse_run_pdr(result_path: Path) -> float:
    if not result_path.exists():
        return np.nan
    text = result_path.read_text(encoding='utf-8', errors='ignore')
    match = pdr_pattern.search(text)
    return float(match.group(1)) if match else np.nan


def build_future_table_for_run(run_id: int) -> pd.DataFrame:
    base = random_base_root / str(run_id) / seed_name
    path = base / 'aodv_transmission_failure_diagnosis_1s.csv'
    dia = pd.read_csv(
        path,
        usecols=[
            'time', 'node', 'localCbr', 'neighborCount',
            'appliedLowThreshold', 'appliedHighThreshold',
            'routeDiscoveryStarted', 'routeDiscoverySucceeded',
            'routeDiscoveryFailed', 'routeDiscoveryDelayAvgMs',
        ],
    )

    dia['sec'] = pd.to_numeric(dia['time'], errors='coerce').astype('Int64')
    for column in [
        'localCbr', 'neighborCount', 'appliedLowThreshold', 'appliedHighThreshold',
        'routeDiscoveryStarted', 'routeDiscoverySucceeded', 'routeDiscoveryFailed', 'routeDiscoveryDelayAvgMs',
    ]:
        dia[column] = pd.to_numeric(dia[column], errors='coerce').fillna(0.0)

    dia = dia.dropna(subset=['sec', 'node']).drop_duplicates(['sec', 'node']).copy()

    lookup = dia.set_index(['sec', 'node'])[
        ['routeDiscoveryStarted', 'routeDiscoverySucceeded', 'routeDiscoveryFailed', 'routeDiscoveryDelayAvgMs']
    ]

    future_rows = []
    for sec, node in dia[['sec', 'node']].itertuples(index=False):
        started_sum = 0.0
        succeeded_sum = 0.0
        failed_sum = 0.0
        delay_values = []
        for offset in range(future_window_sec + 1):
            try:
                values = lookup.loc[(int(sec) + offset, node)]
            except KeyError:
                continue
            started_sum += float(values['routeDiscoveryStarted'])
            succeeded_sum += float(values['routeDiscoverySucceeded'])
            failed_sum += float(values['routeDiscoveryFailed'])
            delay_value = float(values['routeDiscoveryDelayAvgMs'])
            if delay_value > 0:
                delay_values.append(delay_value)

        future_rows.append({
            'sec': int(sec),
            'node': node,
            'futureStarted': started_sum,
            'futureSucceeded': succeeded_sum,
            'futureFailed': failed_sum,
            'futureDelay': float(np.mean(delay_values)) if delay_values else 0.0,
        })

    future_df = pd.DataFrame(future_rows)
    base_state = dia[['sec', 'node', 'localCbr', 'neighborCount', 'appliedLowThreshold', 'appliedHighThreshold']].copy()
    return base_state.merge(future_df, on=['sec', 'node'], how='left')


def load_decision_sample_for_run(run_id: int) -> pd.DataFrame:
    base = random_base_root / str(run_id) / seed_name
    future_table = build_future_table_for_run(run_id)
    decision_path = base / 'aodv_cbr_rrep_decisions.csv'

    parts = []
    total = 0
    for chunk_index, chunk in enumerate(pd.read_csv(decision_path, usecols=['time', 'node', 'hopCount'], chunksize=chunk_size), start=1):
        chunk = chunk.dropna(subset=['time', 'node', 'hopCount']).copy()
        if chunk.empty:
            continue

        remaining = max_rows_per_run - total
        if remaining <= 0:
            break

        take = min(remaining, max(1, int(len(chunk) * 0.03)))
        if take < len(chunk):
            chunk = chunk.sample(n=take, random_state=random_state + run_id + chunk_index)

        chunk['sec'] = np.floor(pd.to_numeric(chunk['time'], errors='coerce')).astype('Int64')
        chunk['isDirectRoute'] = (pd.to_numeric(chunk['hopCount'], errors='coerce') <= 1).astype(float)
        parts.append(chunk)
        total += len(chunk)

    decision_sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if decision_sample.empty:
        return decision_sample

    decision_sample = decision_sample.merge(future_table, on=['sec', 'node'], how='left')
    decision_sample['run'] = run_id
    decision_sample['runPdr'] = parse_run_pdr(base / 'result.txt')
    return decision_sample


frames = []
run_summary = []
for run_id in run_ids:
    base = random_base_root / str(run_id) / seed_name
    frame = load_decision_sample_for_run(run_id)
    frames.append(frame)
    run_summary.append({
        'run': run_id,
        'sample_rows': len(frame),
        'run_pdr': parse_run_pdr(base / 'result.txt'),
        'decision_log_mb': (base / 'aodv_cbr_rrep_decisions.csv').stat().st_size / 1024 / 1024,
    })

dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
display(pd.DataFrame(run_summary))
display(pd.DataFrame({'dataset_rows_before_cleaning': [len(dataset)]}))


# In[3]:


if dataset.empty:
    raise ValueError('dataset is empty. Check random_base_root and completed runs.')

for column in [
    'localCbr', 'neighborCount', 'hopCount', 'isDirectRoute',
    'appliedLowThreshold', 'appliedHighThreshold',
    'futureStarted', 'futureSucceeded', 'futureFailed', 'futureDelay', 'runPdr',
]:
    dataset[column] = pd.to_numeric(dataset[column], errors='coerce')

dataset = dataset.dropna(subset=['localCbr', 'neighborCount', 'hopCount', 'isDirectRoute', 'appliedLowThreshold', 'appliedHighThreshold']).copy()

dataset['stateCbrBin'] = np.clip(np.floor(dataset['localCbr'] / state_cbr_bin_size) * state_cbr_bin_size, 0, 100)
dataset['stateNeighborBin'] = np.clip(np.floor(dataset['neighborCount'] / state_neighbor_bin_size) * state_neighbor_bin_size, 0, 100)
dataset['stateHopBin'] = np.clip(dataset['hopCount'].round().astype(int), 0, hop_bin_max)
dataset['stateDirectBin'] = dataset['isDirectRoute'].round().astype(int)
dataset['stateKey'] = (
    dataset['stateCbrBin'].astype(int).astype(str)
    + '|'
    + dataset['stateNeighborBin'].astype(int).astype(str)
    + '|'
    + dataset['stateHopBin'].astype(int).astype(str)
    + '|'
    + dataset['stateDirectBin'].astype(int).astype(str)
)

dataset['lowInt'] = dataset['appliedLowThreshold'].round().astype(int)
dataset['highInt'] = dataset['appliedHighThreshold'].round().astype(int)

# Offline-only supervision score. These values are not used as online inputs.
dataset['offlineScore'] = (
    dataset['runPdr'].fillna(0)
    + 2.0 * dataset['futureSucceeded'].fillna(0)
    - 3.0 * dataset['futureFailed'].fillna(0)
    - 0.02 * dataset['futureDelay'].fillna(0)
)

pair_score = (
    dataset
    .groupby(['stateKey', 'lowInt', 'highInt'], observed=False)
    .agg(
        scoreMean=('offlineScore', 'mean'),
        sampleCount=('offlineScore', 'size'),
    )
    .reset_index()
)
pair_score = pair_score[pair_score['sampleCount'] >= min_pair_samples_per_state].copy()
pair_score = pair_score.sort_values(['stateKey', 'scoreMean', 'sampleCount'], ascending=[True, False, False])

best_pair_by_state = (
    pair_score
    .groupby('stateKey', observed=False)
    .head(1)
    .rename(columns={
        'lowInt': 'bestLow',
        'highInt': 'bestHigh',
        'scoreMean': 'bestScore',
        'sampleCount': 'bestPairSampleCount',
    })
)

labeled_dataset = dataset.merge(
    best_pair_by_state[['stateKey', 'bestLow', 'bestHigh', 'bestScore', 'bestPairSampleCount']],
    on='stateKey',
    how='inner',
).copy()

display(pd.DataFrame({
    'dataset_rows_after_cleaning': [len(dataset)],
    'labeled_rows': [len(labeled_dataset)],
    'unique_states': [labeled_dataset['stateKey'].nunique()],
    'unique_best_pairs': [labeled_dataset[['bestLow', 'bestHigh']].drop_duplicates().shape[0]],
}))
display(best_pair_by_state.head(20))
display(labeled_dataset[['run', 'localCbr', 'neighborCount', 'hopCount', 'isDirectRoute', 'bestLow', 'bestHigh', 'bestScore']].describe(include='all'))


# In[4]:


X = pd.DataFrame({
    'localCbrNorm': np.clip(labeled_dataset['localCbr'].astype(float) / 100.0, 0, 1),
    'neighborNorm': np.clip(labeled_dataset['neighborCount'].astype(float) / neighbor_norm, 0, 1),
    'hopNorm': np.clip(labeled_dataset['hopCount'].astype(float) / hop_norm, 0, 1),
    'isDirectRoute': labeled_dataset['isDirectRoute'].astype(float),
})
Y = labeled_dataset[['bestLow', 'bestHigh']].astype(float)
groups = labeled_dataset['stateKey'].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
train_idx, test_idx = next(splitter.split(X, Y, groups=groups))

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train = Y.iloc[train_idx].reset_index(drop=True)
y_test = Y.iloc[test_idx].reset_index(drop=True)
train_groups = groups.iloc[train_idx].reset_index(drop=True)
test_groups = groups.iloc[test_idx].reset_index(drop=True)

train_state_set = set(train_groups.unique())
test_state_set = set(test_groups.unique())
state_overlap = train_state_set & test_state_set

display(pd.DataFrame({
    'train_rows': [len(X_train)],
    'test_rows': [len(X_test)],
    'train_states': [len(train_state_set)],
    'test_states': [len(test_state_set)],
    'overlap_states': [len(state_overlap)],
    'overlap_ratio_vs_test_states': [len(state_overlap) / max(len(test_state_set), 1)],
}))

if state_overlap:
    raise RuntimeError(f'stateKey leakage detected: {len(state_overlap)} overlapping states')

models = {
    'extra_trees': ExtraTreesRegressor(
        n_estimators=220,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    ),
    'random_forest': RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    ),
    'hist_gradient_boosting': MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.05,
            random_state=random_state,
        )
    ),
}

results = []
predictions = {}
for model_name, model in models.items():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predictions[model_name] = pred

    r2_each = r2_score(y_test, pred, multioutput='raw_values')
    r2_uniform = r2_score(y_test, pred, multioutput='uniform_average')
    mae_each = mean_absolute_error(y_test, pred, multioutput='raw_values')
    rmse_each = np.sqrt(mean_squared_error(y_test, pred, multioutput='raw_values'))

    results.append({
        'model': model_name,
        'r2_low': r2_each[0],
        'r2_high': r2_each[1],
        'r2_uniform': r2_uniform,
        'mae_low': mae_each[0],
        'mae_high': mae_each[1],
        'rmse_low': rmse_each[0],
        'rmse_high': rmse_each[1],
    })

result_table = pd.DataFrame(results).sort_values('r2_uniform', ascending=False).reset_index(drop=True)
best_model_name = result_table.loc[0, 'model']
display(result_table)
display(pd.DataFrame({
    'best_model': [best_model_name],
    'best_r2_uniform': [result_table.loc[0, 'r2_uniform']],
}))


# In[5]:


best_pred = predictions[best_model_name]
preview = X_test.copy().reset_index(drop=True)
actual = y_test.reset_index(drop=True)

preview['actualLow'] = actual['bestLow']
preview['actualHigh'] = actual['bestHigh']
preview['predLow'] = best_pred[:, 0]
preview['predHigh'] = best_pred[:, 1]
preview['lowError'] = (preview['actualLow'] - preview['predLow']).abs()
preview['highError'] = (preview['actualHigh'] - preview['predHigh']).abs()
preview['pairExactMatch'] = (
    preview['actualLow'].round().astype(int).astype(str)
    + '-'
    + preview['actualHigh'].round().astype(int).astype(str)
) == (
    preview['predLow'].round().astype(int).astype(str)
    + '-'
    + preview['predHigh'].round().astype(int).astype(str)
)

display(preview.head(20))
display(preview[['actualLow', 'actualHigh', 'predLow', 'predHigh', 'lowError', 'highError']].describe())
display(pd.DataFrame({
    'pair_exact_match_rate': [float(preview['pairExactMatch'].mean())],
    'mean_low_error': [float(preview['lowError'].mean())],
    'mean_high_error': [float(preview['highError'].mean())],
}))

