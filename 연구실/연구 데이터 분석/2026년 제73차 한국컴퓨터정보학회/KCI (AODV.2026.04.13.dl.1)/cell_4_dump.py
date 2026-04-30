from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if len(eligible_buckets) < 2:
    raise RuntimeError('Not enough eligible state buckets for pair classification.')

observed_best = (bucket_pair_perf
    .merge(eligible_buckets[['stateBucket']], on='stateBucket', how='inner')
    .sort_values(['stateBucket', 'label_rate', 'rows', 'lowThresholdRaw', 'highThresholdRaw'], ascending=[True, False, False, True, True])
    .drop_duplicates('stateBucket')
    .rename(columns={
        'lowThresholdRaw': 'observedBestLow',
        'highThresholdRaw': 'observedBestHigh',
        'label_rate': 'observedBestRate',
        'rows': 'observedBestRows',
    }))

candidate_pairs = candidate_pairs.copy()
candidate_pairs['pairKey'] = candidate_pairs['lowThresholdRaw'].astype(int).astype(str) + '-' + candidate_pairs['highThresholdRaw'].astype(int).astype(str)
candidate_pairs['pairClass'] = np.arange(len(candidate_pairs), dtype=int)
pair_class_lookup = candidate_pairs[['pairKey', 'pairClass']].copy()

pair_perf_lookup = bucket_pair_perf[['stateBucket', 'lowThresholdRaw', 'highThresholdRaw', 'label_rate', 'rows']].copy()
pair_perf_lookup['pairKey'] = pair_perf_lookup['lowThresholdRaw'].astype(int).astype(str) + '-' + pair_perf_lookup['highThresholdRaw'].astype(int).astype(str)
pair_perf_lookup = pair_perf_lookup.merge(pair_class_lookup, on='pairKey', how='left')

model_df = eligible_buckets.merge(observed_best[['stateBucket', 'observedBestLow', 'observedBestHigh', 'observedBestRate', 'observedBestRows']], on='stateBucket', how='inner').copy()
model_df['observedPairKey'] = model_df['observedBestLow'].astype(int).astype(str) + '-' + model_df['observedBestHigh'].astype(int).astype(str)
model_df = model_df.merge(pair_class_lookup, left_on='observedPairKey', right_on='pairKey', how='left')
model_df = model_df.rename(columns={'pairClass': 'targetClass'})

X = model_df[selected_state_features].copy()
y = model_df['targetClass'].astype(int)
groups = model_df['stateBucket'].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
train_df = model_df.iloc[train_idx].reset_index(drop=True)
heldout_df = model_df.iloc[test_idx].reset_index(drop=True)
X_train = train_df[selected_state_features].copy()
y_train = train_df['targetClass'].astype(int)
X_heldout = heldout_df[selected_state_features].copy()
y_heldout = heldout_df['targetClass'].astype(int)

class_weight_series = train_df['targetClass'].value_counts()
train_weight = train_df['targetClass'].map(lambda v: 1.0 / np.sqrt(max(class_weight_series.get(v, 1), 1))).astype(float)
train_weight *= 1.0 + (train_df['observedBestRate'].astype(float) - train_df['observedBestRate'].min())
train_weight = train_weight / train_weight.mean()

classifier_specs = {
    'HistGradientBoosting': HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, min_samples_leaf=10, random_state=random_state),
    'RandomForest': RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=random_state),
    'KNeighbors': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights='distance')),
    'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, batch_size=32, learning_rate_init=0.001, random_state=random_state, max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=60)),
}

pair_class_to_values = candidate_pairs.set_index('pairClass')[['lowThresholdRaw', 'highThresholdRaw']].to_dict('index')


def predict_proba_with_fallback(model, X_frame):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_frame)
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_frame)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    pred = np.asarray(model.predict(X_frame), dtype=int)
    probs = np.zeros((len(pred), len(candidate_pairs)), dtype=float)
    probs[np.arange(len(pred)), pred] = 1.0
    return probs


def evaluate_probs(eval_df, probs):
    probs = np.asarray(probs, dtype=float)
    class_order = np.argsort(-probs, axis=1)
    records = []
    for top_k in [1, 3, 5]:
        hits = []
        regrets = []
        for row_pos, row in enumerate(eval_df.itertuples(index=False)):
            top_classes = class_order[row_pos, :top_k]
            hits.append(int(int(row.targetClass) in set(top_classes.tolist())))
            top_pair_rows = pair_perf_lookup[(pair_perf_lookup['stateBucket'] == row.stateBucket) & (pair_perf_lookup['pairClass'].isin(top_classes))]
            best_top_rate = float(top_pair_rows['label_rate'].max()) if not top_pair_rows.empty else np.nan
            regrets.append(float(row.observedBestRate - best_top_rate) if pd.notna(best_top_rate) else np.nan)
        records.append({
            'top_k': top_k,
            'hit_rate': float(np.mean(hits)) if hits else np.nan,
            'mean_regret': float(np.nanmean(regrets)) if len(regrets) else np.nan,
            'max_regret': float(np.nanmax(regrets)) if len(regrets) else np.nan,
        })
    top1_class = class_order[:, 0]
    top1_low = [pair_class_to_values[int(c)]['lowThresholdRaw'] for c in top1_class]
    top1_high = [pair_class_to_values[int(c)]['highThresholdRaw'] for c in top1_class]
    top1_prob = probs[np.arange(len(eval_df)), top1_class]
    bucket_eval = eval_df[['stateBucket', 'localCbrNorm', 'neighborNorm', 'hopNorm', 'isDirectRoute', 'total_rows', 'observedBestLow', 'observedBestHigh', 'observedBestRate', 'targetClass']].copy()
    bucket_eval['predBestClass'] = top1_class
    bucket_eval['predBestLow'] = top1_low
    bucket_eval['predBestHigh'] = top1_high
    bucket_eval['predBestProb'] = top1_prob
    bucket_eval['pairMatch'] = ((bucket_eval['observedBestLow'] == bucket_eval['predBestLow']) & (bucket_eval['observedBestHigh'] == bucket_eval['predBestHigh'])).astype(int)
    bucket_eval = bucket_eval.merge(
        pair_perf_lookup[['stateBucket', 'pairClass', 'label_rate']].rename(columns={'pairClass': 'predBestClass', 'label_rate': 'predBestActualRate'}),
        on=['stateBucket', 'predBestClass'],
        how='left'
    )
    bucket_eval['regret'] = bucket_eval['observedBestRate'] - bucket_eval['predBestActualRate']
    return pd.DataFrame(records), bucket_eval, class_order


classifier_results = []
heldout_artifacts = {}
for model_name in classifier_names:
    model = classifier_specs[model_name]
    try:
        model.fit(X_train, y_train, sample_weight=train_weight)
    except (TypeError, ValueError):
        rng = np.random.default_rng(random_state)
        prob = np.asarray(train_weight, dtype=float)
        prob = prob / prob.sum()
        sampled = rng.choice(np.arange(len(X_train)), size=max(len(X_train) * 8, 2048), replace=True, p=prob)
        model.fit(X_train.iloc[sampled].reset_index(drop=True), y_train.iloc[sampled].reset_index(drop=True))
    heldout_probs = predict_proba_with_fallback(model, X_heldout)
    heldout_topk_eval_df, heldout_bucket_eval, heldout_class_order = evaluate_probs(heldout_df, heldout_probs)
    classifier_results.append({
        'model_name': model_name,
        'heldout_top1_hit_rate': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 1, 'hit_rate'].iloc[0]),
        'heldout_top3_hit_rate': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 3, 'hit_rate'].iloc[0]),
        'heldout_top5_hit_rate': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 5, 'hit_rate'].iloc[0]),
        'heldout_top1_mean_regret': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 1, 'mean_regret'].iloc[0]),
        'heldout_top3_mean_regret': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 3, 'mean_regret'].iloc[0]),
        'heldout_top5_mean_regret': float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 5, 'mean_regret'].iloc[0]),
    })
    heldout_artifacts[model_name] = {
        'model': model,
        'topk': heldout_topk_eval_df,
        'bucket_eval': heldout_bucket_eval,
    }

classifier_results_df = pd.DataFrame(classifier_results).sort_values([
    'heldout_top1_hit_rate',
    'heldout_top3_hit_rate',
    'heldout_top1_mean_regret',
], ascending=[False, False, True]).reset_index(drop=True)
display(classifier_results_df)

best_model_name = classifier_results_df.iloc[0]['model_name']
heldout_topk_eval_df = heldout_artifacts[best_model_name]['topk']
heldout_bucket_eval = heldout_artifacts[best_model_name]['bucket_eval']

best_model = classifier_specs[best_model_name]
try:
    best_model.fit(X, y, sample_weight=(1.0 + model_df['observedBestRate'].astype(float)) / np.mean(1.0 + model_df['observedBestRate'].astype(float)))
except (TypeError, ValueError):
    final_weight = (1.0 + model_df['observedBestRate'].astype(float)).to_numpy(dtype=float)
    final_weight = final_weight / final_weight.mean()
    rng = np.random.default_rng(random_state)
    prob = final_weight / final_weight.sum()
    sampled = rng.choice(np.arange(len(X)), size=max(len(X) * 8, 4096), replace=True, p=prob)
    best_model.fit(X.iloc[sampled].reset_index(drop=True), y.iloc[sampled].reset_index(drop=True))

final_probs = predict_proba_with_fallback(best_model, X)
final_topk_eval_df, bucket_eval, final_class_order = evaluate_probs(model_df, final_probs)

bucket_eval_summary = pd.DataFrame({
    'eligible_bucket_rows': [len(bucket_eval)],
    'bucket_pair_match_rate': [float(bucket_eval['pairMatch'].mean()) if len(bucket_eval) else np.nan],
    'observed_best_rate_mean': [float(bucket_eval['observedBestRate'].mean()) if len(bucket_eval) else np.nan],
    'pred_best_prob_mean': [float(bucket_eval['predBestProb'].mean()) if len(bucket_eval) else np.nan],
    'pred_best_actual_rate_mean': [float(bucket_eval['predBestActualRate'].mean()) if len(bucket_eval) else np.nan],
    'mean_regret': [float(bucket_eval['regret'].mean()) if len(bucket_eval) else np.nan],
})
display(bucket_eval_summary)
display(heldout_topk_eval_df)
display(final_topk_eval_df)

export_df = pd.DataFrame({
    'name': [
        'best_model_name',
        'eligible_bucket_rows',
        'heldout_top1_hit_rate',
        'heldout_top3_hit_rate',
        'heldout_top5_hit_rate',
        'heldout_top1_mean_regret',
        'final_top1_hit_rate',
        'final_top3_hit_rate',
        'final_top5_hit_rate',
        'final_top1_mean_regret',
        'final_bucket_pair_match_rate',
    ],
    'value': [
        best_model_name,
        len(bucket_eval),
        float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 1, 'hit_rate'].iloc[0]),
        float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 3, 'hit_rate'].iloc[0]),
        float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 5, 'hit_rate'].iloc[0]),
        float(heldout_topk_eval_df.loc[heldout_topk_eval_df['top_k'] == 1, 'mean_regret'].iloc[0]),
        float(final_topk_eval_df.loc[final_topk_eval_df['top_k'] == 1, 'hit_rate'].iloc[0]),
        float(final_topk_eval_df.loc[final_topk_eval_df['top_k'] == 3, 'hit_rate'].iloc[0]),
        float(final_topk_eval_df.loc[final_topk_eval_df['top_k'] == 5, 'hit_rate'].iloc[0]),
        float(final_topk_eval_df.loc[final_topk_eval_df['top_k'] == 1, 'mean_regret'].iloc[0]),
        float(bucket_eval['pairMatch'].mean()),
    ]
})
display(export_df)
