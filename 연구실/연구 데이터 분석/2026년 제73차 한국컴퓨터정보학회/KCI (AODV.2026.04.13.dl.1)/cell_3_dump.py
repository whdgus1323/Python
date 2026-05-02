state_df = pd.DataFrame({
    'localCbrNorm': np.clip(dataset['localCbr'].fillna(0).astype(float) / 100.0, 0, 1),
    'neighborNorm': np.clip(dataset['neighborCount'].fillna(0).astype(float) / neighbor_norm, 0, 1),
    'hopNorm': np.clip(dataset['hopCount'].fillna(0).astype(float) / hop_norm, 0, 1),
    'isDirectRoute': dataset['isDirectRoute'].fillna(0).astype(float),
})

pair_low = dataset['lowThresholdRaw'].astype(float)
pair_high = dataset['highThresholdRaw'].astype(float)
pair_gap = np.clip(pair_high - pair_low, min_threshold_gap, threshold_max - threshold_min)
pair_center = (pair_low + pair_high) / 2.0

pair_df = pd.DataFrame({
    'lowThresholdNorm': np.clip((pair_low - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1),
    'highThresholdNorm': np.clip((pair_high - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1),
    'gapNorm': np.clip((pair_gap - min_threshold_gap) / max(threshold_max - threshold_min - min_threshold_gap, 1e-6), 0, 1),
    'centerNorm': np.clip((pair_center - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1),
})

state_bucket_df = pd.concat([
    state_df[selected_state_features].copy(),
    pair_df[selected_pair_features].copy(),
    dataset[['lowThresholdRaw', 'highThresholdRaw', 'label']].reset_index(drop=True)
], axis=1)
state_bucket_df['stateBucket'] = (
    state_bucket_df['localCbrNorm'].round(bucket_round_digits).astype(str)
    + '|'
    + state_bucket_df['neighborNorm'].round(bucket_round_digits).astype(str)
    + '|'
    + state_bucket_df['hopNorm'].round(bucket_round_digits).astype(str)
    + '|'
    + state_bucket_df['isDirectRoute'].round(bucket_round_digits).astype(str)
)

bucket_pair_perf = (state_bucket_df
    .groupby(['stateBucket', 'lowThresholdRaw', 'highThresholdRaw'], observed=False)
    .agg(
        rows=('label', 'size'),
        label_rate=('label', 'mean'),
        localCbrNorm=('localCbrNorm', 'mean'),
        neighborNorm=('neighborNorm', 'mean'),
        hopNorm=('hopNorm', 'mean'),
        isDirectRoute=('isDirectRoute', 'mean'),
        lowThresholdNorm=('lowThresholdNorm', 'mean'),
        highThresholdNorm=('highThresholdNorm', 'mean'),
        gapNorm=('gapNorm', 'mean'),
        centerNorm=('centerNorm', 'mean'),
    )
    .reset_index())

bucket_summary = (bucket_pair_perf
    .groupby('stateBucket', observed=False)
    .agg(
        pair_count=('label_rate', 'size'),
        total_rows=('rows', 'sum'),
        localCbrNorm=('localCbrNorm', 'mean'),
        neighborNorm=('neighborNorm', 'mean'),
        hopNorm=('hopNorm', 'mean'),
        isDirectRoute=('isDirectRoute', 'mean'),
    )
    .reset_index())

candidate_pairs = (dataset[['lowThresholdRaw', 'highThresholdRaw']]
    .drop_duplicates()
    .sort_values(['lowThresholdRaw', 'highThresholdRaw'])
    .reset_index(drop=True))
candidate_pairs['lowThresholdNorm'] = np.clip((candidate_pairs['lowThresholdRaw'] - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1)
candidate_pairs['highThresholdNorm'] = np.clip((candidate_pairs['highThresholdRaw'] - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1)
candidate_pairs['gapNorm'] = np.clip(((candidate_pairs['highThresholdRaw'] - candidate_pairs['lowThresholdRaw']) - min_threshold_gap) / max(threshold_max - threshold_min - min_threshold_gap, 1e-6), 0, 1)
candidate_pairs['centerNorm'] = np.clip((((candidate_pairs['lowThresholdRaw'] + candidate_pairs['highThresholdRaw']) / 2.0) - threshold_min) / max(threshold_max - threshold_min, 1e-6), 0, 1)

eligible_buckets = bucket_summary[(bucket_summary['pair_count'] >= bucket_min_pairs) & (bucket_summary['total_rows'] >= bucket_min_rows)].copy()
display(pd.DataFrame({
    'eligible_buckets': [len(eligible_buckets)],
    'all_buckets': [state_bucket_df['stateBucket'].nunique()],
    'candidate_pairs': [len(candidate_pairs)],
    'bucket_rows': [len(bucket_pair_perf)],
}))
