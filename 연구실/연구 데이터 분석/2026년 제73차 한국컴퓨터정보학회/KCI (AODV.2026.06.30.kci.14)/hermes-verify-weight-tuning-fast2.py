import json
from pathlib import Path

NOTEBOOK = Path('v.13_score_weight_tuning_repro.ipynb')
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
code_cells = [''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type') == 'code']

g = {'__name__': '__main__'}

exec(compile(code_cells[0], 'cell0.py', 'exec'), g)

g['TRAIN_RUNS'] = [4, 10]
g['VALID_RUNS'] = [31]
g['TEST_RUNS'] = [22]
g['ALL_RUNS'] = g['TRAIN_RUNS'] + g['VALID_RUNS'] + g['TEST_RUNS']
g['TARGET_COEFF'] = (2.0, 3.0, 0.02, 1.0, 0.25, 0.1)
g['CANDIDATE_GRID'] = {
    'aS': [2.0, 3.0],
    'aF': [3.0, 4.0, 5.0],
    'aD': [0.02],
    'aR': [1.0],
    'aI': [0.25],
    'aX': [0.1],
}
g['OUTPUT_DIR'] = g['BASE_DIR'] / 'outputs_v13_weight_tuning_verify_fast2'
g['OUTPUT_DIR'].mkdir(exist_ok=True)

exec(compile(code_cells[1], 'cell1.py', 'exec'), g)
exec(compile(code_cells[2], 'cell2.py', 'exec'), g)
exec(compile(code_cells[3], 'cell3.py', 'exec'), g)

cell4 = code_cells[4]
cell4 = cell4.replace('patience=6', 'patience=2')
cell4 = cell4.replace('patience=3', 'patience=1')
cell4 = cell4.replace('epochs=40, batch_size=4096', 'epochs=8, batch_size=1024')
exec(compile(cell4, 'cell4_fast.py', 'exec'), g)

# load and aggressively subsample while preserving split-wise real data usage
exec(compile(code_cells[5], 'cell5.py', 'exec'), g)
base_df = g['base_df']
parts = []
for split, n in [('train', 20000), ('valid', 5000), ('test', 5000)]:
    part = base_df[base_df['split'] == split]
    take = min(n, len(part))
    parts.append(part.sample(n=take, random_state=g['SEED']) if take < len(part) else part)
g['base_df'] = g['pd'].concat(parts, ignore_index=True)
print('reduced_base_df_shape', g['base_df'].shape, flush=True)

# stage0 distribution + stage1 ranking
exec(compile(code_cells[6], 'cell6.py', 'exec'), g)
exec(compile(code_cells[7], 'cell7.py', 'exec'), g)

# shortlist: top3 + current coeff row, to keep the paper's current weights in direct comparison
stage1_df = g['stage1_df']
current = stage1_df[
    (stage1_df['aS'] == 2.0) & (stage1_df['aF'] == 3.0) & (stage1_df['aD'] == 0.02) &
    (stage1_df['aR'] == 1.0) & (stage1_df['aI'] == 0.25) & (stage1_df['aX'] == 0.1)
]
shortlist_df = g['pd'].concat([stage1_df.head(3), current], ignore_index=True).drop_duplicates().reset_index(drop=True)
g['shortlist_df'] = shortlist_df
shortlist_df.to_csv(g['OUTPUT_DIR'] / '01b_shortlist_used.csv', index=False, encoding='utf-8-sig')
print(shortlist_df[['stage1_rank','aS','aF','aD','aR','aI','aX']].to_string(index=False), flush=True)

# stage2 + selection summary
exec(compile(code_cells[9], 'cell9.py', 'exec'), g)
exec(compile(code_cells[10], 'cell10.py', 'exec'), g)
print('FAST2_VERIFY_DONE', flush=True)
