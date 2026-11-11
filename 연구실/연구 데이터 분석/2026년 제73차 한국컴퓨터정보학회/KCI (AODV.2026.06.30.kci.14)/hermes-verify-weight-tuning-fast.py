import json
from pathlib import Path

NOTEBOOK = Path('v.13_score_weight_tuning_repro.ipynb')
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
code_cells = [''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type') == 'code']

g = {'__name__': '__main__'}

# Cell 0: imports + global defaults
exec(compile(code_cells[0], 'cell0.py', 'exec'), g)

# Fast reduced verification overrides using the real notebook logic on a smaller slice.
g['TRAIN_RUNS'] = [4, 10, 14]
g['VALID_RUNS'] = [31]
g['TEST_RUNS'] = [22]
g['ALL_RUNS'] = g['TRAIN_RUNS'] + g['VALID_RUNS'] + g['TEST_RUNS']
g['PER_RUN_SAMPLE'] = 5000
g['TARGET_COEFF'] = (2.0, 3.0, 0.02, 1.0, 0.25, 0.1)
g['CANDIDATE_GRID'] = {
    'aS': [2.0, 3.0],
    'aF': [3.0, 4.0, 5.0],
    'aD': [0.02],
    'aR': [1.0],
    'aI': [0.25],
    'aX': [0.1],
}
g['SHORTLIST_SIZE'] = 4
g['OUTPUT_DIR'] = g['BASE_DIR'] / 'outputs_v13_weight_tuning_verify_fast'
g['OUTPUT_DIR'].mkdir(exist_ok=True)

# Cell 1-3: dataclass/functions/data loading helpers
exec(compile(code_cells[1], 'cell1.py', 'exec'), g)
exec(compile(code_cells[2], 'cell2.py', 'exec'), g)
exec(compile(code_cells[3], 'cell3.py', 'exec'), g)

# Cell 4: evaluation logic, with reduced training budget for fast verification.
cell4 = code_cells[4]
cell4 = cell4.replace('patience=6', 'patience=3')
cell4 = cell4.replace('patience=3', 'patience=2')
cell4 = cell4.replace('epochs=40, batch_size=4096', 'epochs=12, batch_size=2048')
exec(compile(cell4, 'cell4_fast.py', 'exec'), g)

# Remaining cells: run the real pipeline on the reduced slice.
for idx in range(5, len(code_cells)):
    print(f'===== EXEC CELL {idx} =====', flush=True)
    exec(compile(code_cells[idx], f'cell{idx}.py', 'exec'), g)

print('FAST_VERIFY_DONE')
