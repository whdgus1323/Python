# Research Summary

This document summarizes the work directly inspected in this workspace and the linked experiment folders as of 2026-06-11.

This file is intentionally strict about evidence.

- Facts labeled `Verified` were directly read from files in the workspace or from the specified external experiment folders.
- Items labeled `Interpretation` are reasoned conclusions from verified data.
- Items labeled `Unknown` or `Not yet verified` were not directly established from the available files.
- No claim below is intended to hide uncertainty.

## 1. Scope

The following locations were directly used during analysis:

- Workspace root:
  `[KCI (AODV.2026.06.08.kci.5)](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5))`
- Main notebook analyzed first:
  `[v.7_regression.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.7_regression.ipynb)`
- Later notebook variants inspected:
  `[v.10_regression_runtime_aligned.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.10_regression_runtime_aligned.ipynb)`
  `[v.11_regression_conservative_teacher.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.11_regression_conservative_teacher.ipynb)`
  `[v.12_regression_score_tolerance_d3.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.12_regression_score_tolerance_d3.ipynb)`
  `[v.13_regression_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.13_regression_stability_score.ipynb)`
  `[v.14_regression_aggressive_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.14_regression_aggressive_stability_score.ipynb)`
- Dataset outputs used for row/state inspection:
  `[dataset_outputs](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\dataset_outputs)`
- External runtime experiment folder explicitly requested by the user:
  `C:\Users\Choe JongHyeon\Desktop\map_v3`

## 2. High-Level Research Framing

### 2.1 Current modeling style

Verified from `[v.7_regression.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.7_regression.ipynb)`:

- The current deep learning pipeline is not online reinforcement learning.
- It first constructs an offline teacher policy from simulation logs.
- It then trains a neural network to predict threshold pairs `(low, high)` from current state features.

Interpretation:

- The most accurate description is `offline teacher policy distillation` or `supervised policy approximation`.
- It is not correct to describe the current method as runtime online optimization.

### 2.2 Why this matters

Interpretation based on the inspected code structure:

- The network does not directly optimize routing performance at runtime.
- It approximates a threshold policy that was already chosen offline by a score function.
- Therefore, any claim about “optimality” is conditional on the score definition used during teacher generation.

## 3. v.7 Regression Notebook

Source:
`[v.7_regression.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.7_regression.ipynb)`

### 3.1 Structure

Verified:

- The notebook has 2 code cells.
- Cell 0 contains the full data loading, teacher creation, model training, and evaluation pipeline.
- Cell 1 exports trained weights into OMNeT++ `.ini` style lines.

### 3.2 Data sources

Verified:

- Decision log:
  `aodv_cbr_rrep_decisions.csv`
- Future outcome log:
  `aodv_transmission_failure_diagnosis_1s.csv`

### 3.3 Run split

Verified:

- `TRAIN_RUNS = [4, 10, 14, 15, 16, 17, 18, 19, 20, 28]`
- `VALID_RUNS = [31, 35]`
- `TEST_RUNS = [22, 25]`

Interpretation:

- This is run-level splitting, not row-level random splitting.
- That reduces leakage from highly similar rows within the same simulation run.

### 3.4 Sampling

Verified:

- `PER_RUN_SAMPLE = 60000`
- Sampling is performed while reading `aodv_cbr_rrep_decisions.csv` in chunks.
- Only rows with valid `time`, `node`, `hopCount`, `localCbr`, `appliedLowThreshold`, and `appliedHighThreshold` are kept.
- `hopCount` is restricted to `1..6`.

Important limitation:

- This is not a fully uniform global sample over the entire file.
- It is chunk-based capped sampling.

### 3.5 Features and target

Verified:

- Input features:
  - `localCbr`
  - `neighborCount`
  - `hopCount`
  - `isDirectRoute`
- Target:
  - `labelLowCont`
  - `labelHighCont`

### 3.6 State abstraction

Verified:

- `STATE_CBR_BIN = 5`
- `STATE_NEIGHBOR_BIN = 5`
- `stateHopBin = round(hopCount)` clipped to `1..6`
- `stateDirectBin = isDirectRoute`
- `stateKey = stateCbrBin|stateNeighborBin|stateHopBin|stateDirectBin`

### 3.7 Original teacher score

Verified:

- `offlineScore = 2.0 * futureSucceeded - 3.0 * futureFailed - 0.02 * futureDelay`

### 3.8 Teacher creation

Verified:

- Candidate pairs are grouped by:
  - state bins
  - `lowInt`
  - `highInt`
- Mean score and support are computed.
- Pairs with support lower than `MIN_PAIR_SUPPORT` are removed.
- For each state, the top-scoring pair becomes the teacher target.

### 3.9 Model

Verified:

- Standardization is applied to input and output.
- Model:
  - Dense 256 ReLU
  - Dense 256 ReLU
  - Dense 128 ReLU
  - Dense 2 linear output
- Optimizer:
  - Adam `1e-3`
- Loss:
  - MSE

### 3.10 Saved outputs

Verified:

- `[outputs_v7_regression_run_split](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\outputs_v7_regression_run_split)`
- Files:
  - `summary.csv`
  - `summary.json`
  - `state_eval.csv`

### 3.11 Saved v.7 metrics

Verified from
`[outputs_v7_regression_run_split\summary.json](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\outputs_v7_regression_run_split\summary.json)`:

- `full_rows = 770469`
- `teacher_states = 705`
- `train_rows = 569638`
- `valid_rows = 112329`
- `test_rows = 39171`
- `train_states = 705`
- `valid_states = 705`
- `test_states = 474`
- `row_r2_low = 0.6003235578536987`
- `row_r2_high = 0.5435311794281006`
- `row_mae_low = 0.6572465300559998`
- `row_mae_high = 1.015523910522461`
- `state_r2_low = 0.732313299227641`
- `state_r2_high = 0.7138319813193019`
- `state_mae_low = 0.5617854082131688`
- `state_mae_high = 0.8685279234552182`

### 3.12 Training log note

Verified from notebook outputs:

- Training reached epoch 40.
- Validation loss decreased through most of training and did not show obvious catastrophic divergence.

## 4. Main Methodological Limits Identified

These are not accusations of failure. They are the research risks and interpretation limits that were repeatedly identified from direct code inspection.

### 4.1 It is not online optimization

Verified from code design:

- The system does not search thresholds online during runtime.
- It learns from an offline teacher extracted from logs.

### 4.2 Score dependence

Verified:

- Teacher policy depends on the score formula.

Interpretation:

- Changing score weights changes what “best threshold” means.

### 4.3 Unseen-state issue

Verified:

- Valid/test rows are merged only with states that exist in the train teacher table.

Interpretation:

- Performance is evaluated on teacher-covered state space.
- Completely unseen states are not demonstrated by this pipeline.

### 4.4 Output constraint issue

Verified:

- The model predicts continuous `low/high`.
- The training loss itself does not enforce semantic constraints such as “safe operating region”.

Interpretation:

- Runtime post-processing matters.
- If scaling/export/runtime logic is wrong, outputs can collapse or clip.

### 4.5 Threshold prediction is not the same as routing improvement

Interpretation based on the experiment structure:

- The notebook evaluates threshold prediction quality.
- It does not itself prove final routing improvement in a live runtime.
- Runtime experiments are needed for that link.

## 5. Node[8] Analysis

This section is intentionally limited to what was directly checked.

### 5.1 Data source used

Verified:

- `[dataset_outputs\row_supervised_full.csv](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\dataset_outputs\row_supervised_full.csv)`
- Raw runtime logs under:
  `C:\Users\Choe JongHyeon\Desktop\map_v2\random_base1\*\seed_1`

### 5.2 Direct facts for node[8]

Verified:

- In `row_supervised_full.csv`, `node[8]` had `2417` rows.
- Splits in that aggregated dataset:
  - `train = 1691`
  - `valid = 330`
  - `test = 396`
- Present runs in that aggregated inspection:
  - `[4, 10, 14, 15, 16]`

### 5.3 Whether node[8] ever used threshold 0,5 in the inspected map_v2 runs

Verified by direct scan of `aodv_cbr_rrep_decisions.csv` across runs:

- For runs
  `[4, 10, 14, 15, 16, 17, 18, 19, 20, 28, 31, 35, 22, 25]`
- `node[8]` had `0` rows with:
  - `lowInt == 0`
  - `highInt == 5`

Important consequence:

- For `node[8]` in the inspected `map_v2` training/test logs, there was no direct evidence that actual threshold pair `(0,5)` was applied.

### 5.4 Whether node[8] had state bin `0|5|...`

Verified:

- In raw diagnosis rows for `node[8]`, there were `35` rows with:
  - `stateCbrBin == 0`
  - `stateNeighborBin == 5`
- In those `35` rows:
  - `routeDiscoveryStarted = 0`
  - `routeDiscoverySucceeded = 0`
  - `routeDiscoveryFailed = 0`
  - `routeDiscoveryDelayAvgMs = 0`

Interpretation:

- For `node[8]`, that `0|5` state region was idle, not active route-discovery behavior.
- Therefore, that specific `node[8]` evidence does not support the claim that `node[8]` itself caused a PDR improvement through `0|5` active routing behavior.

### 5.5 Best-performing node[8] threshold pairs in aggregated supervised data

Verified from grouped `node[8]` rows in `row_supervised_full.csv`:

Top mean score pairs included:

- `19,52`
- `20,51`
- `16,52`
- `20,50`
- `17,51`

Interpretation:

- In aggregated offline supervision data, `node[8]`'s useful threshold region was not centered on `(0,5)`.

## 6. Runtime Experiment Comparison on map_v3

User clarification:

- `test` folder is the `legacy` baseline.

The following folders were directly compared for scenario `8`:

- `C:\Users\Choe JongHyeon\Desktop\map_v3\test\8`
- `C:\Users\Choe JongHyeon\Desktop\map_v3\train_d\8`
- `C:\Users\Choe JongHyeon\Desktop\map_v3\train_d_2\8`

Not directly available at analysis time:

- `C:\Users\Choe JongHyeon\Desktop\map_v3\train_d_3\8`

Verified:

- `train_d_3\8` did not exist at the time of direct check.
- The user later clarified that `d_3` had not been run yet.

## 7. Runtime PDR Results for Scenario 8

### 7.1 Legacy = test

Verified from
`C:\Users\Choe JongHyeon\Desktop\map_v3\test\8\result.txt`
and
`C:\Users\Choe JongHyeon\Desktop\map_v3\test\8\PDR.txt`:

- Sent: `7000`
- Received: `1646`
- Final PDR: `23.5143%` in `result.txt`
- Last-line PDR in `PDR.txt`: `23.7826`
- PDR around time 40 from `PDR.txt`: `69.3`

### 7.2 d = train_d

Verified from
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d\8\result.txt`
and
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d\8\PDR.txt`:

- Sent: `7000`
- Received: `5337`
- Final PDR: `76.2429%` in `result.txt`
- Last-line PDR in `PDR.txt`: `75.8986`
- PDR around time 40 from `PDR.txt`: `64.9`

### 7.3 d_2 = train_d_2

Verified from
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d_2\8\result.txt`
and
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d_2\8\PDR.txt`:

- Sent: `7000`
- Received: `2065`
- Final PDR: `29.5%` in `result.txt`
- Last-line PDR in `PDR.txt`: `29.8116`
- PDR around time 40 from `PDR.txt`: `77.0`

## 8. Runtime Mechanism Differences Across test, d, d_2

### 8.1 Direct aggregate diagnosis metrics

Verified from direct aggregation of
`aodv_transmission_failure_diagnosis_1s.csv`:

#### test

- `routeDiscoveryStarted_sum = 1424`
- `routeDiscoverySucceeded_sum = 1364`
- `routeDiscoveryFailed_sum = 8`
- `rreqReceived_sum = 758319`
- `rrepReceived_sum = 1525`
- `rrepCandidates_sum = 15652`
- `rrepAllowed_sum = 15652`
- `rrepBlocked_sum = 0`
- `routeInvalidate_sum = 15394`
- `routeExpireInactive_sum = 32884`
- `routeDelete_sum = 2549`
- `rerrOriginated_sum = 9530`
- `delay_weighted_avg_ms = 205.682922`

#### d

- `routeDiscoveryStarted_sum = 403`
- `routeDiscoverySucceeded_sum = 395`
- `routeDiscoveryFailed_sum = 8`
- `rreqReceived_sum = 347500`
- `rrepReceived_sum = 403`
- `rrepCandidates_sum = 11088`
- `rrepAllowed_sum = 1637`
- `rrepBlocked_sum = 9451`
- `routeInvalidate_sum = 2840`
- `routeExpireInactive_sum = 29225`
- `routeDelete_sum = 10166`
- `rerrOriginated_sum = 1580`
- `delay_weighted_avg_ms = 253.363046`

#### d_2

- `routeDiscoveryStarted_sum = 1563`
- `routeDiscoverySucceeded_sum = 1528`
- `routeDiscoveryFailed_sum = 12`
- `rreqReceived_sum = 882728`
- `rrepReceived_sum = 1668`
- `rrepCandidates_sum = 22137`
- `rrepAllowed_sum = 11701`
- `rrepBlocked_sum = 10436`
- `routeInvalidate_sum = 15044`
- `routeExpireInactive_sum = 31468`
- `routeDelete_sum = 1637`
- `rerrOriginated_sum = 9039`
- `delay_weighted_avg_ms = 200.909161`

### 8.2 Important direct comparison

Verified:

- `d` has much lower:
  - route discovery starts
  - RREQ receptions
  - route invalidations
  - RERR originations
- `d_2` has:
  - much more normal-looking route activity than `d`
  - but much lower final PDR than `d`

Interpretation:

- The large PDR increase in `d` is strongly associated with severe suppression of routing/control propagation.
- It is not consistent with “normal DL threshold prediction worked well”.

## 9. What Actually Happened in d

### 9.1 Debug file

Verified from
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d\8\aodv_dl_direct_threshold_debug.csv`:

- File row count: `11088`
- Columns include:
  - `rawLow`
  - `rawHigh`
  - `predictedLow`
  - `predictedHigh`
  - `decision`

### 9.2 Predicted threshold collapse

Verified:

- In `train_d/8`, all `11088` rows rounded to:
  - `predictedLow = 0`
  - `predictedHigh = 5`

This was directly confirmed by grouping rounded predictions.

### 9.3 Decision counts in d

Verified:

- `dl_direct_threshold_blocked = 9451`
- `dl_direct_threshold_allowed = 919`
- `dl_direct_threshold_direct_bypass_allow = 718`

### 9.4 Interpretation of d

Interpretation, strongly grounded in the verified debug and diagnosis logs:

- `d` is not evidence that a healthy DL threshold predictor produced a better nuanced policy.
- `d` is evidence that a broken or collapsed output policy `(0,5)` caused extremely aggressive blocking.
- That aggressive blocking greatly reduced control overhead and route churn in scenario 8.
- In this specific scenario, that suppression coincided with much higher final PDR.

This is an interpretation, but it is the most evidence-consistent interpretation found during the analysis.

## 10. What Happened in d_2

### 10.1 Debug file

Verified from
`C:\Users\Choe JongHyeon\Desktop\map_v3\train_d_2\8\aodv_dl_direct_threshold_debug.csv`:

- File row count: `22137`
- Thresholds were not collapsed to `(0,5)`.
- Example rounded predictions clustered around:
  - `19,48`
  - `20,47`
  - `17,48`
  - `17,49`
  - `17,47`
  - `18,47`
  - `18,48`

### 10.2 0,5 clipping in d_2

Verified:

- Rows with `predictedLow <= 0` and `predictedHigh <= 5`: `0`

### 10.3 Decision counts in d_2

Verified:

- `dl_direct_threshold_allowed = 10710`
- `dl_direct_threshold_blocked = 10436`
- `dl_direct_threshold_direct_bypass_allow = 991`

### 10.4 Interpretation of d_2

Interpretation:

- `d_2` behaved like a normal threshold-predicting DL system.
- It did not produce the pathological `(0,5)` global clamp seen in `d`.
- However, in scenario 8 that normal-looking policy did not produce the same drastic reduction in route churn/control load as `d`.
- Therefore, its PDR remained far below `d`.

## 11. Core Runtime Conclusion from Scenario 8

This is the central conclusion supported by the direct scenario-8 comparison.

### Verified facts supporting the conclusion

- `d` had far more blocking than legacy and far fewer route discovery/control events.
- `d` had far fewer:
  - `rerrOriginated`
  - `routeInvalidate`
  - `routeDiscoveryStarted`
  - `rreqReceived`
- `d` had much higher final PDR than both legacy and `d_2`.
- `d` was also visibly abnormal because all predicted thresholds collapsed to `(0,5)`.

### Interpretation

- In scenario 8, the observed PDR gain in `d` is best explained by aggressive suppression of route expansion and resulting reductions in route churn and control overhead.
- Put differently:
  - the mechanism appears to be `stability-through-suppression`,
  - not `accurate continuous threshold regression`.

## 12. Why Adding Mild Stability Penalties Did Not Change PDR Much

### 12.1 v.13 notebook

Source:
`[v.13_regression_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.13_regression_stability_score.ipynb)`

Verified:

- Added score penalties:
  - `rerrOriginated`
  - `routeInvalidate`
  - `routeDelete`
- Weights:
  - `success = 2.0`
  - `fail = 3.0`
  - `delay = 0.02`
  - `rerr = 1.0`
  - `invalidate = 0.25`
  - `delete = 0.1`

### 12.2 v.13 outputs

Verified from
`[outputs_v13_regression_stability_score\summary.json](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\outputs_v13_regression_stability_score\summary.json)`:

- `teacher_states = 984`
- `row_r2_low = 0.6562938094139099`
- `row_r2_high = 0.4532298445701599`
- `state_r2_low = 0.7345414344743915`
- `state_r2_high = 0.6052107615369812`

### 12.3 Why runtime PDR likely did not move much

Verified from
`[outputs_v13_regression_stability_score\teacher_table.csv](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\outputs_v13_regression_stability_score\teacher_table.csv)`:

- Teacher thresholds still cluster heavily in roughly:
  - `low = 19..20`
  - `high = 45..52`

Comparison against v.12 inferred teacher labels from state eval:

- v.12 already clustered around similar regions.
- v.13 did not drastically shift teacher policy lower/more conservative.

Interpretation:

- The added penalties were not strong enough, or not frequently active enough, to actually move the selected teacher thresholds in a major way.
- Therefore runtime behavior probably stayed similar.

## 13. Newly Created Notebooks

### 13.1 v.13

Created:
`[v.13_regression_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.13_regression_stability_score.ipynb)`

Purpose:

- Add `RERR`, `routeInvalidate`, and `routeDelete` penalties into teacher score.

Outputs:

- `[outputs_v13_regression_stability_score](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\outputs_v13_regression_stability_score)`

### 13.2 v.14

Created:
`[v.14_regression_aggressive_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.14_regression_aggressive_stability_score.ipynb)`

Purpose:

- Same structure as v.13.
- Stronger penalties so teacher thresholds may move more aggressively.

Verified weights in v.14:

- `success = 2.0`
- `fail = 3.0`
- `delay = 0.02`
- `rerr = 5.0`
- `invalidate = 2.0`
- `delete = 1.0`

Outputs when executed will go to:

- `outputs_v14_regression_aggressive_stability_score`

### 13.3 Important status note

Verified:

- `v.14` was created but not executed during the analysis recorded in this summary.

Therefore:

- No claims are made here about `v.14` performance.

## 14. Current Best Research Interpretation

This section is the best-supported interpretation, not a mathematical proof.

### 14.1 What scenario 8 appears to teach

Interpretation:

- In scenario 8, reducing `RERR` and route churn appears to be more important than precisely regressing a continuous threshold pair.
- The `d` result suggests that aggressive suppression of unstable/expansive route behavior can dramatically improve PDR in at least some settings.

### 14.2 What this does not prove

Unknown / not established:

- It is not proven that globally lower thresholds always improve PDR.
- It is not proven that `0,5` is a good policy in general.
- It is not proven that the same behavior helps in other scenarios.
- It is not proven that node-local evidence from `node[8]` alone explains the whole scenario-level PDR increase.

### 14.3 More careful phrasing

Safer interpretation:

- In the inspected scenario 8 runtime, a collapsed `(0,5)` threshold output strongly increased blocking and coincided with major reductions in route churn and control overhead, alongside a major PDR increase.

That is a factual-runtime observation plus interpretation.

## 15. Implications for Future Notebook Design

These items are recommendations derived from the inspected behavior.

### 15.1 Threshold regression alone may not be the right objective

Interpretation:

- If the important runtime mechanism is route-stability control, then predicting `(low, high)` accurately may be secondary.

### 15.2 Better design direction

Interpretation:

- Policy-focused modeling may be more appropriate than pure continuous threshold regression.

Examples:

- Stage 1:
  predict conservative vs moderate vs open behavior
- Stage 2:
  predict actual threshold values only within the chosen policy region

### 15.3 Better score candidates

Interpretation:

- If the goal is to reproduce the beneficial part of `d` without pathological collapse, future score design may need to include additional control-load terms such as:
  - `routeDiscoveryStarted`
  - `rreqReceived`
  - `rrepAllowed`
  - `rrepBlocked`

Important note:

- This was discussed as a research direction.
- It was not yet implemented in a verified notebook during this analysis.

## 16. Things Not Yet Done

Verified not completed during this session:

- `train_d_3/8` runtime experiment does not exist yet.
- `v.14` was not executed yet.
- No new runtime experiment was run from the newly created notebooks during this summary period.
- No final demonstration was produced that a new score improves runtime PDR over `d_2`.

## 17. Summary in Plain Language

Verified facts and safest interpretation combined:

- The current research pipeline is offline teacher-policy learning, not online optimization.
- `v.7` learns threshold pairs from state features using an offline score based on success, failure, and delay.
- In scenario 8 runtime tests on `map_v3`, the abnormal `d` experiment achieved very high PDR.
- That same `d` experiment also showed total threshold collapse to `(0,5)` for all debugged decisions.
- In `d`, blocking was extremely high and control/routing activity was drastically reduced.
- `RERR`, `routeInvalidate`, and route discovery counts dropped sharply relative to legacy.
- `d_2` used normal-looking DL thresholds, but its PDR was much lower than `d`.
- The best-supported interpretation is that, in scenario 8, aggressive route/control suppression reduced route churn and improved effective delivery.
- Adding mild stability penalties in `v.13` did not seem sufficient to move the teacher policy much.
- A stronger-penalty notebook `v.14` was created for the next experiment, but has not yet been run.

## 18. Exact Files Created During This Work

Verified created in the workspace:

- `[v.13_regression_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.13_regression_stability_score.ipynb)`
- `[v.14_regression_aggressive_stability_score.ipynb](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\v.14_regression_aggressive_stability_score.ipynb)`
- `[RESEARCH_SUMMARY_2026-06-11.md](D:\Github\Python\연구실\연구 데이터 분석\2026년 제73차 한국컴퓨터정보학회\KCI (AODV.2026.06.08.kci.5)\RESEARCH_SUMMARY_2026-06-11.md)`

## 19. Final Caution

This document is comprehensive within the limits of what was directly inspected.

It does not claim:

- that every notebook in the repository was exhaustively executed,
- that every scenario behaves like scenario 8,
- or that the current strongest interpretation has been experimentally proven beyond the inspected files.

It is meant to be a full, explicit, evidence-first record of what was actually established.
