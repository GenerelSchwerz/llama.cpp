# Resource preflight experiment, rejected

Current retained source is restored exactly to frozen100. Candidate153, patches, binaries and all failures remain in external generic-prefill-gap-20261009. No commit, push or promotion. The1700 tok/s goal remains open.

## Implementation and qualification

The experiment reused successful checked ordinary scratch/library queries to omit duplicate kernel emission only during prefill resource capture. Initial146 was not exercised because source graph fusion dispatched before the shortcut; diagnostic147 reported zero omissions. Corrected153 passed an optional preflight-only query mask through the existing upstream graph dispatcher, after fusion selection. Original fused groups, input/reuse images, unknown operations and expert projection measurement remained captured. Replay/decode arithmetic, resource estimates, binding validation and canonical owners were unchanged.

Combined152 passes13 commands: configure/buildj18; CPU discovery and body program; mixed-format complete bodies; held-reader overlap; owner; sort/softmax/scratch/matmul resource checks; attention; CPU/GPU prefill overlap. Added synthetic resource fixtures110/117/124 were removed after exposing their handcrafted region-boundary assumptions. Tests returned to exact100 source. Setup mistakes in131 legacy attention selection and138 CPU-off overlap were corrected without changing tolerances or counters; all failed evidence and empty teardown trees are preserved.

Diagnostic154 skips523 covered operations, with no recaptures, pool growth or library requests. Main capture185.44ms versus earlier101248.53ms is a component observation. Unprofiled frozen153/100 ABBA155-158 preserves2049 prompt IDs, four output IDs846,198,7734,264,3072 configured startup slots/5364121600 copied payload bytes, source mainCPU/failures/fallback0,47 bodies431waves689chunks47sharedinputs94sharedplans,128MiB staging and12349MiB sampled peak.

| Run | Arm | Prefill tok/s | Main prepare ms | Main replay ms |
| --- | --- | ---: | ---: | ---: |
|155|Candidate|1137.15|249.45|1167.86|
|156|Control|729.28|612.12|1130.24|
|157|Control|1063.85|376.66|1128.97|
|158|Candidate|944.57|269.40|1188.31|

Mean1040.86 versus896.56 (+16.09% observational) is inflated by slow control156. Candidate main replay1178.09 versus1129.61ms is4.29% slower. Do not claim an end-to-end speedup.

## Cold and repeated full prompt diagnostics

These are two fresh contexts, two complete identical requests each, cache_prompt=false and verified cache_n0/prompt_n2049. They are diagnostic observations, not a repeated speed qualification. All output IDs match and teardown passes.

| Arm | Request | Prefill tok/s | Main capture ms | Main prepare ms | Main replay ms |
| --- | --- | ---: | ---: | ---: | ---: |
|Control159|Cold|1115.01|225.01|298.98|1128.57|
|Candidate160|Cold|987.40|186.36|276.86|1206.08|
|Control159|Repeated full|1405.91|11.54|62.95|1116.59|
|Candidate160|Repeated full|1418.33|9.62|62.32|1115.69|

Cold capture saves38.66ms but main replay costs77.51ms more. Warm execution is essentially unchanged. This is consistent with moving first-use kernel loading from capture into replay; there is no driver/module attribution trace proving the entire delta. Total cold prefill is worse in the followup and the combined evidence does not establish a gain. Reject153 rather than retain extra interface/dispatch complexity. Three source files were restored from the exact saved pre-experiment index; all100 source hashes match. Build161 passes all seven original checks and returns the owned mutable build to that source, with unchanged source hashes and empty teardown tree.

Canonical Strata generate.cpp defaults CUDA_MODULE_LOADING toEAGER only when unset. Frozen specialized52's preserved native-results/result.json explicitly recordsLAZY, matching generic controls. This default difference does not explain the matched comparison. Generic eager diagnostic71 was already rejected on initialization/peak-memory evidence.

## Next evidence

Stop optimizing capture-only counters. Attribute the still-unmeasured request preparation outside core counters with a bounded external timing probe around scheduler preparation/allocation and existing frontend discovery, retaining exact first/last-node geometry and nested timing. Inspect existing CPU preparation and source graph allocation before proposing lifetime changes. Separately measure upstream CPU input embedding placement through an existing tensor override if needed. Cross-layer copy-ahead and cache lending remain source-backed scheduling differences, not quantified promises. Keep cold total, startup and full repeated prompts separate. Do not replace the1700 end-to-end target with warm/component equivalence.
