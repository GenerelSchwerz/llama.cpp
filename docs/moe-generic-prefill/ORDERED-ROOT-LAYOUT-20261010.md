# Expert-order root layout experiment, 2026-10-10

## Source change

Strata prefill.cpp:3659-3675 keeps combined gate/up in expert order using identity output rows. Retained generic100 maps separate roots back to token-route order, then GET_ROWS gathers them back. Prototype180 extends the existing prepared mapped-MMQ launch with a checked optional output-row map. It uses each eligible root's existing gather-index storage, padded by the backend requirement query; zeroed guard tail and stream lifetime stay explicit. Root ordering is private: direct live outputs and their view aliases retain original route order. Existing GET_ROWS remains. No new kernel, fused arithmetic, model dimensions, executor or residency owner is added.

The routed-resource declaration moves from the kernel-heavy mmq.cuh into the existing host-readable mmid.cuh so fixtures query the same bounds. The prototype adds no public CLI or stable API.

## Qualification

Build188 passes all nine configure/build/discovery/body/overlap/owner/direct-reader/view-reader commands under ordered locks with a plain j18 build.168 main-source numerical replays have maximum relative MSE1.60269249e-7 against the original mapped oracle.56 direct and56 nonzero-offset contiguous-view reader checks pass.16 independent ordered-root cases are bit-exact, including null/short output-map rejection;96 existing canonical-owner cases pass. Other structural/format paths retain required-hybrid execution. These are focused single-device Linux checks, not broad platform/request/MTP qualification.

Failed attempts remain external:178 finite queue timeout before build;179 queried an op-NONE private storage clone rather than original operation metadata;185/186 external-reader fixture setup omitted its live-output declaration and then used a legacy single-output metadata helper;187 corrects routed metadata and passes direct readers, but its feature-sliced public view hits an unchanged HEAD publication restriction. The partial fixture mode remains. HEAD core:1410/current1636 rejects noncontiguous private-to-public publication; do not claim those public layouts work. All attempt trees are empty.

## Unprofiled ABBA181-184

Each process handles two full uncached2049-token prompts and128 outputs, no MTP, ubatch2048, CPU prefill off, same profile/startup capacity and required source route. All harness, route and whole-tree teardown gates pass. Cold selected payload is27043993600bytes in every arm, with47 bodies431 waves689 chunks. Candidate orders94 roots. Shared compact workspace grows37109760 ->37110784bytes; scratch peak100288512bytes and sampled GPU peak12363MiB are unchanged.

| Phase | Control prompt tok/s | Candidate prompt tok/s | Control request seconds | Candidate request seconds |
| --- | ---: | ---: | ---: | ---: |
| Cold |1065.40|989.22|4.30111|4.45446|
| Repeated full prompt |1427.84|1432.50|3.66191|3.66568|

Warm throughput changes+0.326%, while whole request changes+0.103% slower. Cold throughput changes-7.15% and whole request+3.57% slower. Two observations per arm and known cold preparation variance do not establish the cause of the cold decline. There is no meaningful qualified gain. Candidate181 cold and184 repeated outputs diverge at zero-based token97; other comparisons match128 tokens. Repeated main residency/work differs, including waves420 versus418 and selected payload. Do not attribute late divergence to a specific CPU/GPU/adaptation cause without evidence, relax numerical bounds, or treat the warm ratio as matched-work proof.

Raw report: external generic-prefill-gap-20261009/ORDERED-ROOT-ABBA-REPORT.json. Candidate180 remains a frozen unqualified prototype; retained100 is the control. No commit, push or default promotion. The1700 end-to-end objective is open.

## Next change

The ordered layout now permits removing private root GET_ROWS through existing tensor views. Make an eligible gathered value a compact view into the full retained root, bind its row offset per chunk and propagate data pointers through the existing topological view chain. Retain the identity output-map tensor explicitly as a graph leaf so the allocator owns it even without a gather dependency. Keep external roots on the existing gather/scatter path. Use original mapped arithmetic, checked spans, fixed owner leases and bounded storage; do not create a new allocator or kernel. Qualify short/skewed chunks and direct/view readers before model serving, then compare against both180 and retained100. Route-helper reuse, cross-layer issuance and cold preparation remain separate unresolved costs.
