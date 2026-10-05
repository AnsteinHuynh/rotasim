# Released weights license

The adapter checkpoint in this bundle is licensed **CC-BY-NC-4.0** (non-commercial).

Why: it was trained on a mixture that includes the **DiffIQA** dataset (A-FINE,
CVPR 2025, <https://github.com/ChrisDud0257/AFINE>), whose terms restrict the
images *and derived data* to non-commercial academic research use — the same
constraint that applies to the two `*-diffiqa-v1` bundles. The controller
parameters are small tensors derived from that corpus, so the strictest corpus
term governs.

If you use this checkpoint, please cite the A-FINE paper (CVPR 2025).

The other training corpora — **FGResQ** and **BAPPS** — carry no non-commercial
clause, but the DiffIQA term is the binding one here.

The **code** in `dreamsim_oft/` is licensed Apache-2.0 — see `LICENSE`.
