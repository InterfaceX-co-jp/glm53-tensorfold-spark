# Published vLLM reference receipts (Alex Ellis, RigMark)

Unmodified copies of Alex Ellis's published RigMark receipts and cards for GLM-5.3-Flash on DGX Spark, taken from
[alexellis/rigmark](https://github.com/alexellis/rigmark) at commit `c5a0db0` (`results/reference/`), included so the
comparison in [`../README.md`](../README.md) can be checked without leaving this repo. They are Alex's measurements,
not ours: vLLM, LibertAIDAI/GLM-5.3-Flash-NVFP4 (TP2, two directly connected Sparks) and Red Hat NVFP4 (TP4, four
Sparks). Copyright (c) 2026 Alex Ellis, OpenFaaS Ltd, MIT licence ([`LICENSE-rigmark`](LICENSE-rigmark)).

| Receipt | Setup |
|---|---|
| `glm53-libert-nvfp4-tp2-low.json` | vLLM TP2, DFlash2 k=7, reasoning low |
| `glm53-libert-nvfp4-tp2-adaptive-low.json` | vLLM TP2, adaptive DFlash2 k=3/5, reasoning low |
| `glm53-redhat-nvfp4-tp4-static-k7-low.json` | vLLM TP4 (four Sparks), DFlash2 k=7, reasoning low (for context; not a 2-Spark comparison) |
