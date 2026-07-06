# 3-Minute Demo Video Script — VAJRA (Kazuki's 4-step structure, two domains)

Record 1080p. Open on the **landing page** so the "one engine, two lifelines" thesis lands
first. Spend ~2/3 on GRID (the deeper pack) and ~1/3 on AGRO to prove generality. Dry-run first.

---

**0:00–0:20 — THE INSIGHT** *(landing page /)*
> "Grid load-shedding and drought response look unrelated — but they're the same problem:
> under climate stress, allocate a scarce resource across many units, each with a risk score
> and a fairness history, faster than conditions change. VAJRA builds that once, on GPU, and
> ships two decision rooms."

**0:20–0:50 — GRID: user + pipeline** *(click ⚡ GRID)*
> "A DISCOM control room at 46 °C. This engineer must shed megawatts without touching hospitals,
> before transformers burn out — but her AMI pipeline runs overnight on CPU, so she's a day
> behind. VAJRA cleans 100M messy smart-meter readings and scores every feeder on an NVIDIA L4
> with RAPIDS cudf.pandas — same code as pandas, one flag."
*(hover the pipeline stage bar + data-quality chips)*

**0:50–1:25 — GRID: acceleration + decision** *(Acceleration tab → What-If)*
> "Same pipeline, both engines, measured: {X}s on CPU, {Y}s on GPU — {Z}× — and it now fits
> inside one 15-minute dispatch block, so she decides on current data. The grid needs 40 MW of
> relief: VAJRA just evaluated {N} thousand shed plans in {t} seconds, hospitals excluded,
> fairness enforced — Plan A, lowest customer pain. By hand that's {h} hours."

**1:25–2:35 — AGRO: the same engine, a different lifeline** *(click 🌾 AGRO)*
> "Now the exact same architecture, pointed at food security. Monsoon dry spell; a district
> officer covering hundreds of thousands of plots. VAJRA gap-fills cloud-broken Sentinel-2 NDVI
> — the heavy per-plot groupby that GPUs crush — forecasts yield, and ranks villages by advisory
> priority with explainable reason codes: dry-spell-in-reproductive-stage, NDVI decline,
> insurance trigger."
*(Insurance Triggers tab)* > "These villages crossed a PMFBY-style trigger — payouts can start
> on this week's data, not a season-end assessment."
*(Allocate Water tab → 40 ML)* > "And with only 40 megalitres of canal water this week, VAJRA
> evaluated {N} thousand allocations in {t}s to save the most yield — deliberately favouring
> chronically under-served tail-reach villages."

**2:35–3:00 — CLOSE** *(Copilot in either domain)*
> "In plain language too — grounded only in live pipeline tools, no invented numbers. One Google
> Cloud + NVIDIA pipeline; two climate lifelines; a faster pipeline becomes a fairer, cheaper,
> better decision. Link and repo below."

---

**Pre-record checklist:** both demo packs loaded (no empty panels on /grid AND /agro) ·
bench/results.json has GPU rows for both domains · What-If pre-tested at 40 MW · Allocate
pre-tested at 40 ML · Copilot works (fallback fine) · zoom 100% · notifications off · < 3:00 hard.
