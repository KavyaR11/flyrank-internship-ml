[index.md](https://github.com/user-attachments/files/31577537/index.md)
# Engagement Risk Scoring: Finding Under-Engaging Content Before It Wastes Editorial Time

## Abstract

Content teams can't manually review every page for engagement problems, so they need a way to prioritize. This paper asks whether a machine learning model can rank content by engagement risk more usefully than a simple rule. Using FlyRank's anonymized `engagement_fix` lane data, I built a percentile-based engagement risk score, trained a Random Forest regressor against it, and validated the result against a Linear Regression baseline trained on identical features. The Random Forest achieved a Spearman rank correlation of 0.171 versus 0.028 for Linear Regression — a roughly 6x improvement from the same information, supporting the case that engagement risk depends on feature interactions a linear model cannot capture. The absolute correlation remains modest, and the paper is explicit about why: a noisy proxy target, a severely limited data slice, and a model blind spot on low-traffic content.

---

## 1. Question

**The question:** among a client's content, which pages are most likely under-engaging their readers, and can a ranked list of "engagement risk" replace a manual, page-by-page audit?

**Unit of analysis:** one piece of content, for one client, over a rolling window (30-day snapshot for the lane dataset; daily grain in the underlying warehouse).

**The decision this supports:** whether a content team should invest limited editorial time rewriting or restructuring a specific page, versus leaving it alone or addressing a different problem entirely (e.g. search visibility, technical issues).

**The action someone takes:** a content strategist works down a ranked list of flagged pages instead of manually auditing hundreds of pages with no sense of priority.

**The cost of a wrong call:** flagging a healthy page wastes writer and editor hours on a rewrite that won't move the needle. Missing a genuinely underperforming page lets it keep losing readers silently — and worse, a client may have already deprioritized it based on a bad recommendation, actively working against their own interests.

**Why this is a scoring problem, not classification:** a binary "needs fix / doesn't" flag treats all bad pages as equally bad. A continuous risk score lets a strategist prioritize by severity, which matters more than a threshold when editorial hours are the real constraint.

---

## 2. Data

**Source:** FlyRank's anonymized internship dataset — the `engagement_fix` subset of `FlyRank/internship-lanes` for lane development, cross-checked against the full `FlyRank/internship-warehouse` daily fact table (`fact_content_daily_performance`) to confirm grain and availability at scale.

**Grain:** one row = one piece of content, for one client (content_hash_id × client_hash_id), aggregated over a 30-day window in the lane dataset; content_hash_id × client_hash_id × report_date in the daily warehouse table. Verified unique via a `GROUP BY ... HAVING COUNT(*) > 1` query that returned zero rows.

**Date window:** development and validation were done on a single 30-day lane snapshot. For warehouse-level checks, a mid-panel month (`month = '2026-03'`) was used deliberately — the dataset's final month (June 2026, the `_sample` table) was treated as a sealed test window and never used to build label logic, per the program's data-use guidance.

**Scale and availability:** the March 2026 daily fact table contains 9,841,378 rows spanning the full calendar month. Of those, only 413,966 rows (~4.2%) have GA4 engagement data available at all (`ga4_data_available = TRUE`) — GSC and GA4 are independently toggled per client, so most rows in the warehouse simply cannot support an engagement label. Within the lane dataset itself, the GA4-available slice used for modeling contained 33,202 rows.

**What I excluded, and why:**
- Query-level keyword detail — engagement_fix is about reader behavior *after* landing on a page, not which query brought them there.
- GSC-only columns (impressions, clicks, average position) as inputs to the *label* — they measure search discovery, not post-landing engagement. They remain available as predictive features.
- Individual AI-referral source columns (ChatGPT, Perplexity, Gemini, etc.) and individual traffic-channel columns — collapsed into a single AI-traffic-share ratio rather than used at full granularity, since per-source detail adds noise without adding decision-relevant signal for this lane.

**Public-safety compliance:** all identifiers used throughout (content_hash_id, client_hash_id) are pre-anonymized hash keys. No client names, domains, URLs, or private queries appear anywhere in this analysis, its code, or its outputs.

---

## 3. Methodology

**Assumptions.** Two independently toggled data sources (GSC, GA4) mean any engagement-based label can only be computed on the subset of content where GA4 is connected — this constrains every downstream step and is treated as a hard limitation, not something to paper over.

**Target definition (the proxy).** The engagement label is a percentile-rank composite:

```
engagement_pct = rank(engagement_rate_30d, pct=True)
scroll_pct     = rank(scroll_rate_30d, pct=True)
risk_score     = 0.6 * (1 - engagement_pct) + 0.4 * (1 - scroll_pct)
```

Percentile ranking was used deliberately instead of a naive `1 - rate`, because the raw `engagement_rate_30d` field is not bounded between 0 and 1 in this data — a naive inversion would have produced invalid negative "risk" values for most rows.

This is explicitly a **proxy**, not ground truth. True engagement failure means a reader's intent wasn't satisfied by the page. A short session can mean "found the answer instantly and left satisfied" or "gave up looking and left frustrated" — GA4's engaged-session definition cannot distinguish these, and the risk score inherits that blur.

**Features.** All model features are either static content attributes (content_type, main_intent, age_tier, word_count_tier) or session-level counts (sessions_30d, impressions_30d, ctr_30d, avg_position_30d) — none are derived from the same-day outcome used to build the label.

**Baseline validity check (a deliberate leak, caught and removed).** Before modeling, I ran a leakage experiment: adding a same-day-outcome column disguised as a "feature" pushed a quick correlation check to ~1.0 — not because it was predictive, but because it *was* the label. It was removed, and the honest feature set (all trailing/static signals) was used going forward.

**Baseline selection (a second leak, caught during model comparison).** My first candidate baseline compared the Random Forest against FlyRank's pre-existing `needs_engagement_fix` flag. That flag turned out to be `True` for all 33,202 rows in the GA4-available slice — zero variance, undefined correlation — revealing that the `engagement_fix` lane is already a pre-filtered at-risk subset, not a general content population. I also confirmed that using components of the risk-score formula itself (`engagement_pct`, `scroll_pct`) as a "baseline" produced artificially strong correlations (up to −0.703) purely because they are 40–60% of how the label was built, not because they were genuinely predictive.

The methodologically sound comparison is therefore against **Linear Regression trained on the identical, independent feature set** — this isolates whether model complexity adds value, holding information constant.

**Split design.** Grouped by client_hash_id, 80/20 train/validation. The lane dataset has no date column, so a time-aware split was not possible; instead, the split ensures no client's pages appear in both train and validation, since pages from the same client likely share structural or audience traits that could let a model "cheat" by recognizing the client rather than learning generalizable content signal. Zero client overlap was confirmed directly.

**Model.** Random Forest Regressor (200 trees, max depth 8), chosen over Gradient Boosting because the target is a noisy proxy rather than clean ground truth, and Random Forest's averaging is more robust to overfitting on a signal this weak — while also providing permutation importance for error analysis.

**Success metric.** Spearman rank correlation between predicted and actual risk score on the held-out validation set. Ranking quality was prioritized over exact-value accuracy, since the decision this supports is a prioritized list a strategist works down, not a precise numeric forecast.

---

## 4. Results (vs. baseline)

| Method | Spearman correlation (validation) |
|---|---|
| Linear Regression (same features) | 0.028 |
| **Random Forest** | **0.171** |

The Random Forest achieved roughly a **6x improvement** over Linear Regression using the exact same inputs. Because the improvement comes from identical information, the gain is attributable to the model's ability to capture non-linear interactions between features (e.g. what counts as "normal" session volume may vary by content type or word count) — not from having more data available to it.

0.171 is a modest correlation in absolute terms. The model produces some real, usable ranking signal beyond noise, but is not strong enough on its own to fully replace human review of flagged content.

---

## 5. Limitations

- **Severely restricted usable data.** Only ~4.2% of the daily warehouse rows for a representative month have GA4 data available at all. Any engagement-based analysis, including this one, only ever sees a small, GA4-connected slice of FlyRank's full content population — clients without GA4 connected are invisible to this lane entirely, not just weakly measured.

- **No content metadata in the daily warehouse table.** The raw daily fact table used for warehouse-scale verification contains only behavioral measures and hash keys — no content_type, age_tier, word_count, or intent columns. Any content-context signal used in modeling came from the separately maintained lane dataset, not the daily warehouse directly.

- **A systematic model blind spot.** The ten largest validation errors were all "keyword article" content with very low session counts (10–16 sessions), where the model predicted high risk but actual risk was low. Low-traffic pages make the underlying rate metrics unstable, and the model appears to latch onto that instability rather than a genuine pattern.

- **Some features contribute no measurable value.** Permutation importance showed main_intent, age_tier, and content_type all had *negative* importance — shuffling them slightly improved validation performance, meaning the model isn't meaningfully using these fields despite their intuitive relevance. This likely reflects sparse per-category sample sizes rather than a genuine absence of signal.

- **The target is a proxy, not ground truth**, and the correlation achieved is modest. This work can claim observed, directional, decision-support value — a prioritized list a human reviews — and cannot claim causal proof that any specific trait *causes* low engagement, nor can it predict Google's ranking behavior or guarantee that fixing flagged pages will move business outcomes.

---

## 6. Ranked recommendations

Based on the validated model, the recommended action playbook is:

1. **Use the model's ranking as a triage tool, not an automated verdict.** Given the modest absolute correlation, treat the top of the ranked list as "review first," not "act on automatically."
2. **Discount or manually verify flags on very low-traffic content.** The identified blind spot means pages with under ~20 sessions in the window should be flagged for human judgment rather than trusted at face value.
3. **Prioritize investigating impressions, word count, and CTR as first-line signals** when triaging a flagged page, since these carried the most real signal in the model.
4. **Treat GA4 connection status as a prerequisite for this workflow entirely** — expanding GA4 coverage across more clients would directly expand how much of the content population this approach can serve.
5. **Revisit the proxy definition in future iterations.** Because engagement_rate alone conflates "quick and satisfied" with "frustrated and gone," a future version should explore supplementary signals (e.g. return-visit rate, scroll depth combined with dwell time) to sharpen the target before scaling this approach.

---

## 7. Artifacts

- Feature importance chart (`work/outputs/feature_importance.png`)
- Model vs. baseline comparison table (Section 4, above)
- Worst-10-prediction error table, used to identify the low-traffic blind spot (generated in `work/notebooks/w05_model.ipynb`)
- Week 4 baseline ranked queue (`work/outputs/baseline_action_score.csv`, regenerated on each notebook run per the project's data-hygiene rules — not committed as a static file)

---

## Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset. Learn more at [flyrank.ai](https://flyrank.ai).
