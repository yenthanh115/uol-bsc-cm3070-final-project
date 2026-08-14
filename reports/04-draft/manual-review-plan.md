# Manual Review Plan — Draft Report v0.8.14

Quy trình review thủ công từng đoạn văn, từng section, và tổng thể. Mỗi bước có checklist cụ thể để đánh dấu khi hoàn thành.

---

## Quy Tắc Chung

Khi đọc mỗi đoạn văn, tự hỏi 5 câu:
1. **Câu đầu tiên** có nêu rõ ý chính của đoạn không?
2. **Mỗi câu tiếp theo** có build on câu trước một cách tự nhiên không?
3. **Câu cuối** có transition sang đoạn kế hoặc kết luận ý không?
4. **Người đọc** (marker không chuyên ML sâu) có hiểu được không cần đọc lại?
5. **Tone** có nhất quán — academic nhưng không khô khan?

---

## Phase A: Section-by-Section Deep Review

### A1. Abstract (~269 words) ✅ REVIEWED

**Mục tiêu:** Tóm tắt toàn bộ project trong 1 đoạn. Reader phải biết: vấn đề gì, làm gì, kết quả ra sao.

- [x] Câu mở đầu nêu problem rõ ràng? ✓ (surges + largely unaddressed)
- [x] Methodology tóm tắt đủ? ✓ (composite metric, observation-time features, 3 classifiers, temporal CV)
- [x] Kết quả chính xác? ✓ XGBoost 0.892 (WSB), RF 0.753 (pennystocks) — khớp Table 11
- [x] Cross-transfer AUC = 0.684 khớp Table 16? ✓
- [x] Conclusion sentence nêu contribution chính? ✓ (3 findings: predict from past, density > complexity, framework transferable)
- [x] Không có jargon chưa define? ✓ (đã bỏ "stretch/target tier")
- [x] Đọc to lên — nghe tự nhiên? ✓ (đã fix flow qua nhiều lần edit)
- [x] HTML comments đã xóa hết? ✓

**Issues đã fix trong quá trình review:**
- "platoformss" → N/A (không ở Abstract)
- Old numbers (0.861/0.854/0.746) → updated to 0.892/0.753
- Cross-transfer 0.694 → 0.684
- "stretch performance tier" jargon → removed
- "mainly asked a question" → "The central question is:"
- "sinple" → "simple"
- "RandomForest" → "Random Forest"
- RF pennystocks 0.734 (sai) → 0.753 (đúng)
- "a great fit" → rewritten to academic register
- Run-on sentence ("that suggests") → ", suggesting that"
- Fragment cuối → merged into complete sentence
- "at time?" → "at observation time?"
- Bold trên "machine learning" → removed
- "our findings" voice → acceptable (used once)

---

### A2. Section 1 — Introduction (~860 words, limit 1000)

**Đọc từng paragraph:**

#### §1.1 Project Concept and Objectives
- [x] Paragraph 1: Nêu template gốc + instantiation → rõ ràng?
- [x] Fix typo "platoformss" → "platforms"
- [x] 3 objectives: mỗi bullet có action verb + measurable outcome?
- [x] Hypothesis paragraph: "scales with data density rather than model complexity" — tự giải thích được không? (comment cũ nói "do not understand")
- [x] Xóa comment `<!--do not understand meaning of the later point -->` (cuối cùng)

#### §1.2 Problem Statement and Motivation
- [x] Para 1: Problem → vivid example (2 posts → 50 posts)
- [x] Para 2: Research question + practical relevance → tốt?
- [x] Para 3: Literature gap (forward reference to 2.6) — reader có bị lost không?
- [x] Para 4: Why ML not simple threshold → justifies approach?

#### §1.3 Prediction Scope and Surge Definition
- [x] Distinguishes "trend" vs "surge" clearly?
- [x] Explains composite metric motivation (volume alone misses agitation)?
- [x] Forward ref to Section 3.3 — đúng?
- [x] HTML comments nếu còn → xóa cuối cùng

#### §1.4 Scope
- [x] In/out scope concise?
- [x] Numbers (80,212 / 1,293,981) khớp Table 4?

#### §1.5 Report Structure
- [x] Preview mỗi section → matches actual content?

---

### A3. Section 2 — Literature Review (~1974 words, limit 2500)

**Đọc từng paragraph:**

#### §2.1 Predictability of Online Attention
- [x] Para 1: Scoping — giải thích tại sao chỉ review 17 sources, không review time-series / graph methods?
- [x] Para 2: Szabo & Huberman — nêu contribution + limitation?
- [x] Para 3: Lerman & Hogg, Wang & Huberman, Kong et al. — builds chronologically?
- [x] Para 4: Consensus summary + tension (predictability plateaus) → positions this project?
- [x] Mermaid diagram: Figure 1 — labelled? Referenced trước khi xuất hiện?

#### §2.2 Shift Toward Pre-Engagement Prediction
- [x] Para 1: Bandari → prediction before engagement
- [x] Para 2: Cheng → rate of early spread
- [x] Para 3: Second consensus + identifies remaining gap (eventual vs onset)?
- [x] Table 1: Headers clear? Data accurate? Adds value vs prose?

#### §2.3 Sentiment as Predictive Signal
- [x] Para 1: Bollen — mood predicts Dow Jones
- [x] Para 2: Tool evolution (OpinionFinder → VADER → FinBERT)
- [x] Para 3: Takeaway for this project (sentiment *change* as leading indicator)
- [x] Table 2: Adds value? Column headers clear?

#### §2.4 Financial Discussion on Reddit
- [x] Para 1: Why penny stocks are special (low liquidity, high social influence)
- [x] Para 2: Long et al. — Reddit → trading volume
- [x] Para 3: Costola et al. — consensus formation
- [x] Para 4: Mancini et al. — pump-and-dump
- [x] Para 5: Synthesis — connects back to earlier work + identifies gap?

#### §2.5 Methodological Weaknesses
- [x] Para 1: Pattern identified (no temporal evaluation)
- [x] Para 2: Specific evidence per study
- [x] Para 3: Tashman + Bergmeir → solution exists but unadopted
- [x] Table 3: Clear? Accurate? Referenced?

#### §2.6 Research Gap and Project Position
- [x] 4 cumulative findings — each maps to prior subsections?
- [x] 4 gaps — each clearly distinct?
- [x] Final paragraph: project addresses each gap → maps to later sections?
- [x] Transition to Section 3 smooth?

#### Cuối Section 2:
- [x] **XÓA block comment lớn** (lines 62–100): "How to write a lit review" scaffolding

---

### A4. Section 3 — Design (~2349 words, limit 2000) ⚠️ OVER

**Critical: Cần cắt ~350 words. Target: Section 3.5**

#### §3.1 System Architecture
- [x] 6-stage pipeline clear?
- [x] Mermaid diagram matches text?
- [x] Overarching constraint (no future info) stated?

#### §3.2 Data Selection and Characteristics
- [x] Why Reddit? (subreddit structure, public archive, timestamps)
- [x] Why these 2 subreddits? (density spectrum)
- [x] Table 4 numbers match elsewhere?
- [x] Ethical considerations addressed?
- [x] Known limitations acknowledged?

#### §3.3 Surge Definition (Target Variable)
- [x] Formula clear? (composite = w₁×z_vol + w₂×z_sent)
- [x] 5 computation steps logical?
- [x] Why 24-hour window? (trading cycle argument)
- [x] Why include sentiment? (empirical evidence cited)
- [x] Table 5: threshold sensitivity — clear?
- [x] Two-phase validation explained?

#### §3.4 Feature Engineering
- [x] 11 features listed — Table 6 clear?
- [x] Backward-looking constraint explicit?
- [x] Why exclude `score` and `num_comments`?
- [x] Category rationale (content, temporal, activity, interaction)?

#### §3.5 Methodological Scope ← **CẮT Ở ĐÂY**
- [x] "Adopted techniques" — concise?
- [x] "Excluded techniques" — **quá dài**. Target:
  - ARIMA/LSTM: cắt từ ~120 words → 50 words
  - Network analysis: cắt từ ~100 words → 40 words
  - Topic modelling: cắt từ ~80 words → 30 words
- [x] Guiding principle paragraph — keep

#### §3.6 Model Selection
- [x] Why binary classification? (brief, effective justification)
- [x] 3 models span complexity spectrum?
- [x] Primary metric (AUC-ROC) justified?
- [x] Imbalance handling (cost-sensitive, no SMOTE) justified?

#### §3.7 Temporal Validation Design
- [x] 2-level strategy clear?
- [x] Mermaid diagram (expanding window) correct?
- [x] Why k=4 explained?
- [x] Threshold tuning never touches test data — stated?

#### §3.8 Evaluation Framework
- [x] 4 questions listed?
- [x] Table 7a (tiers) clear?
- [x] Baselines specified?
- [x] Statistical tests specified?
- [x] Table 7b (metrics) clear?
- [x] Cross-dataset transfer protocol?
- [x] Sensitivity analysis specified?

#### Cuối Section 3:
- [x] **XÓA "SIDE NOTE (DELETE LATER)" block** (lines 227–240)
- [x] **XÓA "How to write methodology" block** (lines 241–278)

---

### A5. Section 4 — Implementation (~1297 words, limit 2000)

#### §4.1 Code Organisation
- [ ] Directory tree clear?
- [ ] Module → stage mapping explained?
- [ ] Table 8 (CLI) useful?

#### §4.2 Data Loading and Preprocessing
- [ ] 4 steps (clean, extract, normalise, explode) clear?
- [ ] Ticker extraction logic explained (2 regex patterns, stopword set)?
- [ ] Table 9 (attrition) numbers match Table 4?

#### §4.3 Feature Engineering
- [ ] Table 11 matches Table 6 in Section 3.4?
- [ ] Code snippet (acceleration) clear and correct?
- [ ] Interaction features justified (ablation B2 reference)?

#### §4.4 Surge Labelling
- [ ] Temporal split mechanism clear?
- [ ] Z-score freezing explained?
- [ ] Composite formula repeated (ok for reader)?
- [ ] Exclusion logic explained (< 2 posts in forward window)?

#### §4.5 Model Training and Evaluation
- [ ] Expanding-window CV implementation matches design (Section 3.7)?
- [ ] Table 10 (hyperparameter grids) reasonable?
- [ ] StandardScaler fresh per fold — stated?
- [ ] Figure 4 (ROC curves) referenced before appearing?
- [ ] Figure 5 (feature importance) referenced before appearing?

#### §4.6 Implementation Decisions
- [ ] Each decision driven by empirical finding?
- [ ] Timestamp mismatch → verification checks?
- [ ] Data sparsity → second dataset?
- [ ] Threshold collapse → τ=1.0 + scale_pos_weight?

#### §4.7 Implementation Status
- [ ] Table shows all 6 stages complete?
- [ ] Runtime reported?
- [ ] Tests/linting pass mentioned?

---

### A6. Section 5 — Evaluation (~1851 words, limit 2500)

#### §5.1 Evaluation Against Objectives
- [ ] §5.1.1: Objective 1 answered? Numbers match Table 11?
- [ ] §5.1.2: Objective 2 answered? "18.5-point gap" correct? (✓ fixed)
- [ ] §5.1.3: Objective 3 answered? Temporal validity demonstrated?
- [ ] §5.1.4: Reproducibility — seed sensitivity range (0.734–0.753) match log?

#### §5.2 Results
- [ ] Table 11: All numbers verified against experiment log? ✓
- [ ] XGB pennystocks "Target†" footnote — explained?
- [ ] Table 12 (tuned thresholds) matches threshold_tuning.json? ✓
- [ ] Table 13 (confusion matrix) verified? ✓
- [ ] §5.2.2 Table 14 (McNemar) — χ² values match final_summary.json? ✓
- [ ] Table 15 (baselines) — single feature AUCs match? ✓
- [ ] §5.2.3 Table 16 (cross-transfer) verified? ✓
- [ ] §5.2.4 Table 17 (feature importance) — source? (check feature_importance.json)
- [ ] §5.2.5 Table 18, 19 (sentiment/weight) — verified? ✓

#### §5.3 Critical Analysis
- [ ] §5.3.1: RF > XGB on pennystocks explained (3 factors)?
- [ ] §5.3.2: Sentiment paradox (weak alone, strong in context) resolved?
- [ ] §5.3.3: Transfer asymmetry explained (distributional mismatch)?
- [ ] §5.3.4: Precision concern (4/5 false alarms) contextualised? Temporal stability discussed?

#### §5.4 Limitations and Proposed Improvements
- [ ] §5.4.1: 5 limitations honest? Sentiment circularity addressed?
- [ ] §5.4.2: Table (improvements) — effort/priority reasonable?

#### §5.5 Originality and Contribution
- [ ] 3 contributions clearly stated?
- [ ] Figure references correct? (🟡 "Figure 3" vs "Figure 4" numbering issue)
- [ ] Comparison with published baselines appropriately hedged?

---

### A7. Section 6 — Conclusion (~1539 words, limit 1000) ⚠️ OVER

**Critical: Cần cắt ~540 words.**

#### Strategy to cut:
- [ ] §6.1 (Current Achievements): 2 paragraphs → compress to 1 (cut ~100 words)
- [ ] §6.2 (Key Findings): 3 detailed findings → summarise each in 2–3 sentences instead of full paragraphs. Reader already knows from Section 5. Point back to Section 5 for detail. (cut ~300 words)
- [ ] Cross-transfer explanation: 1 sentence summary, not full re-explanation (cut ~80 words)
- [ ] §6.3 (Limitations/Future): Keep but tighten prose (cut ~60 words)

#### Content check:
- [ ] §6.1: Answers the research question directly?
- [ ] §6.2: Findings are non-obvious insights, not just number recitation?
- [ ] §6.2: Tone — tighten "What came out of that effort..." to academic register
- [ ] §6.3: Limitations distinct from Section 5.4.1 (higher-level)?
- [ ] §6.3: Future work actionable and specific?
- [ ] Final paragraph: Strong closing sentence?

---

### A8. References

- [ ] 17 references — all cited in text?
- [ ] Citation format consistent (ACM style)?
- [ ] All [1]–[17] have matching entries?
- [ ] No broken URLs?
- [ ] Dates/venues accurate?

---

## Phase B: Cross-Section Consistency

Sau khi review từng section, kiểm tra toàn bộ:

### B1. Numeric Consistency
- [ ] Mỗi con số xuất hiện ở 2+ nơi — khớp nhau?
- [ ] Table numbers reference the same experimental config?
- [ ] WSB test surges: 668 (Table 11) vs 2,582 (Table 4) — different thresholds explained?

### B2. Figure/Table Numbering
- [ ] Figures numbered sequentially (1, 2, 3, 4, 5...)?
- [ ] Tables numbered sequentially?
- [ ] No duplicate numbers?
- [ ] All referenced before/when they appear?

### B3. Terminology Consistency
- [ ] "surge" vs "trend" — used correctly throughout?
- [ ] "backward-looking" vs "observation-time" — consistent?
- [ ] Model names: "XGBoost" vs "XGB" vs "xgboost" — when to use which?
- [ ] "stretch tier" / "target tier" — always with threshold definition nearby?

### B4. Forward/Backward References
- [ ] Every "Section X.Y" reference points to existing content?
- [ ] Every "Table N" reference points to existing table?
- [ ] Every "Figure N" reference points to existing figure?

### B5. Narrative Thread
- [ ] Introduction promises → Design delivers → Implementation builds → Evaluation proves?
- [ ] Research question posed in §1 → answered in §6?
- [ ] Gaps identified in §2.6 → addressed in §3 → results in §5?

---

## Phase C: Overall Quality Checks

### C1. Word Count Compliance
- [ ] Count body text only (exclude tables, figures, code, references, captions)
- [ ] Introduction ≤ 1000 ✓
- [ ] Literature Review ≤ 2500 ✓
- [ ] Design ≤ 2000 ← need cuts (~350)
- [ ] Implementation ≤ 2000 ✓
- [ ] Evaluation ≤ 2500 ✓
- [ ] Conclusion ≤ 1000 ← need cuts (~540)
- [ ] Total ≤ 9500 ← need cuts (~370 after section cuts)

### C2. Cleanup Checklist
- [ ] ALL HTML comments removed
- [ ] No draft scaffolding remaining
- [ ] No "TODO" or "DELETE LATER" notes
- [ ] No placeholder text
- [ ] No orphan figures (referenced but file missing)
- [ ] Spelling check (especially "platoformss")

### C3. Rubric Self-Assessment
Score yourself honestly against each rubric criterion:

| Criterion | Self-Score | Evidence | Action Needed? |
|-----------|-----------|----------|----------------|
| Clearly written (0–6) | | | |
| Diagrams appropriate (0–4) | | | |
| Prototype quality (0–4) | | | |
| Technically challenging (0–3) | | | |
| Evaluation quality (0–3) | | | |
| Code clarity (0–4) | | | |
| Code quality (0–6) | | | |

### C4. Read-Aloud Test
- [ ] Print or read on tablet (different medium from writing)
- [ ] Read Abstract aloud — flows naturally?
- [ ] Read Conclusion aloud — satisfying ending?
- [ ] Any sentence you have to re-read twice → rewrite it

### C5. Fresh-Eyes Test
- [ ] Give to someone who hasn't read it
- [ ] Can they explain what you did after reading Abstract + Conclusion only?
- [ ] Do they get confused at any point?

---

## Execution Order

Thực hiện theo thứ tự này — edit nội dung trước, cleanup/cuts cuối cùng:

1. **Section-by-section** (A1–A8) — đọc kỹ, edit nội dung, mark issues → 2–3 hours
2. **Cross-section consistency** (B1–B5) — after all sections stable → 30 min
3. **Overall quality** (C3, C4) — rubric self-score, read-aloud → 30 min
4. **Word count cuts** (A4 §3.5 + A7) — cắt sau khi nội dung đã ổn, biết chính xác cần cắt gì/giữ gì → 45 min
5. **Final cleanup** (C1, C2) — xóa comments, fix typos, verify word count → 15 min
6. **Fresh-eyes test** (C5) — if time permits → next day

**Lý do đặt cuts + cleanup cuối:** Trong quá trình edit section-by-section, nội dung có thể thay đổi — thêm/bớt ý, rewrite đoạn văn, restructure arguments. Cắt word count quá sớm có thể phải cắt lại hoặc cắt nhầm phần sau đó được viết lại. Cleanup cuối cùng đảm bảo không sót gì sau tất cả edits.

**Estimated total: 4–5 hours** (split over 2 sessions recommended)
