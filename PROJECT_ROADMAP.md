# 🏠 German Rental Classification Project: Roadmap

**Dataset:** `immo_data.csv` (German rental listings, ImmoScout24)
**Task type:** Classification
**Decision attribute (target):** `rentClass`, where a flat is **Low / Medium / High** rent per m²
**Method:** CRISP-DM (Cross-Industry Standard Process for Data Mining)
**Final result:** Random Forest, **80.3% test accuracy** (goal ≥ 70%), far mistakes (Low ↔ High) only 0.47%

---

## How we work

1. **Explain:** what we do and *why*, in plain language (new concepts first, then code).
2. **Code:** the full code for each step, explained line by line.
3. **Run:** I run the code myself and read the output.
4. **Question:** I answer insight questions about the output.
5. **Conclude:** I write the conclusion in the notebook.
6. **Save:** at the end of each phase, commit and push to GitHub (on a feature branch).

> Rule: every claim must be backed by code I ran.

### Checklist for choosing a decision attribute
- [x] **Few nulls:** rows without a label can't be used for training → `rentClass` has 0 missing
- [x] **Balanced:** classes have similar sizes → 33.4 / 33.3 / 33.3%
- [x] **Meaningful:** predicting it answers a real question → "is this flat cheap, fair or expensive?"
- [x] **No leakage:** the input columns don't contain the answer → rent columns removed, check passed

### Project structure
| Notebook | Phases | Hands off |
|---|---|---|
| `01_data_preparation.ipynb` | 1–5 | `data/processed/immo_clean.csv` |
| `02_eda.ipynb` | 6 | insights |
| `03_modeling.ipynb` | 7–8 | `models/final_random_forest.joblib`, `models/test_set.joblib` |
| `04_evaluation.ipynb` | 9 | `data/processed/test_predictions.csv`, figures |

Reference guides: `doc/tuning_guide.md`

---

## Phase 0: Project setup
- [x] 0.1 Check that Python and Git are installed
- [x] 0.2 Create the project folder structure (`data/`, `notebooks/`, `reports/`…)
- [x] 0.3 Create a virtual environment
- [x] 0.4 Install packages and save them to `requirements.txt`
- [x] 0.5 Initialize Git and write `.gitignore` (raw data, processed data and models are too big for GitHub)
- [x] 0.6 Create the GitHub repository and push the first commit
- [x] 0.7 Create the notebook and connect it to the environment

## Phase 1: Business understanding
- [x] 1.1 Write a first draft of the problem: who would use this and why
- [x] 1.2 Define how the target will be built and what each class means
- [x] 1.3 Decide on success criteria *before* looking at results

## Phase 2: Data understanding (first contact)
- [x] 2.1 Load the data and check its size (rows × columns)
- [x] 2.2 Look at the first rows
- [x] 2.3 Check data types: is each column a number, text, True/False…?
- [x] 2.4 Build a **data dictionary**: type, missing %, number of unique values, example
- [x] 2.5 Classify each column as numeric, categorical, code, boolean, text, ID or date
- [x] 2.6 Check for duplicate rows and duplicate IDs, and for redundant column pairs
- [x] 2.7 Leakage check: list the columns that contain the price

## Phase 3: Data quality assessment
- [x] 3.1 Missing values per column: count and %
- [x] 3.2 Missing-value plan per column (drop / "unknown" / yes-no flag / median / model-based)
- [x] 3.3 Summary statistics: min, max, mean, median
- [x] 3.4 Find **impossible values** (rent = 0, size = 0 m², built in year 1000…)
- [x] 3.5 Find **outliers** with the IQR rule (decision: keep them, they are real flats)
- [x] 3.6 Check text and encoding problems (only in `street`, which is dropped)
- [x] 3.7 Check category columns for rare categories (rare districts grouped per state)

## Phase 4: Data cleaning
- [x] 4.1 Remove unneeded columns (25 of 49)
- [x] 4.2 Fix impossible values (remove rows for target ingredients; set NaN for other columns)
- [x] 4.3 Fill missing values with fixed rules ("unknown", yes/no flags); learned filling moved to Phase 7
- [x] 4.4 Tidy up categories (merge by meaning, group rare values)
- [x] 4.5 Write a **cleaning log** and save the clean data

## Phase 5: Building and checking the target
- [x] 5.1 Create `rentPerSqm` = baseRent ÷ livingSpace
- [x] 5.2 Look at its distribution with a histogram (right-skewed); remove impossible prices per m² (keep €2–40)
- [x] 5.3 Split into Low / Medium / High with tertiles (cut-offs €6.07 and €9.06)
- [x] 5.4 **Check class balance**: counts and % per class
- [x] 5.5 Check that the target has no missing values
- [x] 5.6 **Final leakage check**: features vs target correlation (none above 0.9)

## Phase 6: Exploratory data analysis (EDA)
- [x] 6.1 Yes/no features vs rent class (crosstabs)
- [x] 6.2 Category features vs rent class (crosstabs, spread, state chart)
- [x] 6.3 Number features vs rent class (medians, boxplot, size groups → U-shape)
- [x] 6.4 Correlation between number features (heatmap)
- [x] 6.5 Two features together, controlling for location (garden, new builds)
- [x] 6.6 Statistical tests: chi-square + Cramér's V
- [x] 6.7 Summary of insights
- [x] 6.8 Final problem statement and goal, backed by evidence

## Phase 7: Data preparation for modeling
- [x] 7.1 Choose the final input columns (X) and target (y)
- [x] 7.2 Split into train and test sets (80/20, stratified)
- [x] 7.3 Preprocessing pipeline: median filling + scaling, one-hot encoding, yes/no passthrough (23 → 220 columns)

## Phase 8: Modeling
- [x] 8.0 Validation set (20% of train) as a practice exam
- [x] 8.1 **Baseline**: always the most common class (33.4%)
- [x] 8.2 Decision Tree, with a depth sweep (best: depth 20, 72.4%)
- [x] 8.3 Random Forest (79.4%)
- [x] 8.4 More models from the course: Logistic Regression, k-NN, Naive Bayes, Bagging, Voting
- [x] 8.5 Cross-validation, 5 folds (Random Forest 79.6% ± 0.2, wins on every fold)
- [x] 8.6 Hyperparameter tuning (`min_samples_leaf`, `max_features`): defaults already near-optimal
- [x] 8.7 Median vs model-based filling: no real difference, keep the median
- [x] 8.8 Final model choice
- [x] 8.9 Train the final model on all training data and save it

## Phase 9: Evaluation
- [x] 9.1 Final test: accuracy, balanced accuracy, precision, recall and F1 for each class (80.3%)
- [x] 9.2 Confusion matrix: far mistakes (Low ↔ High) only 0.47%
- [x] 9.3 Feature importance: location dominates (~39%), then year built and living space
- [x] 9.4 Error analysis: most mistakes are borderline flats; far mistakes = possible deals / overpriced / data errors
- [x] 9.5 Check the result against the success criteria from Phase 1 (all met)

## Phase 10: Documentation and delivery
- [ ] 10.1 Clean up the notebooks with markdown explanations
- [x] 10.2 Final report: problem, data, method, results, insights, limitations (`reports/final_report.md` + `reports/final_report.docx`)
- [x] 10.3 A README for GitHub
- [ ] 10.4 Final commit and push
