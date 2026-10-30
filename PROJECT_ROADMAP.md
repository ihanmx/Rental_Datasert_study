# 🏠 German Rental Classification Project: Roadmap

**Dataset:** `immo_data.csv` (German rental listings, ImmoScout24)
**Task type:** Classification
**Decision attribute (target):** `rentClass`, where a flat is **Low / Medium / High** rent per m²
**Method:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

---

## How we work

1. **Explain:** what we do and *why*.
2. **Hint:** which function or command to use (not the full solution).
3. **Do:** I write and run the code myself.
4. **Question:** I answer insight questions about the output.
5. **Conclude:** I write the conclusion in the notebook.
6. **Save:** at the end of each phase, commit and push to GitHub.

> Rule: every claim must be backed by code I ran.

### Checklist for choosing a decision attribute
- [ ] **Few nulls:** rows without a label can't be used for training
- [ ] **Balanced:** classes have similar sizes (or imbalance is handled)
- [ ] **Meaningful:** predicting it answers a real question
- [ ] **No leakage:** the input columns don't contain the answer

---

## Phase 0: Project setup
- [ ] 0.1 Check that Python and Git are installed
- [ ] 0.2 Create the project folder structure (`data/`, `notebooks/`, `reports/`…)
- [ ] 0.3 Create a virtual environment
- [ ] 0.4 Install packages and save them to `requirements.txt`
- [ ] 0.5 Initialize Git and write `.gitignore` (the CSV is 270 MB, too big for GitHub)
- [ ] 0.6 Create the GitHub repository and push the first commit
- [ ] 0.7 Create the notebook and connect it to the environment

## Phase 1: Business understanding
- [ ] 1.1 Write a first draft of the problem: who would use this and why
- [ ] 1.2 Define how the target will be built and what each class means
- [ ] 1.3 Decide on success criteria *before* looking at results

## Phase 2: Data understanding (first contact)
- [ ] 2.1 Load the data and check its size (rows × columns)
- [ ] 2.2 Look at the first rows
- [ ] 2.3 Check data types: is each column a number, text, True/False…?
- [ ] 2.4 Build a **data dictionary**: what each column means, its type, and a first idea of whether it's useful
- [ ] 2.5 Classify each column as numeric, categorical, text, ID or date
- [ ] 2.6 Check for duplicate rows and duplicate IDs

## Phase 3: Data quality assessment
- [ ] 3.1 Missing values per column: count and %
- [ ] 3.2 Decide what to do with each column with missing values (drop, fill, or keep "missing" as its own category)
- [ ] 3.3 Summary statistics: min, max, mean, median
- [ ] 3.4 Find **impossible values** (rent = 0, size = 0 m², built in year 1000…)
- [ ] 3.5 Find **outliers** using boxplots and the IQR rule (interquartile range)
- [ ] 3.6 Check text and encoding problems (the strange characters in German names)
- [ ] 3.7 Check category columns for rare or odd categories

## Phase 4: Data cleaning
- [ ] 4.1 Remove useless columns (IDs, free text, house numbers…)
- [ ] 4.2 Remove impossible rows and decide on outliers, explaining each choice
- [ ] 4.3 Handle missing values
- [ ] 4.4 Fix data types
- [ ] 4.5 Write a **cleaning log**: rows before, rows after, and why

## Phase 5: Building and checking the target
- [ ] 5.1 Create `rentPerSqm` = baseRent ÷ livingSpace
- [ ] 5.2 Look at its distribution with a histogram
- [ ] 5.3 Choose where to split into Low / Medium / High, and justify the choice
- [ ] 5.4 **Check class balance**: counts and % per class, with a bar chart
- [ ] 5.5 Check that the target has no missing values
- [ ] 5.6 **Leakage check**: list the columns that must be removed from the inputs

## Phase 6: Exploratory data analysis (EDA)
- [ ] 6.1 **Univariate**: each column on its own (histograms, bar charts)
- [ ] 6.2 **Bivariate**: each column against the target (crosstabs, medians, boxplots)
- [ ] 6.3 **Correlation** between number columns (heatmap)
- [ ] 6.4 **Multivariate**: two columns together against the target
- [ ] 6.5 Statistical tests: is a difference real or just chance? (chi-square test)
- [ ] 6.6 Write an insight sentence for every finding
- [ ] 6.7 Final problem statement and goal, now backed by evidence

## Phase 7: Data preparation for modeling
- [ ] 7.1 Choose the final input columns (features)
- [ ] 7.2 Feature engineering: create useful new columns
- [ ] 7.3 Encode categories as numbers
- [ ] 7.4 Split into train and test sets, stratified so both keep the class balance
- [ ] 7.5 Scale numbers if the model needs it
- [ ] 7.6 Build a **pipeline** so the same steps run every time without mistakes

## Phase 8: Modeling
- [ ] 8.1 **Baseline**: a "dumb" model that always guesses the most common class. Every real model must beat it.
- [ ] 8.2 Decision Tree
- [ ] 8.3 Random Forest
- [ ] 8.4 One or two more (for example k-NN, Logistic Regression, Naive Bayes)
- [ ] 8.5 Cross-validation, which gives a more reliable score
- [ ] 8.6 Hyperparameter tuning

## Phase 9: Evaluation
- [ ] 9.1 Accuracy, precision, recall and F1 for each class
- [ ] 9.2 Confusion matrix: which classes get mixed up?
- [ ] 9.3 Compare the models in a table
- [ ] 9.4 Feature importance: what drives rent class?
- [ ] 9.5 Error analysis: look at the flats the model got wrong
- [ ] 9.6 Check the result against the success criteria from Phase 1

## Phase 10: Documentation and delivery
- [ ] 10.1 Clean up the notebook with markdown explanations
- [ ] 10.2 Final report: problem, data, method, results, insights, limitations
- [ ] 10.3 A README for GitHub
- [ ] 10.4 Final commit and push
