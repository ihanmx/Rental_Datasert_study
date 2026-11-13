# 🔧 Tuning Guide: Classifiers from the Course

A reference for tuning each classifier used in this project. Every example uses the helper functions from `notebooks/03_modeling.ipynb`:
- `make_model(model, dense=False)` builds the pipeline (preparation + model)
- `evaluate(pipe, name, sample=None)` trains on the train part and prints train / validation / gap

---

## Key ideas

| Term | Meaning |
|---|---|
| **Parameters** | What the model **learns** from the data (e.g. which questions a tree asks) |
| **Hyperparameters** | Settings **we choose before training** (e.g. tree depth, number of neighbours) |
| **Tuning** | Trying a few hyperparameter values and keeping the one with the **best validation score** |

**Analogy:** the model is a cake recipe; the hyperparameters are the oven temperature and baking time. You bake test cakes at different settings, taste them (validation), and keep the best setting.

### The general recipe (any model)
1. Pick the **1–2 knobs** that matter most for that model.
2. Try a few values for **one knob at a time**, scoring on **validation** (never on the test set).
3. Lock the best value, then move to the next knob.
4. Watch the **gap** (train − validation): both scores low = underfitting; big gap = overfitting.
5. If two settings score almost the same (< ~0.5 points apart), choose the **simpler** one (Occam's razor).

---

## 1. Decision Tree

| Knob | What it controls | Small value | Large value |
|---|---|---|---|
| **`max_depth`** | How many questions deep the tree can go | Underfitting | Overfitting |
| **`min_samples_leaf`** | Minimum flats per leaf | Overfitting | Underfitting |
| `criterion` | How the best split is measured: `"gini"` or `"entropy"` | Usually tiny differences | |

```python
for depth in [10, 15, 20, 25]:
    for leaf in [1, 5, 20]:
        tree = make_model(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=42))
        evaluate(tree, f"Tree depth={depth}, leaf={leaf}");
```
Trees are fast, so a small **combination** grid (here 12) is fine.

---

## 2. k-Nearest Neighbours (k-NN)

| Knob | What it controls | Small value | Large value |
|---|---|---|---|
| **`n_neighbors`** (k) | How many similar flats vote | Overfitting (one odd neighbour decides) | Underfitting (votes become too general) |
| **`weights`** | `"uniform"`: all neighbours count equally · `"distance"`: closer neighbours count more | | |

```python
for k in [5, 15, 30, 50]:
    for w in ["uniform", "distance"]:
        knn = make_model(KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1))
        evaluate(knn, f"k-NN k={k}, weights={w}", sample=30000);
```
- Keep `sample=30000`, because k-NN is slow on large data.
- With 3 classes, ties are possible; `weights="distance"` also helps break them.
- k-NN needs **scaled** numbers (distance-based), which the pipeline already does.

---

## 3. Logistic Regression

| Knob | What it controls | Small value | Large value |
|---|---|---|---|
| **`C`** | Inverse of **regularization**: how much the model may trust the training data | Strong penalty → simple model → underfitting | Weak penalty → complex model → overfitting |

**Regularization, explained simply:** a penalty that stops the model from giving **extreme weights** to any single feature (e.g. a huge weight to a rare district with only a few flats). Smaller `C` = stronger "keep it simple" rule.

```python
for c in [0.01, 0.1, 1, 10]:
    logreg = make_model(LogisticRegression(C=c, max_iter=1000))
    evaluate(logreg, f"LogReg C={c}");
```
Values usually jump by **factors of 10**, because the effect is gradual.

---

## 4. Naive Bayes (GaussianNB)

| Knob | What it controls |
|---|---|
| **`var_smoothing`** | Adds a small safety margin to each feature's spread, avoiding extreme probabilities from tiny groups |

```python
for vs in [1e-9, 1e-6, 1e-3, 1e-1]:
    nb = make_model(GaussianNB(var_smoothing=vs), dense=True)
    evaluate(nb, f"Naive Bayes var_smoothing={vs}");
```
- `1e-9` = 0.000000001 (the default).
- Needs `dense=True` (GaussianNB cannot read sparse tables).
- Naive Bayes has very few knobs, so tuning rarely fixes its main weakness: the **independence assumption**.
- The course's `CategoricalNB` is for category-only data; `GaussianNB` is for number features.

---

## 5. Bagging

| Knob | What it controls |
|---|---|
| **`n_estimators`** | How many trees (more = steadier vote, slower) |
| **`max_samples`** | Fraction of flats in each tree's bootstrap sample (1.0 = all, 0.5 = half). Smaller → **more different** trees |
| Base tree's knobs | e.g. `max_depth` of the copied tree |

```python
for ms in [0.3, 0.6, 1.0]:
    bag = make_model(BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50, max_samples=ms, n_jobs=-1, random_state=42))
    evaluate(bag, f"Bagging max_samples={ms}");
```
**Note:** bagging helps **unstable** models (deep trees). Stable models (Logistic Regression, Naive Bayes, k-NN with a large k) gain little from it.

---

## 6. Voting

| Knob | What it controls |
|---|---|
| **`voting`** | `"hard"`: one model, one vote · `"soft"`: average the models' probabilities (confident models count more) |
| **`weights`** | Give better models more votes, e.g. `[2, 1, 1]` |
| **Which voters** | Remove weak voters (e.g. Naive Bayes) or add strong ones |

```python
for voting_type, w in [("hard", None), ("soft", None), ("soft", [2, 1, 1])]:
    vote = make_model(VotingClassifier(
        estimators=[("logreg", LogisticRegression(max_iter=1000)),
                    ("tree",   DecisionTreeClassifier(max_depth=20, random_state=42)),
                    ("nb",     GaussianNB())],
        voting=voting_type, weights=w, n_jobs=-1), dense=True)
    evaluate(vote, f"Voting {voting_type}, weights={w}");
```
`[2, 1, 1]` = Logistic Regression's vote counts double. A vote is only as good as its voters: one weak voter can pull the result down.

---

## 7. Random Forest

| Knob | What it controls |
|---|---|
| **`min_samples_leaf`** | Minimum flats per leaf (larger = less memorizing) |
| **`max_features`** | How many features each split may consider: `"sqrt"` (default, ≈15 of 220), `0.1`, `0.2` … Fewer = more diverse trees; more = stronger but more similar trees |
| `n_estimators` | Number of trees (more = steadier, slower; gains level off after ~100) |

```python
for leaf in [1, 3, 5, 10]:
    rf = make_model(RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, n_jobs=-1, random_state=42))
    evaluate(rf, f"RF leaf={leaf}");
```
- A 100% train score is **normal** for forests; judge them by **validation**.
- Random Forest = Bagging with trees + random feature subsets at each split.

---

## The automatic way: `GridSearchCV`

Tries **every combination** of the given values, each scored with **cross-validation**, and reports the best.

```python
from sklearn.model_selection import GridSearchCV

pipe = make_model(LogisticRegression(max_iter=1000))

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],     # "model__" = the step named "model" inside the pipeline
}

search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)

print("Best setting:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_ * 100:.1f}%")
```

| Code | Meaning |
|---|---|
| `param_grid = {...}` | Which knobs to try, and which values |
| **`"model__C"`** | The model sits inside a pipeline under the name `"model"`; the **double underscore** means "knob `C` of step `model`". For a forest: `"model__min_samples_leaf"` |
| `cv=3` | Each combination scored with 3-fold cross-validation |
| `search.best_params_` | The winning knob values |
| `search.best_score_` | Their average CV score (0–1, so × 100 for %) |

### Manual loops vs GridSearchCV

| | Manual loop + validation | GridSearchCV |
|---|---|---|
| Speed | ✅ Fast (each setting trained once) | ❌ Slow (combinations × folds) |
| Reliability | One practice exam | ✅ Several (cross-validation) |
| Best for | **Large data / slow models** (e.g. Random Forest on ~171k flats) | **Small data / fast models** (Logistic Regression, Naive Bayes, single trees) |

**Cost example:** 4 values × 3 folds = 12 trainings. A forest with 4 × 3 settings × 3 folds = **36 forests**, which takes hours on large data.

---

## Summary: the main knobs per model

| Model | Main knob(s) | Reduces overfitting by… |
|---|---|---|
| Decision Tree | `max_depth`, `min_samples_leaf` | Limiting how detailed the tree gets |
| Random Forest | `min_samples_leaf`, `max_features` | Bigger leaves; more diverse trees |
| k-NN | `n_neighbors`, `weights` | A larger k |
| Logistic Regression | `C` | A smaller C (stronger regularization) |
| Naive Bayes | `var_smoothing` | Smoothing the probabilities |
| Bagging | `n_estimators`, `max_samples` | Smaller samples → more diverse trees |
| Voting | `voting`, `weights`, voter choice | Trusting the better models more |

💡 In practice, tune only the **best 1–2 candidate models**, not every model.
