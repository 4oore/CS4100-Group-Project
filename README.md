# CS4100-Group-Project
## Meal Planning System (Simulated Annealing + Genetic Algorithm)
=================================================================

This project implements a **daily-updating meal planning system** that dynamically generates meal recommendations based on what the user has already eaten.

The system uses a **rolling-horizon optimization framework**, where:
- Meals already consumed are treated as fixed
- Remaining meals are optimized to meet nutritional goals

Two optimization methods are supported:
- **Simulated Annealing (SA)** – local search with probabilistic exploration
- **Genetic Algorithm (GA)** – population-based global search

A **Gaussian Naive Bayes classifier** (implemented from scratch with NumPy) is trained on the food dataset to predict user preference scores based on nutritional features, replacing hardcoded category-keyword rules.

---

## Key Features
---------------

- Dynamic re-planning based on logged meals
- Multi-objective optimization including:
  - Nutrition targets (calories, protein, etc.)
  - Variety across meals
  - Meal compatibility (realistic combinations)
  - User preferences via Gaussian Naive Bayes (trained on nutritional features)
  - Follow-through modeling (foods the user tends to accept/reject)
- Greedy initialization for fast convergence
- Support for both SA and GA optimizers

---

## Fitness Function
-------------------

Each meal plan is evaluated using a weighted score:
FinalFitness =
w_nutrition * NutritionScore
	•	w_variety * VarietyScore
	•	w_compatibility * MealCompatibilityScore
	•	w_preference * UserPreferenceScore
	•	w_follow * FollowThroughScore

Where:
- **NutritionScore**: how close the plan is to macro targets
- **VarietyScore**: penalizes repeated categories
- **MealCompatibilityScore**: encourages realistic meal compositions
- **UserPreferenceScore**: predicted by a Gaussian Naive Bayes classifier trained on nutritional features; generalizes to foods in unknown categories
- **FollowThroughScore**: models whether the user is likely to follow the plan

---

## Project Pipeline
-------------------

1. **Preprocess data**
   - Combines USDA + OpenFoodFacts datasets
   - Cleans and limits dataset size

2. **Greedy Initialization**
   - Builds a strong starting meal plan

3. **Optimization**
   - SA refines a single solution
   - GA evolves a population of solutions

4. **Final Output**
   - Meal plan per slot
   - Full-day nutritional totals
   - Fitness breakdown

---

## Setup Instructions
---------------------

### 1. Install dependencies

```bash
pip install pandas numpy
```

```bash
pip install deep-translator
```

### 2. Download datasets
Download:
	•	USDA FoodData Central (Foundation Foods) https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2025-12-18.zip
	•	OpenFoodFacts dataset https://www.kaggle.com/datasets/openfoodfacts/world-food-facts
Place them inside:

```bash
data/
```

such that the relative path of the FoodData dataset folder is data\FoodData_Central_foundation_food_csv_2025-12-18
and the OpenFoodFacts .tsv file is data\en.openfoodfacts.org.products.tsv

### 3. Preprocess data
```bash
python src/preprocess.py
```
This generates:
```bash
data/processed/foods.csv
```

## How to Run
---------------------
### Run with Simulated Annealing

```bash
python main.py \
  --optimizer sa \
  --foods_path data/processed/foods.csv \
  --greedy_init \
  --slots breakfast lunch dinner snack \
  --slots_done breakfast
  --seed $RANDOM 
```

### Run with Genetic Algorithm
```bash
python main.py \
  --optimizer ga \
  --foods_path data/processed/foods.csv \
  --greedy_init \
  --slots breakfast lunch dinner snack \
  --slots_done breakfast \
  --seed $RANDOM
```

### For Windows Computers:
python main.py --optimizer sa --foods_path data/processed/foods.csv --greedy_init --slots breakfast lunch dinner snack
python main.py --optimizer sa --foods_path data/processed/foods.csv --greedy_init --slots breakfast lunch dinner snack --slots_done breakfast
python main.py --optimizer ga --foods_path data/processed/foods.csv --greedy_init --slots breakfast lunch dinner snack --slots_done breakfast


### IMportant Arguments
	•	--optimizer: sa or ga
	•	--greedy_init: auto-generate starting plan (recommended)
	•	--slots: all meal slots
	•	--slots_done: meals already eaten
	•	--foods_path: path to processed dataset

Optional:
	•	nutrition targets (--target_energy, etc.)
	•	scoring weights (--w_nutrition, etc.)

## Example output
---------------------
The program prints:
	•	Meal plan per slot
	•	Nutritional totals vs targets
	•	Fitness breakdown

Example:
```bash
Full day totals (logged + planned)
  energy_kcal: 1907 / 2000
  protein_g: 50.1 / 50.0

Fitness Breakdown
  NutritionScore: 0.973
  VarietyScore: 1.000
  Final Fitness: 0.904
```

## Notes & Limitations
---------------------
	•	Some meal combinations may be nutritionally optimal but unrealistic
	•	Meal compatibility is currently heuristic-based
	•	Genetic Algorithm typically performs better than Simulated Annealing for this problem

## Future IMprovements
---------------------
	•	Stronger meal realism constraints
	•	Better mapping between food categories and meal types
	•	Multi-day planning with grocery optimization

Greedy Initialization (initialize.py)
======================================

Generates a starting meal plan for the simulated annealing optimizer.
Rather than starting from a random or hand-supplied state, this module
builds an initial plan that is already reasonably close to the user's
nutritional targets, which speeds up SA convergence.


How it works
------------

For each remaining meal slot:

1. Divide the remaining nutrient budget evenly across unfilled slots.
2. Sample a pool of candidate foods at random from the food database.
3. For each candidate, compute the serving multiplier that best hits
   the per-slot calorie target, then score it by how well it covers
   the other nutrient targets (protein, fat, carbs, fiber).
4. Apply a soft penalty to candidates whose category has already
   appeared in the plan, to encourage variety.
5. Pick the lowest-scoring candidate and add it to the plan.
6. Subtract that food's nutrient contribution from the remaining
   budget before moving on to the next slot.


Usage
-----

Call greedy_init() directly:

    from src.initialize import greedy_init

    initial_state = greedy_init(
        foods_df=foods,
        logged=logged_today,
        targets=targets,
        n_slots=len(remaining_slots),
        penalty_cfg=penalty_cfg,
        candidate_pool_size=50,  # optional, default 50
        seed=42,                 # optional, default 42
    )

Or pass --greedy_init when running main.py:

    python main.py --greedy_init

This replaces the need to supply --initial_state manually.


Parameters
----------

foods_df            Cleaned food database (output of preprocess.py).

logged              Nutrients already consumed today, e.g.
                    {"energy_kcal": 400, "protein_g": 15, ...}.

targets             Full-day nutrient targets, e.g.
                    {"energy_kcal": 2000, "protein_g": 50, ...}.

n_slots             Number of remaining meal slots to fill.

penalty_cfg         Config dict shared with the SA scorer. Must contain
                    "portion_min" and "portion_max". Optionally contains
                    "variety_penalty" (default 1.2), a score multiplier
                    applied when a candidate's category is already in
                    the plan.

candidate_pool_size How many foods to evaluate per slot. Higher values
                    improve initial plan quality at the cost of speed.
                    50 is a reasonable default for most dataset sizes.

seed                Random seed for reproducibility.


Notes
-----

- The initializer only considers targets (lower-bound nutrients like
  calories, protein, fiber). Upper limits like sodium and sugar are
  not penalized here -- the SA scoring function handles those.

- Serving sizes are anchored to the per-slot calorie target. A food
  providing 200 kcal per 100g in a 500 kcal slot will be assigned a
  multiplier of 2.5 (250g), clamped to [portion_min, portion_max].

- If the food database is very small, the pool sampling falls back to
  allowing repeated foods to avoid failure.
