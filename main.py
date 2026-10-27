import pandas as pd
import os

from src.constants import BATCH_IDS, FEATURES, TARGET, POLY_DEGREES, KNN_NEIGHBORS,SELECTED_DEGREE
from src.functions import (load_data, calculate_economics, tune_polynomial, tune_knn, build_model,
                           find_best_conditions, plot_validation_curve, plot_condition_effects,
                           plot_harvest_time)

df = load_data(BATCH_IDS)
print(df.shape)

# Space-time profit of each measurement
cost, value, stp, sty = calculate_economics(
    df["Initial Glucose [mM]"], df["Initial Glutamine [mM]"], df["Initial VCD [cells/mL]"],
    df["mAb [mg/L]"], df["Time [d]"])
df["Space-Time Profit [$/(L d)]"] = stp

best = df.loc[df["Space-Time Profit [$/(L d)]"].idxmax()]
print("\nBest observed batch:")
print(best[["Batch ID", "Initial Glucose [mM]", "Initial Glutamine [mM]",
            "Initial VCD [cells/mL]", "Time [d]", "mAb [mg/L]", "Space-Time Profit [$/(L d)]"]])

# Model inputs and output
x = df[FEATURES].to_numpy()
y = df[TARGET].to_numpy()
groups = df["Batch ID"].to_numpy()

poly_gscv = tune_polynomial(x, y, groups, POLY_DEGREES)
poly_results = pd.DataFrame(poly_gscv.cv_results_)
print("\nPolynomial regression:")
for _, row in poly_results.iterrows():
    print(f"  degree {row['param_polynomialfeatures__degree']} | "
          f"train RMSE {-row['mean_train_score']:.3f} | "
          f"valid RMSE {-row['mean_test_score']:.3f}")

knn_gscv = tune_knn(x, y, groups, KNN_NEIGHBORS)
print("\nkNN regression:")
print(f"Best: {knn_gscv.best_params_}")
print(f"Best valid RMSE: {-knn_gscv.best_score_:.3f}")

# Final model and optimization
model = build_model(x, y, SELECTED_DEGREE)
rmse = -poly_results.loc[poly_results["param_polynomialfeatures__degree"] == SELECTED_DEGREE,
                         "mean_test_score"].iloc[0]

ranking = find_best_conditions(model, rmse)
print("\nRecommended conditions:")
print(ranking.iloc[0])

# Export tables
poly_table = pd.DataFrame({
    "degree": poly_results["param_polynomialfeatures__degree"],
    "train_rmse": -poly_results["mean_train_score"],
    "valid_rmse": -poly_results["mean_test_score"],
}).round(3)
poly_table.to_csv(os.path.join("tables", "polynomial_results.csv"), index=False)

comparison = pd.DataFrame({
    "model": [f"Polynomial (degree {SELECTED_DEGREE})", f"kNN {knn_gscv.best_params_}"],
    "valid_rmse": [rmse, -knn_gscv.best_score_],
}).round(3)
comparison.to_csv(os.path.join("tables", "model_comparison.csv"), index=False)

ranking.head(10).round(2).to_csv(os.path.join("tables", "top_10_conditions.csv"), index=False)

# Export figures
plot_validation_curve(poly_results, os.path.join("figures", "polynomial_validation_curve.png"))
plot_condition_effects(ranking, os.path.join("figures", "condition_effects.png"))
plot_harvest_time(ranking, os.path.join("figures", "harvest_time.png"))