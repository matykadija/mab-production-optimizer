import itertools
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.constants import (DATASET_DIR, INITIAL_CONDITIONS_PATH, DOWNTIME_D,
                           GLUCOSE_COST, GLUTAMINE_COST, VCD_COST, MAB_VALUE, CV_FOLDS,GLUCOSE_LEVELS, GLUTAMINE_LEVELS,
                           VCD_LEVELS, HARVEST_TIMES_D, Z_95)

def load_data(batch_ids):
    """Combine the initial conditions and the time series of each batch into one table."""
    df_ic = pd.read_csv(INITIAL_CONDITIONS_PATH)

    batches = []
    for batch_id in batch_ids:
        df_batch = pd.read_csv(os.path.join(DATASET_DIR, f"batch_{batch_id:04d}.csv"))
        ic = df_ic[df_ic["Batch ID"] == batch_id]

        df_batch["Batch ID"] = batch_id
        df_batch["Initial Glucose [mM]"]= ic["Glucose [mM]"].iloc[0]
        df_batch["Initial Glutamine [mM]"]= ic["Glutamine [mM]"].iloc[0]
        df_batch["Initial VCD [cells/mL]"]= ic["VCD [cells/mL]"].iloc[0]
        df_batch["Time [d]"] = df_batch["Time [h]"] / 24

        batches.append(df_batch)

    return pd.concat(batches, ignore_index=True)

def calculate_economics(glucose, glutamine, vcd, mab, time_d):
    """Calculate the cost, value, space-time profit and space-time yield."""
    cost = GLUCOSE_COST * glucose + GLUTAMINE_COST * glutamine + VCD_COST * vcd
    value = MAB_VALUE * mab
    profit = value - cost
    stp = profit / (time_d + DOWNTIME_D)
    sty = mab / (time_d + DOWNTIME_D)
    return cost, value, stp, sty
def tune_polynomial(x, y, groups, degrees):
    """Find the best polynomial degree, keeping each batch in a single fold."""
    model = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
    param_grid = dict(polynomialfeatures__degree=degrees)
    model_gscv = GridSearchCV(model, param_grid=param_grid, cv=GroupKFold(n_splits=CV_FOLDS),
                              scoring="neg_root_mean_squared_error", return_train_score=True)
    model_gscv.fit(x, y, groups=groups)
    return model_gscv


def tune_knn(x, y, groups, n_neighbors):
    """Find the best k and weighting for kNN, keeping each batch in a single fold."""
    model = make_pipeline(StandardScaler(), KNeighborsRegressor())
    param_grid = dict(kneighborsregressor__n_neighbors=n_neighbors,
                      kneighborsregressor__weights=["uniform", "distance"])
    model_gscv = GridSearchCV(model, param_grid=param_grid, cv=GroupKFold(n_splits=CV_FOLDS),
                              scoring="neg_root_mean_squared_error", return_train_score=True)
    model_gscv.fit(x, y, groups=groups)
    return model_gscv

def build_model(x, y, degree):
    """Train the final polynomial model on all 30 batches."""
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                          LinearRegression())
    model.fit(x, y)
    return model


def find_best_conditions(model, rmse):
    """Predict every combination of initial conditions and harvest times, ranked by space-time profit."""
    grid = pd.DataFrame(itertools.product(GLUCOSE_LEVELS, GLUTAMINE_LEVELS, VCD_LEVELS, HARVEST_TIMES_D),
                        columns=["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]", "Harvest Time [d]"])
    glucose = grid["Glucose [mM]"]
    glutamine = grid["Glutamine [mM]"]
    vcd = grid["VCD [cells/mL]"]
    time_d = grid["Harvest Time [d]"]

    # Predicted mAb and its 95% prediction interval
    mab = model.predict(grid.to_numpy())
    margin = Z_95 * rmse

    cost, value, stp, sty = calculate_economics(glucose, glutamine, vcd, mab, time_d)
    _, value_low, stp_low, sty_low = calculate_economics(glucose, glutamine, vcd, mab - margin, time_d)
    _, value_high, stp_high, sty_high = calculate_economics(glucose, glutamine, vcd, mab + margin, time_d)

    grid["mAb [mg/L]"] = mab
    grid["Cost [$/L]"] = cost
    grid["Value [$/L]"] = value
    grid["Value Low [$/L]"] = value_low
    grid["Value High [$/L]"] = value_high
    grid["Space-Time Profit [$/(L d)]"] = stp
    grid["Space-Time Profit Low [$/(L d)]"] = stp_low
    grid["Space-Time Profit High [$/(L d)]"] = stp_high
    grid["Space-Time Yield [mg/(L d)]"] = sty
    grid["Space-Time Yield Low [mg/(L d)]"] = sty_low
    grid["Space-Time Yield High [mg/(L d)]"] = sty_high

    return grid.sort_values("Space-Time Profit [$/(L d)]", ascending=False)

def plot_validation_curve(results, filepath):
    """Plot the training and validation errors for each polynomial degree."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    degrees = results["param_polynomialfeatures__degree"]
    ax.plot(degrees, -results["mean_train_score"], marker="o", label="Training")
    ax.plot(degrees, -results["mean_test_score"], marker="o", label="Validation")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("mAb RMSE (mg/L)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_condition_effects(ranking, filepath):
    """Plot the best achievable space-time profit and yield for each level of each initial condition."""
    conditions = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
    outputs = ["Space-Time Profit [$/(L d)]", "Space-Time Yield [mg/(L d)]"]

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for i, output in enumerate(outputs):
        for j, condition in enumerate(conditions):
            best = ranking.groupby(condition)[output].max()
            axs[i, j].plot(best.index, best.values, marker="o")
            axs[i, j].set_xlabel(f"Initial {condition}")
            axs[i, j].set_ylabel(f"Best {output}")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_harvest_time(ranking, filepath):
    """Plot the space-time profit over harvest time for the recommended initial conditions."""
    best = ranking.iloc[0]
    mask = ((ranking["Glucose [mM]"] == best["Glucose [mM]"])
            & (ranking["Glutamine [mM]"] == best["Glutamine [mM]"])
            & (ranking["VCD [cells/mL]"] == best["VCD [cells/mL]"]))
    df = ranking[mask].sort_values("Harvest Time [d]")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["Harvest Time [d]"], df["Space-Time Profit [$/(L d)]"], label="Prediction")
    ax.fill_between(df["Harvest Time [d]"], df["Space-Time Profit Low [$/(L d)]"],
                    df["Space-Time Profit High [$/(L d)]"], alpha=0.3, label="95% prediction interval")
    ax.axvline(best["Harvest Time [d]"], color="gray", linestyle="--", label="Recommended harvest")
    ax.set_xlabel("Harvest time (d)")
    ax.set_ylabel("Space-time profit ($/(L d))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)