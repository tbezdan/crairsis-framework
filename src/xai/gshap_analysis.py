import warnings

warnings.simplefilter("ignore")
import gshap
from gshap.probability_distance import ProbabilityDistance
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import joblib
import os
from sklearn.preprocessing import StandardScaler


from utils.logger import setup_logger
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from gshap.intergroup import IntergroupDifference
from gshap.hypothesis import HypothesisTest

logger = setup_logger(__name__)


def get_data_model(
    datasets_path,
    models_path,
    file,
    filename,
    filter_column,
    filter_value,
    target,
    datetime_col,
):

    data_path = os.path.join(
        datasets_path,
        f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}.csv",
    )

    X = pd.read_csv(data_path)

    columns_to_drop = ["usage", "id", target]
    if datetime_col:
        columns_to_drop.append(datetime_col)

    X_train = X[X["usage"] == "train"]
    X_test = X[X["usage"] == "test"]
    y_train = X[X["usage"] == "train"][target]
    y_test = X[X["usage"] == "test"][target]
    X_train = X_train.drop(columns_to_drop, axis=1)
    X_test = X_test.drop(columns_to_drop, axis=1)

    feature_names = X_train.columns

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    model_path = os.path.join(models_path, file + ".joblib")
    model = joblib.load(model_path)

    return X_train, X_test, y_train, y_test, feature_names, model


def sample_size(X_train):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    std_dev = np.mean(np.std(X_train_scaled, axis=0))

    margin_of_error = 0.05
    z_score = 1.96

    required_samples = (z_score * std_dev / margin_of_error) ** 2
    return int(np.ceil(required_samples))


def cls_gshap(gshap_folder, model, X_train, X_test, feature_names, file):
    num_classes = model.predict_proba(X_test[:1]).shape[1]

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                logger.info(f"Computing G-SHAP values for Class {i} vs. Class {j}")

                epsilon = 1e-8
                pos_distribution = lambda y_pred: np.clip(
                    y_pred[:, i], epsilon, 1 - epsilon
                )
                neg_distribution = lambda y_pred: np.clip(
                    y_pred[:, j], epsilon, 1 - epsilon
                )
                g = ProbabilityDistance(pos_distribution, neg_distribution)

                explainer = gshap.KernelExplainer(model.predict_proba, X_train, g)
                X = X_test[model.predict(X_test) == i]
                X = X.reshape((1, -1)) if len(X.shape) == 1 else X

                gshap_values = explainer.gshap_values(X)

                g_comparison, g_background = explainer.compare(
                    X, bootstrap_samples=sample_size(X_train)
                )

                g_shap_sum = gshap_values.sum()

                df = pd.DataFrame(
                    {
                        "features": feature_names,
                        "gshap": list(gshap_values),
                    }
                ).sort_values("gshap", ascending=False)

                df["pos_dist_class"] = i
                df["neg_dist_class"] = j

                df.to_csv(
                    os.path.join(gshap_folder, file + f"_gshap_values_{i}_vs_{j}.csv"),
                    index=False,
                )

                pd.DataFrame(
                    {
                        "g_comparison": [g_comparison],
                        "g_background": [g_background],
                        "g_shap_sum": [g_shap_sum],
                        "pos_dist_class": [i],
                        "neg_dist_class": [j],
                    }
                ).to_csv(
                    os.path.join(
                        gshap_folder, file + f"_probabilities_and_sum_{i}_vs_{j}.csv"
                    ),
                    index=False,
                )


def silverman_bandwidth(data):
    n = len(data)
    std_dev = np.std(data, ddof=1)
    return 1.06 * std_dev * n ** (-1 / 5)


def reg_gshap(gshap_folder, model, X_train, X_test, y_train, feature_names, file):

    optimal_bandwidth = silverman_bandwidth(y_train)
    kde_all = KernelDensity(bandwidth=optimal_bandwidth)
    kde_all.fit(y_train.reshape(-1, 1))

    median = np.quantile(y_train, 0.5)
    y_train_q1 = y_train[y_train > median]

    optimal_bandwidth_q1 = silverman_bandwidth(y_train_q1)
    kde_q1 = KernelDensity(bandwidth=optimal_bandwidth_q1)
    kde_q1.fit(y_train_q1.reshape(-1, 1))

    y_pred = model.predict(X_test)

    pos_density = lambda x: np.exp(
        kde_q1.score_samples(x.reshape(-1, 1))
    )  # positive density above average kde_q1
    neg_density = lambda x: np.exp(
        kde_all.score_samples(x.reshape(-1, 1))
    )  # background dataset kde_all

    g = ProbabilityDistance(pos_density, neg_density)
    explainer = gshap.KernelExplainer(model.predict, X_train, g)
    x = X_test[y_pred > median]
    gshap_values = explainer.gshap_values(x)

    g_comparison, g_background = explainer.compare(
        x, bootstrap_samples=sample_size(X_train)
    )

    g_shap_sum = gshap_values.sum()

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap": list(gshap_values),
        }
    ).sort_values("gshap", ascending=False)

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison": [g_comparison],
            "g_background": [g_background],
            "g_shap_sum": [g_shap_sum],
        }
    ).to_csv(
        os.path.join(gshap_folder, file + f"_probabilities_and_sum.csv"),
        index=False,
    )


def cls_gshap_model_failure(
    gshap_folder, model, X_train, X_test, y_train, y_test, feature_names, file
):

    # performance on training data
    g = lambda y_pred: f1_score(y_train, y_pred.argmax(axis=1), average="weighted")
    explainer_train = gshap.KernelExplainer(model.predict_proba, X_train, g)
    gshap_values_train = explainer_train.gshap_values(X_train)

    # performance on test data
    g = lambda y_pred: f1_score(y_test, y_pred.argmax(axis=1), average="weighted")
    explainer_test = gshap.KernelExplainer(model.predict_proba, X_train, g)
    gshap_values_test = explainer_test.gshap_values(X_test)

    g_comparison_train, g_background_train = explainer_train.compare(
        X_train, bootstrap_samples=sample_size(X_train)
    )
    gshap_values_train_sum = gshap_values_train.sum()

    g_comparison_test, g_background_test = explainer_test.compare(
        X_test, bootstrap_samples=sample_size(X_train)
    )
    gshap_values_test_sum = gshap_values_test.sum()

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap_train": list(gshap_values_train),
            "gshap_test": list(gshap_values_test),
        }
    ).sort_values("gshap_test")

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values_model_failure.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison_train": [g_comparison_train],
            "g_background_train": [g_background_train],
            "gshap_values_train_sum": [gshap_values_train_sum],
            "g_comparison_test": [g_comparison_test],
            "g_background_test": [g_background_test],
            "gshap_values_test_sum": [gshap_values_test_sum],
        }
    ).to_csv(
        os.path.join(gshap_folder, file + f"_probabilities_and_sum_model_failure.csv"),
        index=False,
    )


def reg_gshap_model_failure(
    gshap_folder, model, X_train, X_test, y_train, y_test, feature_names, file
):

    # performance on training data
    g = lambda y_pred: r2_score(y_train, y_pred)
    explainer_train = gshap.KernelExplainer(model.predict, X_train, g)
    gshap_values_train = explainer_train.gshap_values(X_train)

    # performance on test data
    g = lambda y_pred: r2_score(y_test, y_pred)
    explainer_test = gshap.KernelExplainer(model.predict, X_train, g)
    gshap_values_test = explainer_test.gshap_values(X_test)

    g_comparison_train, g_background_train = explainer_train.compare(
        X_train, bootstrap_samples=sample_size(X_train)
    )
    gshap_values_train_sum = gshap_values_train.sum()

    g_comparison_test, g_background_test = explainer_test.compare(
        X_test, bootstrap_samples=sample_size(X_train)
    )
    gshap_values_test_sum = gshap_values_test.sum()

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap_train": list(gshap_values_train),
            "gshap_test": list(gshap_values_test),
        }
    ).sort_values("gshap_test")

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values_model_failure.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison_train": [g_comparison_train],
            "g_background_train": [g_background_train],
            "gshap_values_train_sum": [gshap_values_train_sum],
            "g_comparison_test": [g_comparison_test],
            "g_background_test": [g_background_test],
            "gshap_values_test_sum": [gshap_values_test_sum],
        }
    ).to_csv(
        os.path.join(gshap_folder, file + f"_probabilities_and_sum_model_failure.csv"),
        index=False,
    )


def gshap_intergroup_difference(
    gshap_folder,
    model,
    X_train,
    X_test,
    feature_names,
    col_name,
    file,
    gshap_intergroup_difference_selected_values,
    gshap_intergroup_difference_grouping_method,
    gshap_intergroup_difference_grouping_value,
):

    col_index = list(feature_names).index(col_name)

    selected_values = gshap_intergroup_difference_selected_values
    grouping_method = gshap_intergroup_difference_grouping_method
    grouping_value = gshap_intergroup_difference_grouping_value

    # binary values in the column
    if all(v is None for v in [grouping_method, grouping_value, selected_values]):
        unique_values = np.unique(X_test[:, col_index])
        if len(unique_values) == 2:
            binary_grouping = np.where(X_test[:, col_index] == unique_values[0], 0, 1)
    # multiple discrete values in the column
    elif selected_values is not None:
        binary_grouping = np.where(np.isin(X_test[:, col_index], selected_values), 0, 1)
        print("multiple discrete values in the column")
    # user input to separate groups - continues values in the column
    elif all(v is None for v in [grouping_method, selected_values]):
        binary_grouping = X_test[:, col_index] > grouping_value
        print("user input to separate groups - continues values in the column")
    # mean value - continues values in the column
    elif grouping_method == "mean":
        binary_grouping = X_test[:, col_index] > X_test[:, col_index].mean()
        print("mean value - continues values in the column")
    # quantile - continues values in the column
    elif grouping_value is not None:
        binary_grouping = X_test[:, col_index] > np.percentile(
            X_test[:, col_index], int(grouping_value)
        )
        print("quantile - continues values in the column")

    unique_values = np.unique(X_test[:, col_index])
    if len(unique_values) == 2:
        binary_grouping = np.where(X_test[:, col_index] == unique_values[0], 0, 1)

    binary_grouping = X_test[:, col_index] > X_test[:, col_index].mean()

    g = IntergroupDifference(
        group=binary_grouping,
        distance="absolute_mean_distance",
    )

    explainer = gshap.KernelExplainer(model.predict, X_train, g)
    gshap_values = explainer.gshap_values(X_test)

    g_comparison, g_background = explainer.compare(
        X_test, bootstrap_samples=sample_size(X_train)
    )
    g_shap_sum = gshap_values.sum()

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap": list(gshap_values),
        }
    ).sort_values("gshap", ascending=False)

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values_intergroup_difference.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison": [g_comparison],
            "g_background": [g_background],
            "g_shap_sum": [g_shap_sum],
        }
    ).to_csv(
        os.path.join(
            gshap_folder, file + f"_probabilities_and_sum_intergroup_difference.csv"
        ),
        index=False,
    )


def gshap_mediation(
    gshap_folder,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    file,
    ml_model,
    task_type,
    independent_vars,
    model_registry,
):
    XZ, y = X_train, y_train  # both the independent variables (X) and mediators (Z)
    independent_vars_idx = [list(feature_names).index(i) for i in independent_vars]
    X = XZ[:, independent_vars_idx]
    hyperparameters = model.get_params()
    model_name = model_registry[ml_model]
    f_XZ = model_name(**hyperparameters)
    f_XZ.train(XZ, y)
    f_X = model_name(**hyperparameters)
    f_X.train(X, y)

    if task_type == "regression":

        def g(y_pred_XZ):
            f_X = model_name(**hyperparameters)
            f_X.train(X, y_pred_XZ)
            return r2_score(y_pred_XZ, f_X.predict(X))

        explainer = gshap.KernelExplainer(f_XZ.predict, XZ, g)
        gshap_values = explainer.gshap_values(XZ)
        g_comparison, g_background = explainer.compare(
            XZ, bootstrap_samples=sample_size(X_train)
        )
        gshap_sum = gshap_values.sum()
        gshap_values *= (r2_score(y, f_X.predict(X)) - g_background) / (
            g_comparison - g_background
        )

    if task_type == "classification":

        def g(y_pred_XZ):
            f_X = model_name(**hyperparameters)
            f_X.train(X, y_pred_XZ)
            return f1_score(y_pred_XZ, f_X.predict(X), average="weighted")

        explainer = gshap.KernelExplainer(f_XZ.predict, XZ, g)
        gshap_values = explainer.gshap_values(XZ)
        g_comparison, g_background = explainer.compare(
            XZ, bootstrap_samples=sample_size(X_train)
        )
        gshap_sum = gshap_values.sum()
        gshap_values *= (
            f1_score(y, f_X.predict(X), average="weighted") - g_background
        ) / (g_comparison - g_background)

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap": list(gshap_values),
        }
    ).sort_values("gshap", ascending=False)

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values_mediation.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison": [g_comparison],
            "g_background": [g_background],
            "g_shap_sum": [gshap_sum],
        }
    ).to_csv(
        os.path.join(gshap_folder, file + f"_probabilities_and_sum_mediation.csv"),
        index=False,
    )


def gshap_hypothesis(
    gshap_folder,
    model,
    feature_names,
    file,
    X_train,
    X_test,
    y_test,
    gshap_hypothesis_testing_hypothesis_treshold_method,
    gshap_hypothesis_testing_hypothesis_treshold_value,
    gshap_hypothesis_testing_sample_treshold_method,
    gshap_hypothesis_testing_sample_treshold_value,
    gshap_hypothesis_testing_condition,
):

    if gshap_hypothesis_testing_hypothesis_treshold_method == "input":
        h_treshold = gshap_hypothesis_testing_hypothesis_treshold_value
        print("h_test - input")
    elif gshap_hypothesis_testing_hypothesis_treshold_method == "mean":
        h_treshold = y_test.mean()
        print("h_test - mean")
    elif gshap_hypothesis_testing_hypothesis_treshold_method == "quantile":
        h_treshold = np.percentile(
            y_test, int(gshap_hypothesis_testing_hypothesis_treshold_value)
        )
        print("h_test - quantile")

    if gshap_hypothesis_testing_condition == "greater":
        test = lambda y_pred: y_pred.mean() > h_treshold
        print("condition - greater")

    else:
        test = lambda y_pred: y_pred.mean() < h_treshold
        print("condition - less")

    g = HypothesisTest(test, bootstrap_samples=sample_size(X_train))
    explainer = gshap.KernelExplainer(model.predict, X_train, g)

    if gshap_hypothesis_testing_sample_treshold_method == "input":
        sample_treshold = gshap_hypothesis_testing_sample_treshold_value
        print("sample_treshold - input")
    elif gshap_hypothesis_testing_sample_treshold_method == "mean":
        sample_treshold = y_test.mean()
        print("sample_treshold - mean")
    elif gshap_hypothesis_testing_sample_treshold_method == "quantile":
        sample_treshold = np.percentile(
            y_test, int(gshap_hypothesis_testing_sample_treshold_value)
        )
        print("sample_treshold - quantile")

    if gshap_hypothesis_testing_condition == "greater":
        x = X_test[y_test > sample_treshold]
        print("sample_treshold condition - greater")
    else:
        x = X_test[y_test < sample_treshold]
        print("sample_treshold condition - less")

    gshap_values = explainer.gshap_values(x)
    g_comparison, g_background = explainer.compare(
        x, bootstrap_samples=sample_size(X_train)
    )
    gshap_sum = gshap_values.sum()

    df = pd.DataFrame(
        {
            "features": feature_names,
            "gshap": list(gshap_values),
        }
    ).sort_values("gshap", ascending=False)

    df.to_csv(
        os.path.join(gshap_folder, file + f"_gshap_values_hypothesis.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "g_comparison": [g_comparison],
            "g_background": [g_background],
            "g_shap_sum": [gshap_sum],
        }
    ).to_csv(
        os.path.join(gshap_folder, file + f"_probabilities_and_sum_hypothesis.csv"),
        index=False,
    )


def perform_gshap_analysis(
    datasets_path,
    gshap_folder,
    models_path,
    best_models,
    filter_column,
    task_type,
    datetime_col,
    col_name,
    gshap_intergroup_difference_selected_values,
    gshap_intergroup_difference_grouping_method,
    gshap_intergroup_difference_grouping_value,
    independent_vars,
    gshap_hypothesis_testing_hypothesis_treshold_method,
    gshap_hypothesis_testing_hypothesis_treshold_value,
    gshap_hypothesis_testing_sample_treshold_method,
    gshap_hypothesis_testing_sample_treshold_value,
    gshap_hypothesis_testing_condition,
    model_registry,
):

    for i in range(best_models.shape[0]):
        try:
            logger.info(f"Processing model {i+1}/{best_models.shape[0]}: Start")
            target = best_models.loc[i, "target"]
            filename = best_models.loc[i, "filename"]
            filter_value = best_models.loc[i, "filter_value"]
            ml_model = best_models.loc[i, "ml_model"]
            mh_algo = best_models.loc[i, "metaheuristic"]

            file = f"filename_{filename}_filter_col_{filter_column}_filter_val_{filter_value}_target_{target}_ml_model_{ml_model}_mh_algo_{mh_algo}"

            logger.info(f"Loading data and model for {file}")
            X_train, X_test, y_train, y_test, feature_names, model = get_data_model(
                datasets_path,
                models_path,
                file,
                filename,
                filter_column,
                filter_value,
                target,
                datetime_col,
            )

            logger.info("gSHAP started")

            if task_type == "classification":
                cls_gshap(gshap_folder, model, X_train, X_test, feature_names, file)
                logger.info("gSHAP model failure")
                cls_gshap_model_failure(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    feature_names,
                    file,
                )
                logger.info("gSHAP intergroup difference")
                gshap_intergroup_difference(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    feature_names,
                    col_name,
                    file,
                    gshap_intergroup_difference_selected_values,
                    gshap_intergroup_difference_grouping_method,
                    gshap_intergroup_difference_grouping_value,
                )
                logger.info("gSHAP mediation")
                gshap_mediation(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    feature_names,
                    file,
                    ml_model,
                    task_type,
                    independent_vars,
                    model_registry,
                )

            elif task_type == "regression":
                reg_gshap(
                    gshap_folder, model, X_train, X_test, y_train, feature_names, file
                )
                logger.info("gSHAP model failure")
                reg_gshap_model_failure(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    feature_names,
                    file,
                )
                logger.info("gSHAP intergroup difference")
                gshap_intergroup_difference(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    feature_names,
                    col_name,
                    file,
                    gshap_intergroup_difference_selected_values,
                    gshap_intergroup_difference_grouping_method,
                    gshap_intergroup_difference_grouping_value,
                )
                logger.info("gSHAP mediation")
                gshap_mediation(
                    gshap_folder,
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    feature_names,
                    file,
                    ml_model,
                    task_type,
                    independent_vars,
                    model_registry,
                )
                logger.info("gSHAP hypothesis")
                gshap_hypothesis(
                    gshap_folder,
                    model,
                    feature_names,
                    file,
                    X_train,
                    X_test,
                    y_test,
                    gshap_hypothesis_testing_hypothesis_treshold_method,
                    gshap_hypothesis_testing_hypothesis_treshold_value,
                    gshap_hypothesis_testing_sample_treshold_method,
                    gshap_hypothesis_testing_sample_treshold_value,
                    gshap_hypothesis_testing_condition,
                )

            logger.info("gSHAP completed")

        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}", exc_info=True)
