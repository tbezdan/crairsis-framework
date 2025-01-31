import os

# These will be set dynamically by the importing script
ROOT = None  # Placeholder for the ROOT path


def configure_paths(root_path):
    global ROOT
    ROOT = root_path  # Dynamically set ROOT from the main script

    return {
        "original_datasets_path": os.path.join(ROOT, "original_datasets"),
        "models_path": os.path.join(ROOT, "models"),
        "top_sage_features_models_path": os.path.join(
            ROOT, "models", "top_sage_features_models"
        ),
        "output_path": os.path.join(ROOT, "data"),
        "json_path": os.path.join(ROOT, "data", "json_options"),
        "shap_folder": os.path.join(ROOT, "data", "shap_local_relative_normalized"),
        "shap_clusters_folder": os.path.join(ROOT, "data", "shap_clusters"),
        "shap_subclusters_folder": os.path.join(
            ROOT, "data", "shap_clusters", "subclusters"
        ),
        "sage_folder": os.path.join(ROOT, "data", "sage"),
        "interactions_folder": os.path.join(ROOT, "data", "interactions"),
        "interactions_feature_folder": os.path.join(ROOT, "data", "interactions", "fi"),
        "actual_predicted_folder": os.path.join(ROOT, "data", "actual_predicted"),
        "original_data_id_folder": os.path.join(ROOT, "data", "original_data_id"),
        "results_folder": os.path.join(ROOT, "data", "results"),
        "datasets_path": os.path.join(ROOT, "data", "datasets"),
        "nshap_folder": os.path.join(ROOT, "data", "nshap"),
        "gshap_folder": os.path.join(ROOT, "data", "gshap"),
        "isage_folder": os.path.join(ROOT, "data", "isage"),
        "paths_to_check": [
            os.path.join(ROOT, "models"),
            os.path.join(ROOT, "data"),
            os.path.join(ROOT, "data", "json_options"),
            os.path.join(ROOT, "data", "shap_local_relative_normalized"),
            os.path.join(ROOT, "data", "shap_clusters"),
            os.path.join(ROOT, "data", "shap_clusters", "subclusters"),
            os.path.join(ROOT, "data", "sage"),
            os.path.join(ROOT, "data", "interactions"),
            os.path.join(ROOT, "data", "interactions", "fi"),
            os.path.join(ROOT, "data", "actual_predicted"),
            os.path.join(ROOT, "data", "original_data_id"),
            os.path.join(ROOT, "data", "results"),
            os.path.join(ROOT, "data", "datasets"),
            os.path.join(ROOT, "data", "nshap"),
            os.path.join(ROOT, "data", "gshap"),
            os.path.join(ROOT, "data", "isage"),
            os.path.join(ROOT, "models", "top_sage_features_models"),
        ],
        "best_models_path": os.path.join(ROOT, "data", "best_models.csv"),
    }


# import os


# # ROOT = "/Users/timea/Documents/Projekti/craAIRsis/Covid BG"
# # ROOT = "/Users/timea/Documents/Projekti/craAIRsis/TEST"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\cls"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\eu"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\GZJZ"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\cls_tourism"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\EU_reg"
# ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\EU_cls"
# # ROOT = r"C:\Users\tbezdan\Desktop\crAIRsis datasets\reg_test"


# original_datasets_path = os.path.join(ROOT, "original_datasets")


# models_path = os.path.join(ROOT, "models")
# top_sage_features_models_path = os.path.join(ROOT, "models", "top_sage_features_models")
# output_path = os.path.join(ROOT, "data")
# json_path = os.path.join(ROOT, "data", "json_options")
# shap_folder = os.path.join(ROOT, "data", "shap_local_relative_normalized")
# shap_clusters_folder = os.path.join(ROOT, "data", "shap_clusters")
# shap_subclusters_folder = os.path.join(ROOT, "data", "shap_clusters", "subclusters")
# sage_folder = os.path.join(ROOT, "data", "sage")
# interactions_folder = os.path.join(ROOT, "data", "interactions")
# interactions_feature_folder = os.path.join(ROOT, "data", "interactions", "fi")
# actual_predicted_folder = os.path.join(ROOT, "data", "actual_predicted")
# original_data_id_folder = os.path.join(ROOT, "data", "original_data_id")
# results_folder = os.path.join(ROOT, "data", "results")
# datasets_path = os.path.join(ROOT, "data", "datasets")
# nshap_folder = os.path.join(ROOT, "data", "nshap")
# gshap_folder = os.path.join(ROOT, "data", "gshap")
# isage_folder = os.path.join(ROOT, "data", "isage")


# paths_to_check = [
#     models_path,
#     output_path,
#     json_path,
#     shap_folder,
#     shap_clusters_folder,
#     shap_subclusters_folder,
#     sage_folder,
#     interactions_folder,
#     interactions_feature_folder,
#     actual_predicted_folder,
#     original_data_id_folder,
#     results_folder,
#     datasets_path,
#     nshap_folder,
#     gshap_folder,
#     isage_folder,
#     top_sage_features_models_path,
# ]


# best_models_path = os.path.join(ROOT, "data", "best_models.csv")
