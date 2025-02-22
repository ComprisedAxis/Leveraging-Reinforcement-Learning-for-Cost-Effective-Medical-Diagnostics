import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from environment import DiabetesDiagnosisEnv
from sb3_contrib import MaskablePPO
from imputation import Imputer

custom_colors = [ "#e9edee", "#006778", "#6f0a19"]
custom_alphas = [0.30, 1.0, 0.44]  # alpha for each color respectively

# Global configurations
PANEL_COSTS = {
    "Biochemical Profile": -25,
    "Complete Hemochrome (CBC)": -20,
    "Fasting Glucose": -5,
    "Glycohemoglobin (HbA1c)": -10,
    "BMI & Body Composition": -15,
    "Triglycerides & LDL": -15,
    "HDL Cholesterol": -15,
    "Total Cholesterol": -15,
    "Insulin": -20,
    "HS C-Reactive Protein (CRP)": -20
}
test_panels = {
    "Biochemical Profile": ["LBXSGL", "LBXSAL", "LBXSCR"],
    "Complete Hemochrome (CBC)": ["LBXWBCSI", "LBXHGB", "LBXPLTSI"],
    "Fasting Glucose": ["LBXGLU"],
    "Glycohemoglobin (HbA1c)": ["LBXGH"],
    "BMI & Body Composition": ["BMXBMI", "BMXWAIST"],
    "Triglycerides & LDL": ["LBXTR", "LBDLDL"],
    "HDL Cholesterol": ["LBDHDD"],
    "Total Cholesterol": ["LBXTC"],
    "Insulin": ["LBXIN"],
    "HS C-Reactive Protein (CRP)": ["LBXHSCRP"]
}
FEATURE_COLS = [test for panel in test_panels for test in test_panels[panel]]

DATA_PATH = "data/train.csv"
IMPUTE_PARAMS = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}

LAMBDA_VALUES = [5, 10]
RHO_VALUES = [0.1, 0.2]
VERBOSE = False


def load_and_impute_data(data_path: str, feature_cols: list, impute_params: dict) -> tuple:
    """Loads the dataset, imputes missing values, and returns the processed data and imputer."""
    data = pd.read_csv(data_path).dropna(subset=['Has_Diabetes'])
    X_full = data[feature_cols].values
    imputer = Imputer(dim=X_full.shape[1], impute_para=impute_params)
    mask = np.isnan(X_full).astype(int)
    imputer.set_dataset(X_full, mask)
    imputer.train_model(max_iter=50)
    X_imputed = imputer.transform(X_full)
    data[feature_cols] = X_imputed
    return data, imputer

def predict_with_probs(model, obs, action_masks):
    # Get the action as usual.
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

    # Convert observation to tensor using the policy's helper.
    obs_tensor = model.policy.obs_to_tensor(obs)

    # If obs_tensor is a tuple, extract the tensor (e.g., the first element)
    if isinstance(obs_tensor, tuple):
        obs_tensor = obs_tensor[0]

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=action_masks)

        # Attempt to extract the underlying categorical distribution
        if hasattr(distribution, "distribution"):
            base_distribution = distribution.distribution
            # Try to extract probabilities
            if hasattr(base_distribution, "probs"):
                probs = base_distribution.probs.cpu().numpy().squeeze()
            elif hasattr(base_distribution, "logits"):
                probs = torch.softmax(base_distribution.logits, dim=-1).cpu().numpy().squeeze()
            else:
                raise AttributeError("Underlying distribution has neither 'probs' nor 'logits'.")
        else:
            raise AttributeError("Maskable distribution does not have an underlying 'distribution' attribute.")

    return action, probs

def load_pareto_models(lambda_vals, rho_vals, device: torch.device) -> list:
    """Loads Pareto models for given lambda and rho values; returns a list of (lambda, rho, model) tuples."""
    policies = []
    # Loop through all combinations of lambda and rho values to load model files.
    for lambda_param in lambda_vals:
        for rho_param in rho_vals:
            model_path = f"models/ppo_f1_lambda{lambda_param}_rho{rho_param}.pth"
            if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
                print(f"Model {model_path} not found, skipping.")
                continue
            try:
                model = MaskablePPO.load(model_path, device=device)
                policies.append((lambda_param, rho_param, model))
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")
    return policies


def evaluate_system(model, env, n_episodes: int = 1000) -> tuple:
    """
    Evaluates the system using the given model and environment.
    Returns:
      tests_list: List of tests prescribed per episode.
      costs_list: List of accumulated costs per episode.
      accuracy: Overall accuracy.
      f1_score: F1 score.
      FN: False negatives.
      prescribed_tests_freq: Dictionary with frequency count for each test panel.
    """
    tests_list, costs_list = [], []
    # Initialize frequency counter for each test panel
    test_counts = {panel: 0 for panel in env.available_panels}
    correct, TP, FP, TN, FN = 0, 0, 0, 0, 0
    continuous_scores = []
    true_labels_list = []

    for episode in tqdm(range(n_episodes), desc="Evaluating Episodes"):
        obs = env.reset()
        done = False
        tests_count = 0
        cost_accum = 0.0

        while not done:
            action_masks = env.action_masks()
            action, probs = predict_with_probs(model, obs, action_masks)
            obs, reward, done, _ = env.step(action)
            if action >= env.num_panels:
                diagnosis_probs = probs
            if action < env.num_panels:
                tests_count += 1
                panel = env.available_panels[action]
                test_counts[panel] += 1  # count the specific test
                cost_accum += abs(env.panel_costs[panel])
        tests_list.append(tests_count)
        costs_list.append(cost_accum)

        if action == env.num_panels:
            predicted_label = 1
        elif action == env.num_panels + 1:
            predicted_label = 0
        else:
            predicted_label = None

        true_label = env.current_patient["Has_Diabetes"]
        true_labels_list.append(env.current_patient["Has_Diabetes"])

        if diagnosis_probs is not None:
            # Adjust the index according to your action ordering.
            # Here we assume the diagnosis probabilities are in the last two indices,
            # and that the Diabetes probability is at index -2.
            continuous_scores.append(diagnosis_probs[-2])
        else:
            continuous_scores.append(0.0)

        if predicted_label is not None:
            if predicted_label == true_label:
                correct += 1
            if predicted_label == 1:
                if true_label == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if true_label == 0:
                    TN += 1
                else:
                    FN += 1

    accuracy = correct / n_episodes
    f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    return tests_list, costs_list, accuracy, f1_score, FN, test_counts, true_labels_list, continuous_scores

def plot_test_frequencies(test_counts: dict, title: str = "Frequency of Prescribed Tests"):
    plt.figure(figsize=(14,14.5))
    sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), palette=custom_colors)
    plt.title(title)
    plt.xlabel("Test Panel")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

def plot_tests_histogram(tests_list: list):
    plt.figure(figsize=(10, 8))
    sns.histplot(tests_list, kde=True, color=custom_colors[0])
    plt.xlabel("Number of Tests Prescribed per Episode")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tests Prescribed per Episode")

    # Rotate x-axis labels and ensure they fit in the figure
    plt.xticks(rotation=45, ha='right')

    # Adjust layout so the labels aren't cut off
    plt.tight_layout()

    plt.show()

def plot_roc(true_labels_list, continuous_scores):
    fpr, tpr, thresholds = roc_curve(true_labels_list, continuous_scores)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def evaluate_pareto_policies(policies: list, env, n_episodes: int = 1000) -> pd.DataFrame:
    """Evaluates each Pareto policy on the environment and returns a DataFrame with evaluation metrics."""
    results = []
    custom_colors = ["#006778", "#6f0a19", "#e9edee"]
    custom_alphas = [1.0, 0.44, 0.30]
    print("\n=== Evaluating Pareto Policies ===")
    for lambda_param, rho_param, model in tqdm(policies, desc="Evaluating Policies"):
        print(f"\nEvaluating policy with lambda={lambda_param}, rho={rho_param}...")
        env.tp_reward = lambda_param
        env.tn_reward = 1.0
        env.rho_param = rho_param
        tests_list, costs_list, acc, f1, FN, test_counts, true_labels_list, continuous_scores = evaluate_system(model, env, n_episodes)
        results.append({
            "lambda": lambda_param,
            "rho": rho_param,
            "f1_score": f1,
            "accuracy": acc,
            "average_panels": np.mean(tests_list),
            "average_cost": np.mean(costs_list),
            "false_negatives": FN
        })
        plot_roc(true_labels_list, continuous_scores)
        plot_test_frequencies(test_counts)
        plot_tests_histogram(tests_list)

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No models were loaded. Please check that the correct model files exist.")
        return df_results



    # Plot Pareto front: Cost vs. F1 Score trade-off.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_results["average_cost"],
        y=df_results["f1_score"],
        hue=df_results["lambda"],
        palette="coolwarm",
        s=100
    )
    plt.xlabel("Average Cost per Episode ($)")
    plt.ylabel("F1 Score")
    plt.title("Pareto Front: Cost vs. F1-Score Trade-off")
    plt.legend(title="Lambda")
    plt.grid(True)
    plt.show()
    return df_results


def compare_policies(df_results: pd.DataFrame):
    """Prints evaluation results and plots a comparison of false negatives per policy."""
    if df_results.empty:
        print("No evaluation results to compare.")
        return
    print("\n=== Comparing All Policies ===")
    print(df_results)

    # Plot comparison of false negatives per policy.
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_results, x="lambda", y="false_negatives", hue="rho", palette="viridis")
    plt.title("False Negatives per Policy")
    plt.xlabel("Lambda")
    plt.ylabel("False Negatives")
    plt.show()


def main():
    """Main function that loads data, models, evaluates policies, and compares policies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data, imputer = load_and_impute_data(DATA_PATH, FEATURE_COLS, IMPUTE_PARAMS)
    pareto_policies = load_pareto_models(LAMBDA_VALUES, RHO_VALUES, device)
    if not pareto_policies:
        print("No models loaded. Please ensure that the trained model files exist with the correct names.")
        return
    env = DiabetesDiagnosisEnv(data, imputer)
    df_results = evaluate_pareto_policies(pareto_policies, env, n_episodes=1000)
    if df_results.empty:
        print("Evaluation did not produce any results.")
    else:
        df_results.to_csv("evaluation_results.csv", index=False)
        print("Saved evaluation results to evaluation_results.csv")
        compare_policies(df_results)


if __name__ == '__main__':
    main()
