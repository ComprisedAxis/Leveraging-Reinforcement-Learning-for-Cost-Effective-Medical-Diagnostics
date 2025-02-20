import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from environment import DiabetesDiagnosisEnv
from sb3_contrib import MaskablePPO
from imputation import Imputer

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
FEATURE_COLS = [test for panel in test_panels.keys() for test in test_panels[panel]]

DATA_PATH = "data/nhanes_preprocessed.csv"
IMPUTE_PARAMS = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}

LAMBDA_VALUES = [5, 10]
RHO_VALUES = [0.1, 0.2]
VERBOSE = False

def load_and_impute_data(data_path: str, feature_cols: list, impute_params: dict) -> tuple:
    data = pd.read_csv(data_path).dropna(subset=['Has_Diabetes'])
    X_full = data[feature_cols].values
    imputer = Imputer(dim=X_full.shape[1], impute_para=impute_params)
    mask = np.isnan(X_full).astype(int)
    imputer.set_dataset(X_full, mask)
    imputer.train_model(max_iter=50)
    X_imputed = imputer.transform(X_full)
    data[feature_cols] = X_imputed
    return data, imputer

def load_pareto_models(lambda_vals, rho_vals, device: torch.device) -> list:
    from sb3_contrib import MaskablePPO
    policies = []
    for lambda_param in lambda_vals:
        for rho_param in rho_vals:
            model_path = f"ppo_f1_lambda{lambda_param}_rho{rho_param}.pth"
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

def evaluate_system(model, env, n_episodes: int = 10000) -> tuple:
    tests_list, costs_list = [], []
    correct, TP, FP, TN, FN = 0, 0, 0, 0, 0

    for episode in tqdm(range(n_episodes), desc="Evaluating Episodes"):
        obs = env.reset()
        done = False
        tests_count = 0
        cost_accum = 0.0

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, _ = env.step(action)
            if action < env.num_panels:
                tests_count += 1
                panel = env.available_panels[action]
                cost_accum += abs(env.panel_costs[panel])
        if action == env.num_panels:
            predicted_label = 1
        elif action == env.num_panels + 1:
            predicted_label = 0
        else:
            predicted_label = None

        true_label = env.current_patient["Has_Diabetes"]
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

        tests_list.append(tests_count)
        costs_list.append(cost_accum)

    accuracy = correct / n_episodes
    f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    return tests_list, costs_list, accuracy, f1_score, FN

def evaluate_pareto_policies(policies: list, env, n_episodes: int = 10000) -> pd.DataFrame:
    results = []
    print("\n=== Evaluating Pareto Policies ===")
    for lambda_param, rho_param, model in tqdm(policies, desc="Evaluating Policies"):
        print(f"\nEvaluating policy with lambda={lambda_param}, rho={rho_param}...")
        env.tp_reward = lambda_param
        env.tn_reward = 1.0
        env.rho_param = rho_param
        tests_list, costs_list, acc, f1, FN = evaluate_system(model, env, n_episodes)
        results.append({
            "lambda": lambda_param,
            "rho": rho_param,
            "f1_score": f1,
            "accuracy": acc,
            "average_panels": np.mean(tests_list),
            "average_cost": np.mean(costs_list),
            "false_negatives": FN
        })
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No models were loaded. Please check that the correct model files exist.")
        return df_results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_results["average_cost"],
                    y=df_results["f1_score"],
                    hue=df_results["lambda"],
                    palette="coolwarm",
                    s=100)
    plt.xlabel("Average Cost per Episode ($)")
    plt.ylabel("F1 Score")
    plt.title("Pareto Front: Cost vs. F1-Score Trade-off")
    plt.legend(title="Lambda")
    plt.grid(True)
    plt.show()
    return df_results

def compare_policies(df_results: pd.DataFrame):
    if df_results.empty:
        print("No evaluation results to compare.")
        return
    print("\n=== Comparing All Policies ===")
    print(df_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_results, x="lambda", y="false_negatives", hue="rho", palette="viridis")
    plt.title("False Negatives per Policy")
    plt.xlabel("Lambda")
    plt.ylabel("False Negatives")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data, imputer = load_and_impute_data(DATA_PATH, FEATURE_COLS, IMPUTE_PARAMS)
    pareto_policies = load_pareto_models(LAMBDA_VALUES, RHO_VALUES, device)
    if not pareto_policies:
        print("No models loaded. Please ensure that the trained model files exist with the correct names.")
        return
    env = DiabetesDiagnosisEnv(data, imputer)
    df_results = evaluate_pareto_policies(pareto_policies, env, n_episodes=10000)
    if df_results.empty:
        print("Evaluation did not produce any results.")
    else:
        df_results.to_csv("evaluation_results.csv", index=False)
        print("Saved evaluation results to evaluation_results.csv")
        compare_policies(df_results)

if __name__ == '__main__':
    main()