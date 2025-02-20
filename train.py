import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
import os
from imputation import Imputer
from environment import DiabetesDiagnosisEnv

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(model, X, y, epochs=50, lr=1e-3, device=torch.device("cpu"), batch_size=256):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # DataLoader for mini-batch training
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                                             torch.as_tensor(y, dtype=torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Classifier Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return model


def evaluate_classifier(model, X, y, device=torch.device("cpu")):
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.as_tensor(y, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        acc = (preds == y_tensor).float().mean().item()
    return acc

# RL Training Function for a Given (lambda, rho)
def train_rl_policy_for_params(lambda_param, rho_param, policy_kwargs, rl_timesteps=50000,
                               rl_device=torch.device("cpu")):
    """
    Loads data, trains the imputer, creates a vectorized environment,
    trains an RL policy, and saves the model.
    
    Diagnosis rewards:
      - Correct Diabetes prediction yields a reward = lambda_param.
      - Correct Non-Diabetes prediction yields a reward = 1.
      - Incorrect predictions yield the negative of these values.
    
    The test panel cost penalty is given by: rho_param * (panel_cost) (with panel_costs negative).
    """
    print(f"\n[PID {os.getpid()}] Starting RL training for lambda={lambda_param}, rho={rho_param} on {rl_device}")
    
    # Data Loading and Imputer Training 
    data = pd.read_csv("data/nhanes_preprocessed.csv").dropna(subset=['Has_Diabetes'])
    
    # Define panel costs and test panels.
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
    feature_cols = [test for panel in test_panels.keys() for test in test_panels[panel]]
    X_full = data[feature_cols].values
    
    impute_params = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}
    imputer = Imputer(dim=X_full.shape[1], impute_para=impute_params)
    mask = np.isnan(X_full).astype(int)
    imputer.set_dataset(X_full, mask)
    imputer.train_model(max_iter=50)
    X_imputed = imputer.transform(X_full)
    data[feature_cols] = X_imputed  # update data with imputed values
    
    # Set Up the RL Environment with our New Parameters
    def make_env():
        env = DiabetesDiagnosisEnv(
            data, imputer,
            early_diagnosis_penalty=10.0,
            min_tests_required=2,
            gamma=0.1,
            bonus_factor=0.5
        )
        # Set diagnosis rewards
        env.tp_reward = lambda_param
        env.tn_reward = 1.0
        # Set cost penalty scaling
        env.rho_param = rho_param
        return env
    
    num_envs = 2
    from stable_baselines3.common.vec_env import SubprocVecEnv
    envs = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # Initialize and Train the Model
    from sb3_contrib.ppo_mask import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    rl_model = MaskablePPO(
        MaskableActorCriticPolicy,
        envs,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=rl_device
    )
    rl_model.learn(total_timesteps=rl_timesteps)
    
    # Save the Trained Model
    model_filename = f"ppo_f1_lambda{lambda_param}_rho{rho_param}.pth"
    rl_model.save(model_filename)
    print(f"[PID {os.getpid()}] Finished RL training for lambda={lambda_param}, rho={rho_param}. Saved to {model_filename}")
    return (lambda_param, rho_param, model_filename)

def main():
    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for classifier training:", main_device)

    # Classifier Training
    print("\n==> Training the classifier...")
    data = pd.read_csv("data/nhanes_preprocessed.csv").dropna(subset=['Has_Diabetes'])

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
    feature_cols = [test for panel in test_panels.keys() for test in test_panels[panel]]
    X_full = data[feature_cols].values
    y_full = data["Has_Diabetes"].values

    # Train the imputer once for classifier training
    impute_params = {'batch_size': 256, 'lr': 1e-4, 'alpha': 1e6}
    imputer = Imputer(dim=X_full.shape[1], impute_para=impute_params)
    mask = np.isnan(X_full).astype(int)
    imputer.set_dataset(X_full, mask)
    imputer.train_model(max_iter=50)
    X_imputed = imputer.transform(X_full)
    data[feature_cols] = X_imputed

    # Initialize and train the classifier
    classifier_model = SimpleClassifier(input_dim=X_imputed.shape[1])
    classifier_model = train_classifier(classifier_model, X_imputed, y_full, epochs=50, lr=1e-3, device=main_device)
    init_acc = evaluate_classifier(classifier_model, X_imputed, y_full, device=main_device)
    print("Initial classifier accuracy:", init_acc)

    # Training over Hyperparameters
    print("\n==> Starting sequential RL training for various (lambda, rho) combinations...")
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
    lambda_values = [5, 10]
    rho_values = [0.1, 0.2]
    rl_timesteps = 50000

    rl_results = []
    
    rl_device = torch.device("cpu") 

    for lambda_param in lambda_values:
        for rho_param in rho_values:
            try:
                res = train_rl_policy_for_params(lambda_param, rho_param, policy_kwargs, rl_timesteps, rl_device)
                rl_results.append(res)
            except Exception as e:
                print(f"RL training for lambda={lambda_param}, rho={rho_param} generated an exception: {e}")

    print("\n==> All RL policies trained:")
    for res in rl_results:
        print(f"lambda={res[0]}, rho={res[1]} -> Model File: {res[2]}")


if __name__ == '__main__':
    main()