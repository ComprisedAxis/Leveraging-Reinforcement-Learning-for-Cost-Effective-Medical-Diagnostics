import gym
import numpy as np
import pandas as pd
from gym import spaces

class DiabetesDiagnosisEnv(gym.Env):
    """
    Environment for cost-sensitive diagnostic decision-making.
    
    Actions:
      - 0 to (N-1): Order a test panel.
      - N: Diagnose Diabetes.
      - N+1: Diagnose Non-Diabetes.
    
    Observations:
      A vector of imputed test values for all tests (in a fixed order)
      concatenated with a diagnosis flag (1 if a diagnosis has been made, 0 otherwise).
    
    Rewards:
      - Ordering a test: incurs a cost penalty = ρ * (panel_cost) scaled by a dynamic factor.
      - When diagnosing:
          • A correct diagnosis yields a reward (tp_reward for Diabetes or tn_reward for Non-Diabetes).
          • An incorrect diagnosis yields the negative of that reward.
          • If fewer than min_tests_required panels have been ordered, an adaptive penalty is applied.
          • If more panels than required are ordered, a bonus is added.
    """
    
    def __init__(self, data, imputer,
                 early_diagnosis_penalty=10.0,  # full penalty if no tests ordered
                 min_tests_required=2,         # target number of tests before penalty is fully waived
                 gamma=0.1,                    # dynamic cost scaling factor: extra cost per extra test
                 bonus_factor=0.5              # bonus per extra test ordered beyond the minimum
                 ):
        super(DiabetesDiagnosisEnv, self).__init__()
        
        # Data and imputer.
        self.data = data
        self.imputer = imputer
        
        # Diagnosis rewards
        self.tp_reward = 1.0  
        self.tn_reward = 1.0  
        self.lambda_param = None  # set tp_reward
        self.rho_param = None     # scales the test cost penalty
        
        # Parameters for trade-off between cost and accuracy
        self.early_diagnosis_penalty = early_diagnosis_penalty
        self.min_tests_required = min_tests_required
        self.gamma = gamma # additional cost scaling factor per extra test ordered
        self.bonus_factor = bonus_factor  # bonus for each test beyond the minimum
        
        # Panels and their costs
        self.panel_costs = {
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
        self.test_panels = {
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
        
        # Fixed ordering of panels and tests
        self.available_panels = list(self.test_panels.keys())
        self.all_tests = []
        self.panel_of_test = {}
        for panel in self.available_panels:
            tests = self.test_panels[panel]
            self.all_tests.extend(tests)
            for test in tests:
                self.panel_of_test[test] = panel
        
        # Tracking state
        self.panels_taken = {panel: False for panel in self.available_panels}
        self.diagnosis_made = False
        
        # Define action and observation spaces
        self.num_panels = len(self.available_panels)
        self.action_space = spaces.Discrete(self.num_panels + 2)
        obs_dim = len(self.all_tests) + 1  # tests (imputed) + diagnosis flag
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def reset(self):
        self.current_patient = self.data.sample(1).to_dict(orient="records")[0]
        self.panels_taken = {panel: False for panel in self.available_panels}
        self.diagnosis_made = False
        return self._get_observation()
    
    def _get_observation(self):
        test_values = []
        for test in self.all_tests:
            panel = self.panel_of_test[test]
            # Use the actual patient test value if the panel was ordered; otherwise np.nan
            value = self.current_patient.get(test, np.nan) if self.panels_taken[panel] else np.nan
            test_values.append(value)
        test_values = np.array(test_values, dtype=np.float32).reshape(1, -1)
        # Impute missing values
        imputed_values = self.imputer.transform(test_values).flatten()
        diag_flag = 1.0 if self.diagnosis_made else 0.0
        state = np.concatenate([imputed_values, [diag_flag]])
        assert state.shape[0] == self.observation_space.shape[0], \
            f"Observation dimension mismatch: expected {self.observation_space.shape[0]}, got {state.shape[0]}"
        return state
    
    def step(self, action):
        reward = 0.0
        done = False
        
        if action < self.num_panels:
            # Order a test panel
            panel = self.available_panels[action]
            if not self.panels_taken[panel]:
                self.panels_taken[panel] = True
                tests_ordered = sum(self.panels_taken.values())
                # Dynamic cost scaling: later tests cost more
                scaling = 1 + self.gamma * (tests_ordered - 1)
                reward += self.rho_param * self.panel_costs[panel] * scaling
            done = False
        
        elif action == self.num_panels:
            # Diagnose as Diabetes
            tests_ordered = sum(self.panels_taken.values())
            if tests_ordered < self.min_tests_required:
                penalty = self.early_diagnosis_penalty * (self.min_tests_required - tests_ordered) / self.min_tests_required
                reward -= penalty
            else:
                bonus = self.bonus_factor * max(0, tests_ordered - self.min_tests_required)
                reward += bonus
            true_label = self.current_patient["Has_Diabetes"]
            reward += self.tp_reward if true_label == 1 else -self.tp_reward
            self.diagnosis_made = True
            done = True
        
        elif action == self.num_panels + 1:
            # Diagnose as Non-Diabetes
            tests_ordered = sum(self.panels_taken.values())
            if tests_ordered < self.min_tests_required:
                penalty = self.early_diagnosis_penalty * (self.min_tests_required - tests_ordered) / self.min_tests_required
                reward -= penalty
            else:
                bonus = self.bonus_factor * max(0, tests_ordered - self.min_tests_required)
                reward += bonus
            true_label = self.current_patient["Has_Diabetes"]
            reward += self.tn_reward if true_label == 0 else -self.tn_reward
            self.diagnosis_made = True
            done = True
        
        else:
            raise ValueError("Invalid action")
        
        obs = self._get_observation()
        return obs, reward, done, {}
    
    def action_masks(self):
        mask = [not self.panels_taken[panel] for panel in self.available_panels] + [True, True]
        return mask
    
    def render(self, mode='human'):
        print("Current Patient:", self.current_patient)
        print("Panels Taken:", self.panels_taken)
        print("Diagnosis Made:", self.diagnosis_made)