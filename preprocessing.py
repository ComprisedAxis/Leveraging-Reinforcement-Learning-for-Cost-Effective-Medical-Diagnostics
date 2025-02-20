import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load merged NHANES data
df = pd.read_csv("data/nhanes_diabetes.csv")

# Keep SEQN for reference (but exclude it from model training)
seqn = df[['SEQN']]
df.drop(columns=['SEQN'], inplace=True)  # Remove from features

# Convert categorical variables (One-Hot Encoding)
df['Gender_Male'] = (df['Gender'] == 0).astype(int)
df['Gender_Female'] = (df['Gender'] == 1).astype(int)
df.drop(columns=['Gender'], inplace=True)

# Ensure Age remains an integer (NO SCALING)
age_column = df[['Age']].astype(int)
df.drop(columns=['Age'], inplace=True)

# Ensure Has_Diabetes remains an integer
df['Has_Diabetes'] = df['Has_Diabetes'].astype('Int64') 

# Normalize medical test results (excluding Has_Diabetes)
medical_tests = df.columns.difference(['Has_Diabetes'])
scaler = MinMaxScaler()
df[medical_tests] = scaler.fit_transform(df[medical_tests])

# Add SEQN back for reference
df = pd.concat([seqn, age_column, df[['Gender_Male', 'Gender_Female']], df.drop(columns=['Gender_Male', 'Gender_Female'])], axis=1)

# Ensure Gender columns are integers
df[['Gender_Male', 'Gender_Female']] = df[['Gender_Male', 'Gender_Female']].astype(int)

# Save processed dataset for RL
df.to_csv("data/nhanes_preprocessed.csv", index=False)

print("NHANES data is preprocessed correctly with integer Gender values.")