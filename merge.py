import pandas as pd
import pyreadstat 

# Load Demographics
df_demo, meta_demo = pyreadstat.read_xport("data/P_DEMO.xpt", encoding="latin1")
df_demo = df_demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
df_demo.rename(columns={'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender'}, inplace=True)
df_demo['Gender'] = df_demo['Gender'].replace({1: 0, 2: 1})  # Male: 0, Female: 1
df_demo = df_demo.astype({'SEQN': 'int', 'Age': 'int', 'Gender': 'int'})

# Load Diabetes Diagnosis
df_diq, meta_diq = pyreadstat.read_xport("data/P_DIQ.xpt", encoding="latin1")
df_diq = df_diq[['SEQN', 'DIQ010']]
df_diq.rename(columns={'DIQ010': 'Has_Diabetes'}, inplace=True)
df_diq.replace({".": None}, inplace=True)

# Correct mapping of diabetes categories
df_diq['Has_Diabetes'] = df_diq['Has_Diabetes'].replace({
    1: 1, 2: 0, 3: None, 7: None, 9: None
})
df_diq['Has_Diabetes'] = df_diq['Has_Diabetes'].astype('Int64')

# Load Biochemical Profile
df_bio, meta_bio = pyreadstat.read_xport("data/P_BIOPRO.xpt", encoding="latin1")
df_bio = df_bio[['SEQN', 'LBXSATSI', 'LBXSAL', 'LBXSAPSI', 'LBXSASSI', 'LBXSC3SI', 'LBXSBU', 'LBXSCLSI', 
                 'LBXSCK', 'LBXSCR', 'LBXSGB', 'LBXSGL', 'LBXSGTSI', 'LBXSIR', 'LBXSLDSI', 'LBXSOSSI', 
                 'LBXSPH', 'LBXSKSI', 'LBXSNASI', 'LBXSTB', 'LBXSCA', 'LBXSCH', 'LBXSTP', 'LBXSTR', 'LBXSUA']]
df_bio.fillna(df_bio.mean(), inplace=True)
df_bio = df_bio.astype({'SEQN': 'int'})

# Load Complete Hemochrome (CBC)
df_cbc, meta_cbc = pyreadstat.read_xport("data/P_CBC.xpt", encoding="latin1")
df_cbc = df_cbc[['SEQN', 'LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT', 'LBXBAPCT',
                 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO', 'LBXRBCSI', 'LBXHGB',
                 'LBXHCT', 'LBXMCVSI', 'LBXMC', 'LBXMCHSI', 'LBXRDW', 'LBXPLTSI', 'LBXMPSI', 'LBXNRBC']]
df_cbc.fillna(df_cbc.mean(), inplace=True)
df_cbc = df_cbc.astype({'SEQN': 'int'})

# Load Fasting Glucose
df_glu, meta_glu = pyreadstat.read_xport("data/P_GLU.xpt", encoding="latin1")
df_glu = df_glu[['SEQN', 'LBXGLU']]
df_glu.fillna(df_glu.mean(), inplace=True)
df_glu = df_glu.astype({'SEQN': 'int'})

# Load Glycohemoglobin (HbA1c)
df_gly, meta_gly = pyreadstat.read_xport("data/P_GHB.xpt", encoding="latin1")
df_gly = df_gly[['SEQN', 'LBXGH']]
df_gly.fillna(df_gly.mean(), inplace=True)
df_gly = df_gly.astype({'SEQN': 'int'})

# Load Body Mass Index (BMI) and Body Composition
df_bmx, meta_bmx = pyreadstat.read_xport("data/P_BMX.xpt", encoding="latin1")
df_bmx = df_bmx[['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXHIP']]
df_bmx.fillna(df_bmx.mean(), inplace=True)
df_bmx = df_bmx.astype({'SEQN': 'int'})

# Load Triglycerides & LDL
df_trig, meta_trig = pyreadstat.read_xport("data/P_TRIGLY.xpt", encoding="latin1")
df_trig = df_trig[['SEQN', 'LBXTR', 'LBDLDL']]
df_trig.fillna(df_trig.mean(), inplace=True)
df_trig = df_trig.astype({'SEQN': 'int'})

# Load HDL Cholesterol
df_hdl, meta_hdl = pyreadstat.read_xport("data/P_HDL.xpt", encoding="latin1")
df_hdl = df_hdl[['SEQN', 'LBDHDD']]
df_hdl.fillna(df_hdl.mean(), inplace=True)
df_hdl = df_hdl.astype({'SEQN': 'int'})

# Load Total Cholesterol
df_chol, meta_chol = pyreadstat.read_xport("data/P_TCHOL.xpt", encoding="latin1")
df_chol = df_chol[['SEQN', 'LBXTC']]
df_chol.fillna(df_chol.mean(), inplace=True)
df_chol = df_chol.astype({'SEQN': 'int'})

# Load Insulin
df_ins, meta_ins = pyreadstat.read_xport("data/P_INS.xpt", encoding="latin1")
df_ins = df_ins[['SEQN', 'LBXIN']]
df_ins.fillna(df_ins.mean(), inplace=True)
df_ins = df_ins.astype({'SEQN': 'int'})

# Load HS C-Reactive Protein (CRP)
df_crp, meta_crp = pyreadstat.read_xport("data/P_HSCRP.xpt", encoding="latin1")
df_crp = df_crp[['SEQN', 'LBXHSCRP']]
df_crp.fillna(df_crp.mean(), inplace=True)
df_crp = df_crp.astype({'SEQN': 'int'})

# Merge all datasets using SEQN
df = (df_demo
      .merge(df_bio, on="SEQN", how="left")
      .merge(df_cbc, on="SEQN", how="left")
      .merge(df_glu, on="SEQN", how="left")
      .merge(df_gly, on="SEQN", how="left")
      .merge(df_bmx, on="SEQN", how="left")
      .merge(df_trig, on="SEQN", how="left")
      .merge(df_hdl, on="SEQN", how="left")
      .merge(df_chol, on="SEQN", how="left")
      .merge(df_ins, on="SEQN", how="left")
      .merge(df_crp, on="SEQN", how="left")
      .merge(df_diq, on="SEQN", how="left"))

# Fill missing values for numerical columns but exclude Has_Diabetes
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns.difference(['Has_Diabetes'])

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Final dataset preview
print("Final dataset preview:")
print(df.head())
print(df.info())

# Save dataset
df.to_csv("data/nhanes_diabetes.csv", index=False)