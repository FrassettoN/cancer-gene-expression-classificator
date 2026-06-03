import os
import pandas as pd

input_folder = os.path.join("data", "raw")
output_folder = os.path.join("data", "processed")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

idx = 1
num_datasets = 17


# 1 - Bladder Urothelial Carcinoma
cancer_type = r"Bladder Urothelial Carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE13507_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'Control' in x else (1 if 'cancer' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE13507_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 2 - Breast invasive carcinoma cancer
cancer_type = r"Breast invasive carcinoma cancer"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE39004-GPL6244_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)   

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'Normal' in x else (1 if 'Tumor' in x else x))
df.loc[df[0] == '!Sample_title', 0] = -1
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.drop(columns=0) 
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE39004_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 3 - Colon adenocarcinoma
cancer_type = r"Colon adenocarcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE41657_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)
 
df.iloc[1, 1:] = df.iloc[1, 1:].apply(lambda x: 0 if 'Normal' in x else (1 if 'Adenocarcinoma' in x else x))
df.iloc[1, 0] = "-1"
df = df.drop(index=[0, 2])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE41657_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 4 - Esophageal carcinoma
cancer_type = r"Esophageal carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE20347_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'carcinoma' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1,2])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE20347_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 5 - Head and Neck squamous cell carcinoma
cancer_type = r"Head and Neck squamous cell carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE6631_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'Normal' in x else (1 if 'Cancer' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1,2])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE6631_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 6 - Kidney Chromophobe
cancer_type = r"Kidney Chromophobe"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE15641_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'chromophobe RCC kidney tumor' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])

label_row = df.iloc[0]
valid_columns = label_row[(label_row == 0) | (label_row == 1)].index
df = df.loc[:, valid_columns]

df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE15641_1_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 7 - Kidney renal clear cell carcinoma
cancer_type = r"Kidney renal clear cell carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE15641_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'clear cell RCC kidney tumor' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])

label_row = df.iloc[0]
valid_columns = label_row[(label_row == 0) | (label_row == 1)].index
df = df.loc[:, valid_columns]

df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE15641_2_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 8 - Kidney renal papillary cell carcinoma
cancer_type = r"Kidney renal papillary cell carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE15641_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'tumor' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE15641_3_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 9 - Liver hepatocellular carcinoma
cancer_type = r"Liver hepatocellular carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE45267_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'Normal' in x else (1 if 'Tumor' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE45267_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 10 - Lung squamous cell carcinoma
cancer_type = r"Lung squamous cell carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE33479_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'carcinoma' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE33479_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 11 - Lung adenocarcinoma
cancer_type = r"Lung adenocarcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE10072_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'Normal' in x else (1 if 'Adenocarcinoma' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE10072_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 12 - Prostate adenocarcinoma
cancer_type = r"Prostate adenocarcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE6919-GPL8300_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'normal' in x else (1 if 'tumor' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE6919_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 13 - Rectum adenocarcinoma
cancer_type = r"Rectum adenocarcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE20842_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[1, 1:] = df.iloc[1, 1:].apply(lambda x: 0 if 'mucosa' in x else (1 if 'tumor' in x else x))
df.iloc[1, 0] = "-1"
df = df.drop(index=[0])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE20842_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 14 - Stomach adenocarcinoma
cancer_type = r"Stomach adenocarcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE2685_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 0 if 'n' in x else (1 if 't' in x else x))
df.iloc[0, 0] = "-1"
df = df.drop(index=[1])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE2685_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 15 - Thyroid carcinoma
cancer_type = r"Thyroid carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE33630_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[0, 1:] = df.iloc[0, 1:].apply(lambda x: 1 if 'tumour' in x else 0)
df.iloc[0, 0] = "-1"
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE33630_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")
idx += 1


# 16 - Uterine Corpus Endometrial Carcinoma
cancer_type = r"Uterine Corpus Endometrial Carcinoma"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"GSE17025_series_matrix_sel.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df.iloc[1, 1:] = df.iloc[1, 1:].apply(lambda x: 1 if 'endometrioid' in x else 0)
df.iloc[1, 0] = "-1"
df = df.drop(index=[0])
mask = df[0].apply(lambda x: isinstance(x, str) and x.startswith('!'))
df = df[~mask]
df = df.drop(columns=0)
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True)

output_file_path = "GSE17025_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")


# 17 - Breast cancer
cancer_type = r"Breast cancer"
print(f"\n[{idx}/{num_datasets}] Processing {cancer_type}")
input_file_path = r"breast_matrix.txt"
input_dataset_path = os.path.join(input_folder, cancer_type, input_file_path)
df = pd.read_csv(input_dataset_path, sep='\t', header=None, low_memory=False)

df = df.drop(index=[0])
df = df.drop(columns=[0])
df = df.reset_index(drop=True)                                
df.columns = range(df.shape[1])
df = df.T
pd.set_option('future.no_silent_downcasting', True)
df = df.replace({',': '.'}, regex=True) 

output_file_path = "breast_matrix_trasp_mod.csv"
output_dataset_path = os.path.join(output_folder, output_file_path)
df.to_csv(output_dataset_path, index=False, header=False)
print(f"[{idx}/{num_datasets}] Successfully processed {cancer_type}")


print(f"\nDataset processing complete [{idx}/{num_datasets}]")

