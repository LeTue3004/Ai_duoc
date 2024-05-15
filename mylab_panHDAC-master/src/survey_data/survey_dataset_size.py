import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from common.pharmacy_common import PharmacyCommon
from sklearn.metrics import accuracy_score, f1_score
import os

# Configs
train_path          = "/home/mrcong/code/python/mylab-panHDAC/data/survey/dataset_size/subsampled_train_dataset.csv"
val_path            = "../../data/train_test_data/NoCL/20240307_pan_HDAC_train_test_data.xlsx"
result_path         = "/home/mrcong/code/python/mylab-panHDAC/results/survey/data_size"
fpts_name           = "ECFP4"
bits                = 2048
detail_pred_path    = os.path.join(result_path, f"20240313_{fpts_name}_{bits}_detail_pred_result.xlsx")
sum_pred_path       = os.path.join(result_path, f"20240313_{fpts_name}_{bits}_sum_pred_result.xlsx")
sum_pred_fig_path   = os.path.join(result_path, f"20240313_{fpts_name}_{bits}_sum_pred_fig.png")

# Init
common = PharmacyCommon()
train_datasets = pd.read_csv(train_path)
validation_dataset = pd.read_excel(val_path, sheet_name='validation_dataset')
def gen_fpts(data) -> np.ndarray:
    return common.gen_ecfp4_fpts(data=data, bits=bits)

#Encoding data
#X data
X_Validation = gen_fpts(validation_dataset['SMILES'])
y_Validation = np.array(validation_dataset['Bioactivity'])

#Y data
#Original data
print("Original data:")
print(y_Validation[0:5])
#Encoding labels
label_encoder = preprocessing.LabelEncoder()
y_Validation = label_encoder.fit_transform(y_Validation)
#Class encoded
print("Class encoded:")
print(list(label_encoder.classes_))
print(label_encoder.transform(label_encoder.classes_))
print("Encoded data:")
print(y_Validation[0:5])

#Details predictions
print("[+] Making predictions")
result_dict = {
    "Model name": [],
    "Fingerprint": [],
    "Val_ACC": [],
    "Val_F1": [],
    "Subsample_size": [],
    "Random_seed": []
}
for subsample_size in np.round(np.linspace(0.05, 0.95, num=19), decimals=2):    
    for seed in range(10): 
        filtered_df = train_datasets[(train_datasets["subsample_size"] == subsample_size) & (train_datasets["rand_seed"] == seed)]
        ss_model = RandomForestClassifier(n_estimators=200, criterion="gini", random_state=42)
        X_train = gen_fpts(filtered_df["SMILES"])
        y_train = label_encoder.transform(filtered_df["Bioactivity"])
        ss_model.fit(X_train, y_train)
        y_hat = ss_model.predict(X_Validation)
        
        result_dict["Model name"].append("Random Forest")
        result_dict["Fingerprint"].append("Morgan2")
        result_dict["Val_ACC"].append(accuracy_score(y_true=y_Validation, y_pred=y_hat))
        result_dict["Val_F1"].append(f1_score(y_true=y_Validation, y_pred=y_hat))
        result_dict["Subsample_size"].append(subsample_size)
        result_dict["Random_seed"].append(seed)
        
result_df = pd.DataFrame(result_dict)
result_df.to_excel(detail_pred_path, index=False)
print(f"File saved to {detail_pred_path}")

# Summary dataset
print("[+] Summary data")
summary_dict = {
    "Model name": [],
    "Fingerprint": [],
    "mean_val_ACC": [],
    "std_val_ACC": [],
    "mean_val_F1": [],
    "std_val_F1": [],
    "subsample_size": []
}
for subsample_size in result_df["Subsample_size"].unique():
    filtered_df = result_df[(result_df["Subsample_size"] == subsample_size)]
    summary_dict["Model name"].append(result_df.loc[:,"Model name"].unique()[0])
    summary_dict["Fingerprint"].append(result_df.loc[:,"Fingerprint"].unique()[0])
    summary_dict["mean_val_ACC"].append(np.mean(filtered_df["Val_ACC"]))
    summary_dict["std_val_ACC"].append(np.std(filtered_df["Val_ACC"]))
    summary_dict["mean_val_F1"].append(np.mean(filtered_df["Val_F1"]))
    summary_dict["std_val_F1"].append(np.std(filtered_df["Val_F1"]))
    summary_dict["subsample_size"].append(subsample_size)
summary_df = pd.DataFrame(summary_dict)
summary_df.to_excel(sum_pred_path, index=False)
print(f"File saved to {sum_pred_path}")

# Figure plot
# Set up the plot
print("Plot the summary data")
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df["subsample_size"], summary_df["mean_val_ACC"], yerr=summary_df["std_val_ACC"], marker="o", label="Accuracy")
plt.errorbar(summary_df["subsample_size"], summary_df["mean_val_F1"], yerr=summary_df["std_val_F1"], marker="s", label="F1 Score")

# Customize the plot
plt.xlabel("Subsample Size")
plt.ylabel("Performance")
plt.title(f"Random Forest model performance with different subsample size ({fpts_name} - {bits} bits)")
plt.grid(True)
plt.xticks(np.arange(0, 1, 0.05))  # Set custom y-axis ticks
plt.legend()
# Show the plot
plt.savefig(sum_pred_fig_path)
print(f"File saved to {sum_pred_fig_path}")


