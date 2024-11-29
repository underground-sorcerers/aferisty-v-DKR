import pandas as pd

# script to rewrite time into two integer columns for more effective loading later
time_df = pd.read_csv("dataset_files/credit_card_transactions-ibm_v2.csv", usecols=["Time"])

time_df[["Hour", "Minute"]] = time_df["Time"].str.split(":", expand=True)
time_df["Hour"] = time_df["Hour"].astype('uint8')
time_df["Minute"] = time_df["Minute"].astype('uint8')

time_df[["Hour", "Minute"]].to_csv("dataset_files/hours_and_minutes.csv", index=False)