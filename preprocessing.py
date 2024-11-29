import pandas as pd


def currency_to_integer_cents(amount):
    return int(amount.replace("$","").replace(".",""))


def yes_no_to_bool(value):
    return value == "Yes"


importing_columns = [
    'User',
    'Card',
    'Year',
    'Month',
    'Day',
    # 'Time',
    'Amount',
    'Use Chip',
    'Merchant Name',
    # 'Merchant City',
    # 'Merchant State',
    # 'Zip',
    'MCC',
    'Errors?',
    'Is Fraud?']

dtype_conversions = {
    'User': 'uint16',
    'Card': 'uint8',
    'Year': 'uint16',
    'Month': 'uint8',
    'Day': 'uint8',
    # 'Time', -- dropped
    # 'Amount',
    'Use Chip': 'category',
    # 'Merchant Name',
    # 'Merchant City': 'category', -- dropped
    # 'Merchant State': 'category', -- dropped
    # 'Zip', -- dropped
    'MCC': 'uint16',
    'Errors?': 'category',
    # 'Is Fraud?'
}

time_dtype_conversions = {
    "Hour": "uint8",
    "Minute": "uint8"
}

arbitrary_conversions = {
    # 'User': 'uint16',
    # 'Card': 'uint8',
    # 'Year',
    # 'Month': 'uint8',
    # 'Day': 'uint8',
    # 'Time', -- dropped
    'Amount': currency_to_integer_cents,
    # 'Use Chip': 'category',
    # 'Merchant Name',
    # 'Merchant City': 'category', -- dropped
    # 'Merchant State': 'category', -- dropped
    # 'Zip', -- dropped
    # 'MCC': 'uint16',
    # 'Errors?': 'category',
    'Is Fraud?': yes_no_to_bool
}


df = pd.read_csv("dataset_files/credit_card_transactions-ibm_v2.csv",
                 dtype=dtype_conversions, converters=arbitrary_conversions, usecols=importing_columns)
df['Amount'] = df["Amount"].astype("int32")
time_df = pd.read_csv("dataset_files/hours_and_minutes.csv", dtype=time_dtype_conversions)

df["Hour"] = time_df["Hour"]
df["Minute"] = time_df["Minute"]

df = df[df["Use Chip"] == "Online Transaction"].drop("Use Chip", axis=1)
df = df[(df["Year"] == 2015) | (df["Year"] == 2016)]

df.to_csv("dataset_files/preprocessed_transactions.csv", index=False)
