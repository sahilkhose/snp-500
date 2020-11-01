import pandas as pd 

df = pd.DataFrame(columns=["model", "all_ones", "acc", "mcc", "f1", "lr"])
df.to_csv("train_data_sheet.csv")
df.to_csv("test_data_sheet.csv")