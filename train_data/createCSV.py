# %%
import pandas as pd

df = pd.read_csv('./raw/vott-csv-export/bar-tracking-export.csv')
# %%

df[round(df.iloc[:, 1:5] , 3) == 0.0]

# %%
