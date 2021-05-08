import numpy as np
import pandas as pd
# read csv
df = pd.read_excel('IT_3.xlsx')
df = df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket','Miles_driven_annually_bucket','credit_history_bucket'], axis=1)
df.fillna(df.mean(), inplace=True)
df.fillna(axis=0, method='ffill',inplace=True)

