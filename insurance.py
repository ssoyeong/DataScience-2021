import pandas as pd
import numpy as np
import seaborn as sns
import matplot.pyplot as plt

df = pd.read_excel('/Users/kohyojin/Desktop/데스크탑/SoftWare/software2021/data science/term_project/IT_3.xlsx')
df=df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket','Miles_driven_annually_bucket'])
df = df.fillna(method='ffill')
