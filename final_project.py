import pandas as pd 
import numpy as np 

#Reading Dataset
artists_df = pd.read_csv("Artists.csv")

print("shape: ",artists_df.shape)

print("\nhead: \n", artists_df.head())
print("\ntail: \n", artists_df.tail())

print("\ninfo: \n", artists_df.info())
#Gender, Country have missing values

print("\nCheck for Duplication (Number of unique value): \n", artists_df.nunique())

print("\nNull values: \n", artists_df.isnull().sum())

percentage_missing_value = (artists_df.isnull().sum()/(len(artists_df)))*100

print("\nPercentage of missing values in each column: \n", percentage_missing_value)

#data reduction
#drop bc do not add value to our analysis.
artists_df_reduction = artists_df.drop(['ID'], axis = 1)
print("\ninfo: \n", artists_df_reduction.info())


