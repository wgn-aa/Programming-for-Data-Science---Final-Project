import pandas as pd 
import numpy as np 

#Reading Dataset

#Load the data
artists_df = pd.read_csv("Artists.csv")

print("shape: ",artists_df.shape)

#View the data
print("\nhead: \n", artists_df.head())
print("\ntail: \n", artists_df.tail())

#basic information about the dataset
print("\ninfo: \n", artists_df.info())
#Gender, Country have missing values
#Name, Gender, Country, Genres, URI are object datatype
#Age, Popularity, Followers are int64

#Check for the duplicates values
print("\nCheck for Duplication (Number of unique value): \n", artists_df.nunique())
print("\nDuplicate values: \n", artists_df.duplicated().sum()) 
#the function returned ‘0’ --> not a single duplicate value 

#Find null values
print("\nNull values: \n", artists_df.isnull().sum())
percentage_missing_value = (artists_df.isnull().sum()/(len(artists_df)))*100 
#There are some null value in column Gender(17%) and Contry(34%) #!!!!do something with null value

print("\nPercentage of missing values in each column: \n", percentage_missing_value)

#data reduction
#drop bc do not add value to our analysis.
artists_df_reduction = artists_df.drop(['ID'], axis = 1)
print("\ninfo: \n", artists_df_reduction.info())



print("\ndescribe: \n", artists_df_reduction.describe().T)
#age begin at 0, its weird and finish at 149 impossible
#huge difference between the mean of followers and the maximum, maybe the maximum can be removed

print("\nColumn: \n", artists_df_reduction.columns)

#separer peut etre les donnés int des autres données  
#print("\nCorrelation: \n", artists_df_reduction.corr())




