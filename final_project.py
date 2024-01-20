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
#There are some null value in column Gender(17%) and Contry(34%) #!!!!do something with null value

percentage_missing_value = (artists_df.isnull().sum()/(len(artists_df)))*100 
print("\nPercentage of missing values in each column: \n", percentage_missing_value)

print("\nNull values for Gender: \n", artists_df[artists_df.Gender.isnull()])
print("\nNull values for Country: \n", artists_df[artists_df.Country.isnull()])



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



#pour supprimer les valeur null d'une colonne
#artists_df.dropna(subset=['Country'], inplace=True)
#print("\ninfo: \n", artists_df.info())

#affiche que les gens qui ont + de 3 ans et moins de 100 ans
normal_age = (artists_df.Age > 3) & (artists_df.Age < 100)

clean_artists_df = artists_df[normal_age]

print("\ninfo: \n", clean_artists_df.describe().T)

print("\nold mean: ",artists_df.Age.mean())
print("new mean: ",clean_artists_df.Age.mean())

#remplacer les valeurs bizarre par la mean
contraire_normal_age = (artists_df.Age < 3.0) | (artists_df.Age > 100.0)

mean_clean_artists_df = artists_df.copy()
mean_clean_artists_df[contraire_normal_age] = clean_artists_df.Age.mean().astype(int)
print("\ninfo: \n", mean_clean_artists_df.describe().T)
print("\nreplace odd value with mean: ",mean_clean_artists_df.Age.mean())


#replace the problematic values with samples from a normal distribution
#faut peut etre remplacer les valeur bizarre par ce tableau
normal_clean = np.random.normal(loc=clean_artists_df.Age.mean(), scale=clean_artists_df.Age.std(), size=artists_df[contraire_normal_age].shape[0])
print("\nreplace problematic values with samples from a normal distribution: ",normal_clean.mean())

