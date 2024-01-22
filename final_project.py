import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
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
#print("\nDuplicate values: \n", artists_df.duplicated().sum()) 
#print("occu: ",artists_df["Country"].value_counts())


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


male=0
female=0
mixed=0
other=0
for i in artists_df.Gender:
    if i == 'male':
        male+=1
    elif i == 'female':
        female+=1
    elif i == 'mixed':
        mixed +=1
    else:
        other +=1

print("male: ",male)
occurrences = {'male': male, 'female': female, 'mixed':mixed, 'unknow':other}

#faire des sous-graph
fig_1 = plt.figure(figsize=(6,6))
ax_1 = fig_1.add_subplot(2, 2, 1) #1er sous graph
ax_2 = fig_1.add_subplot(2, 2, 2) #2e sous graph
ax_3 = fig_1.add_subplot(2, 2, 3)
ax_4 = fig_1.add_subplot(2, 2, 4)

ax_1.pie(list(occurrences.values()), labels=occurrences.keys())
ax_1.set_title('Gender Distribution')

#faire que les 10 plus grand pays
ax_2.bar(artists_df.Country.value_counts().index, artists_df.Country.value_counts())
ax_2.set_title('Distribution by Country')
ax_2.set_xticklabels(ax_2.get_xticklabels(), rotation=90)  # Rotate x-axis labels for better visibility

ax_3.scatter(artists_df.Age, artists_df.Followers)
ax_3.set_title('Age vs. Followers')
ax_3.set_xlabel('Age')
ax_3.set_ylabel('Followers')

ax_4.hist(artists_df.Age, bins=100, color='skyblue', edgecolor='black')
ax_4.set_title('Age Distribution')
ax_4.set_xlabel('Age')
ax_4.set_ylabel('Frequency')

ax_1.legend()
ax_4.legend()
plt.tight_layout()
plt.show()



import seaborn as sns
fig2 = sns.pairplot(clean_artists_df, hue='Gender')
plt.show()
#les deux truc les plus facile à prédire c'est la popularité en fonction des followers (ou le contraire)
#à la limite l'age en fonction des follower mais bof bof enfaite



#defining input features and target variable
X = clean_artists_df["Popularity"]
y = clean_artists_df["Followers"]
#Split for Training and Testing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

##
X_train_df, X_test_df  = pd.DataFrame(X_train),pd.DataFrame(X_test)
##

#Model Development and Evaluation
lin = LinearRegression()

poly = PolynomialFeatures(degree=3)
#met dans un tableau x^0,x^1,x^2.. pour chaque valeur de x_train
X_poly_train = poly.fit_transform(X_train_df)
#X_test_poly = poly.transform(X_test)
##
X_test_poly = poly.fit_transform(X_test_df)

##


#poly.fit(X_poly_train, y_train)
lin = lin.fit(X_poly_train, y_train)
##
coefficient = lin.coef_
intercept = lin.intercept_

x_axis = np.arange(5,110,0.1)
response = intercept + coefficient[1]* x_axis + coefficient[2]*x_axis**2 + coefficient[3]*x_axis**3

plt.scatter(clean_artists_df["Popularity"], clean_artists_df["Followers"], color = 'b')
plt.plot(x_axis, response, color = 'r')
plt.show()


###

y_pred = lin.predict(X_test_poly)
mean_absolute_error(y_test, y_pred)


y_pred_train = lin.predict(X_poly_train)
mean_absolute_error(y_train, y_pred_train)



st.header("Programming for Data Science - Final Project")
st.subheader("Subheader")
st.code('''    
plt.scatter(clean_artists_df["Popularity"], clean_artists_df["Followers"], color = 'b')
plt.plot(x_axis, response, color = 'r')
plt.show()
''')

#sidebar c'est la fenetre sur le coté qu'on ouvre ou qu'on ferme pour gagner de la place
#checkbox, il faut cocher pour que les valeur à l'intérieur de la condition s'affiche
if st.sidebar.checkbox("Show fig"):
    #on peut le faire qu'avec les object qui peuve etre print
    st.write("A __short__ explanation of the project")
    #pas besoin de faire plt.show pour monter les graphique 
    st.write(fig_1)

col_1, col_2 = st.columns(2)
with col_1:
    st.write("Column 1: ")
    st.write(fig_1)
    st.caption("Caption")

with col_2:
    st.write("Column 2: ")
    st.write(fig_1)


