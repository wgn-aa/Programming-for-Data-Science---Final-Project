import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st

#Reading Dataset

#Load the data
artists_df = pd.read_csv("Artists.csv")

#we look at the number of observations(rows) and features(columns) in the dataset
print("shape: ",artists_df.shape)
#There are 9488 observations and 9 features in our dataset

#View the data
print("\nhead: \n", artists_df.head())
print("\ntail: \n", artists_df.tail())

#basic information about the dataset
artists_df.info()
#Gender, Country have missing values
#Name, Gender, Country, Genres, URI are object datatype
#Age, Popularity, Followers are int64

#display all the columns
print("\nColumn: \n", artists_df.columns)

#we drop columns which don't add value to our analysis. (columns that we never use)
artists_df = artists_df.drop(['Genres','URI'], axis = 1)
#display all the columns after dropping
print("\nColumn: \n", artists_df.columns)


#Check for the duplicates values
print("\nDuplicate values: \n", artists_df.duplicated().sum()) 
#the function returned ‘0’ --> not a single duplicate value 

#Check the unique value of each column
print("\nNumber of unique value for each column: \n", artists_df.nunique())
#name have one duplicate value
#id only unique value, uri also, it is interesting ?

#there is 1 duplicated value in Name, it's not normal so lets see which one is it and let's delete the weird one
print(artists_df.loc[artists_df.duplicated(subset=['Name'])])
print(artists_df.query('Name== "HANA"'))
artists_df = artists_df.drop(8575)
#print(artists_df.query('Name== "HANA"'))
print(artists_df.loc[artists_df.duplicated(subset=['Name'])])


#Check for null values
print("\nNumber of null values for each column: \n", artists_df.isnull().sum())

percentage_missing_value = (artists_df.isnull().sum()/(len(artists_df)))*100 
print("\nPercentage of missing values in each column: \n", percentage_missing_value)
#There are some null value in column Gender(17%) and Contry(34%)
#there are too much null value to just delete it so let's remplace by "unknow" value
#EST CE QUE JE LES SUPPRIME ????

#print("\nNull values for Gender: \n", artists_df[artists_df.Gender.isnull()])
#print("\nNull values for Country: \n", artists_df[artists_df.Country.isnull()])
#pour supprimer les valeur null d'une colonne
#artists_df.dropna(subset=['Country'], inplace=True)
artists_df.fillna("unknow", inplace = True)

#############################################################################CHOISIR ENTRE LES 2 METHODES

#print the description of the dataset
print("\ndescrition of the dataset artists_df (1): \n", artists_df.describe().T)
#age begin at 0, its weird and finish at 149 impossible
#huge difference between the mean of followers and the maximum, maybe the maximum can be removed and the mean value is after the 75%
#lets clean it
#EST CE QUE JE SUPPRIME CEUX AVEC 0 DE POPOULARITE MSKN


#mask to have only person are more than 5 and les than 100
#so we can have a mean closer to reality
normal_age = (artists_df.Age > 10) & (artists_df.Age < 80)
clean_artists_df = artists_df[normal_age]

print("\ndescription of the dataset without outliers value (2): \n", clean_artists_df.describe().T)
#the mean of the age change from 18 to 33


#we replace all the outliers value by the mean of the clean dataset 
#to treat outliers in such a way that they do not disproportionately affect descriptive statistics or mean-based analyses.
outliers_age_value = (artists_df.Age < 10.0) | (artists_df.Age > 80.0)
mean_clean_artists_df = artists_df.copy()
mean_clean_artists_df[outliers_age_value] = clean_artists_df.Age.mean().astype(int)

print("\ndescription of the dataset with outliers values replaced by the mean of the clean dataset (3): \n", mean_clean_artists_df.describe().T)


#replace the problematic values with samples from a normal distribution
normal_clean = np.random.normal(loc=clean_artists_df.Age.mean(), scale=clean_artists_df.Age.std(), size=artists_df[outliers_age_value].shape[0])

# To avoid having people with a negative age
minimum_value = 5  
normal_clean = np.maximum(normal_clean, minimum_value)

normal_clean_artist_df = artists_df.copy()
normal_clean_artist_df.loc[outliers_age_value, 'Age'] = normal_clean.astype(int)

print("\ndescription of the dataset with outliers values replaced by samples from a normal distribution (4): \n", normal_clean_artist_df.describe().T)


#display display histograms of artist age frequencies 
#A histogram for each treatment effected on the age column to see changes 
fig_0 = plt.figure(figsize=(11,6))
ax_1 = fig_0.add_subplot(2, 2, 1) 
ax_2 = fig_0.add_subplot(2, 2, 2)
ax_3 = fig_0.add_subplot(2, 2, 3)
ax_4 = fig_0.add_subplot(2, 2, 4)

ax_1.hist(artists_df.Age, bins=100, color='skyblue', edgecolor='black')
ax_1.set_title('Before any cleaning')

ax_2.hist(clean_artists_df.Age, bins=100, color='skyblue', edgecolor='black')
ax_2.set_title('Only by removing outliers value')

ax_3.hist(mean_clean_artists_df.Age, bins=100, color='skyblue', edgecolor='black')
ax_3.set_title('By replacing outliers value with mean ')

ax_4.hist(normal_clean_artist_df.Age, bins=100, color='skyblue', edgecolor='black')
ax_4.set_title('replacing value with samples from a normal distribution')

fig_0.suptitle('Frequency of each age for artists', fontsize=16)
plt.tight_layout()
#plt.show()


#Let's look at the correlation matrix to see which numerical values are related.
#lets take only numerical value (to avoid error)
numerical_columns = normal_clean_artist_df.select_dtypes(include=['int64']).columns
artists_corr_df = normal_clean_artist_df[numerical_columns].corr()
print("\ncorrelation matrix to see which numerical values are related (1 is really related, 0 is none): \n",artists_corr_df)

#display the correlation matrix as a plot
plt.figure(figsize=(8,6))
sb.heatmap(artists_corr_df, annot=True)
#plt.show()



fig_1, axes = plt.subplots(3, 3, figsize=(10,8))

#calculating the occurrence of each gender
male=0
female=0
mixed=0
other=0
unknow =0
for i in normal_clean_artist_df.Gender:
    if i == 'male':
        male+=1
    elif i == 'female':
        female+=1
    elif i == 'mixed':
        mixed +=1
    elif i == 'other':
        other +=1
    else:
        unknow +=1

occurrences = {'male': male, 'female': female, 'mixed':mixed, 'other':other, 'unknow':unknow}

#plot a pie to show the distribution of gender
axes[0,0].pie(list(occurrences.values()), labels=occurrences.keys())
axes[0,0].set_title('Gender Distribution')


#create a mask to not count unknow value
mask_country = normal_clean_artist_df.Country != 'unknow'
artist_filtered_country_df = normal_clean_artist_df[mask_country]

#plot bar to show the number of countries among the 30 with the most artists
artist_filtered_country_df.Country.value_counts().nlargest(32).plot(kind='bar',ax=axes[0,1])
#axes[0,1].bar(top_30_players.index, top_30_players)
#axes[0,1].bar(artist_filtered_country_df.Country.value_counts().index, artist_filtered_country_df.Country.value_counts())
axes[0,1].set_title('Distribution in the top 30 countries')
axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(),rotation=90, fontsize=6)# Rotate x-axis labels for better visibility


#plot an histogram tio show the distribution of age cleaned previously
normal_clean_artist_df.Age.hist(ax=axes[0,2],bins=100)
axes[0,2].set_title('Age Distribution')


#create a mask to not count unknow value
mask_gender = normal_clean_artist_df.Gender != 'unknow'
artist_filtered_gender_df = normal_clean_artist_df[mask_gender]
#group the data by Gender then calculate the mean of the 'Popularity' column for each gender
popu_per_gender = artist_filtered_gender_df.groupby('Gender')['Popularity'].mean().reset_index()

#plot bar to show the popularity rate for each gender
axes[1,0].bar(popu_per_gender["Gender"],popu_per_gender["Popularity"])
axes[1,0].set_title('Popularity by gender')
axes[1,0].set_xlabel('Gender')
axes[1,0].set_ylabel('Popularity rate')


#group the data by Gender then calculate the mean and the sum of the number of Followers for each gender
followers_per_gender = artist_filtered_gender_df.groupby('Gender')['Followers'].mean().reset_index()
followers_per_gender2 = artist_filtered_gender_df.groupby('Gender')['Followers'].sum().reset_index()

##plot bar to show the total number of followers for each gender
axes[1,1].bar(followers_per_gender2['Gender'],followers_per_gender2['Followers'])
axes[1,1].set_title('Total number of followers by gender')
axes[1,1].set_xlabel('Gender')
axes[1,1].set_ylabel('Number of followers')

#plot bar to show the mean of numbers of followers for each gender
axes[1,2].bar(followers_per_gender['Gender'],followers_per_gender['Followers'])
axes[1,2].set_title('Mean number of followers by gender')
axes[1,2].set_xlabel('Gender')
axes[1,2].set_ylabel('Number of followers')
#we can see that people who consider themselves neither male nor female have a higher popularity rating than others, followed by men who have a slightly higher rating than women and mixed but for the 3 of them, its almost the same. 
#However, when we look at the total number of followers, we see that men are much higher than the others (due to the fact that there are more of them) and that the others have almost none compared to the other categories, 
#and when we see the 3rd graph which gives the median number of followers for each gender, we see that it's the "other" who have a higher median number of followers, followed by women. 
#This is because there are very few people in the "other" category, but they're generally artists with a strong following.


'''
artist_filtered_country_df.groupby("Gender").mean().sort_values(by= artist_filtered_country_df['Country'].value_counts(), ascending = False).head(30)
artists_df.groupby(['Country', 'Name']).mean()
'''
# Count occurrences number by country 
occurrences_by_country = artist_filtered_country_df['Country'].value_counts().reset_index()
occurrences_by_country.columns = ['Country', 'occurrences']
# select most and less frequent country
top_30_country = occurrences_by_country.head(32)['Country']
flop_30_country = occurrences_by_country.tail(32)['Country']
# Filter the dataframe
df_top_10 = artist_filtered_country_df[artist_filtered_country_df['Country'].isin(top_30_country)]
df_flop_10 = artist_filtered_country_df[artist_filtered_country_df['Country'].isin(flop_30_country)]
# Calcul of the mean of popularity for each country
popularity_by_country = df_top_10.groupby('Country')['Popularity'].mean().reset_index()
popularity_by_country2 = df_flop_10.groupby('Country')['Popularity'].mean().reset_index()

#plot bar to show the popularity rate for the top 30 country with the most artists
axes[2,0].bar(popularity_by_country["Country"],popularity_by_country["Popularity"])
axes[2,0].set_title('Popularity of the top 30 countries')
axes[2,0].set_xticklabels(axes[2,0].get_xticklabels(),rotation=90, fontsize=6)

#plot bar to show the popularity rate for the top 30 country with the less artists
axes[2,1].bar(popularity_by_country2["Country"],popularity_by_country2["Popularity"])
axes[2,1].set_title('Popularity of the last 30 countries')
axes[2,1].set_xticklabels(axes[2,1].get_xticklabels(),rotation=90, fontsize=6)
#We note that for the top 30 countries there are only one or two popularity levels below 40, 
#while for the 30 countries with the fewest artists listened to in the USA, almost half have a popularity level below 40.


#Plot scatter to show the popularity by number of followers
axes[2,2].scatter(normal_clean_artist_df.Popularity, normal_clean_artist_df.Followers) 
axes[2,2].set_title('Popularity by number of followers')
axes[2,2].set_xlabel('Popularity rate')
axes[2,2].set_ylabel('Number of Followers')


#axes[0,0].legend()
plt.tight_layout()
#plt.show()




fig2 = sb.pairplot(normal_clean_artist_df, hue='Gender')
sb.relplot(x="Popularity",y="Followers", hue='Gender', data = normal_clean_artist_df)
#plt.show()
#les deux truc les plus facile à prédire c'est la popularité en fonction des followers (ou le contraire)
#à la limite l'age en fonction des follower mais bof bof enfaite




#Split for Training and Testing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_absolute_error


#DATA

#defining input features and target variable
x = clean_artists_df["Popularity"]
y = clean_artists_df["Followers"]

#Divide data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)
X_train_df, X_test_df  = pd.DataFrame(x_train),pd.DataFrame(x_test)


#TRAINING

#Feature(X_train_df and X_test_df) transformation with degree 3 polynomial regression :
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train_df) #met dans un tableau x^0,x^1,x^2.. pour chaque valeur de x_train
X_poly_test = poly.fit_transform(X_test_df)

#Training the linear regression model with train polynomial features
model = LinearRegression()
model = model.fit(X_poly_train, y_train)

print("\nCoefficient of the model: ",model.coef_)
print("Intercept of the model: ",model.intercept_)


#PREDICTION 

#The model is used to predict values on the test set, 
y_pred = model.predict(X_poly_test)
#calculing the mean of the absolute values of the errors between a model's predictions and the true values (to assess the model's performance on the test set).
mae = mean_absolute_error(y_test, y_pred)
print("\nMean absolute error: ", mae)
#In my case the Mean Absolute Error is 1 808 622 which is relatively high. However, the followers value range is 115 998 928, so this MAE can be considered acceptable.


#Lets do the same thing with our training set
y_pred_train = model.predict(X_poly_train)
mae2 =mean_absolute_error(y_train, y_pred_train)
print("Mean absolute error for the training set: ", mae2)
#the Mean Absolute Error is 1 719 868, we have almost the same than our test set so our model is well train (no overfitting)


#To predict the number of followers for a specific popularity rate(e.g. 90)
popularity_to_predict = 90
predicted_followers = model.predict(poly.fit_transform([[popularity_to_predict]])) #poly.transform est utilisé pour transformer la nouvelle valeur de popularité en caractéristiques polynomiales compatibles avec le modèle qui a été ajusté précédemment
print("\nPrediction for popularity  ",popularity_to_predict,": ", predicted_followers[0])


#PREDICTION PLOT

# Calculating curve values for the x-axis
x_axis = np.arange(5,x_train.max(),0.1)
#response = intercept + coefficient[1]* x_axis + coefficient[2]*x_axis**2 + coefficient[3]*x_axis**3

#Transforming polynomial features for the entire x-axis and Predicting y values for the entire x-axis using the trained model
x_axis_poly = poly.fit_transform(x_axis.reshape(-1, 1))
y_pred_for_x_axis_value = model.predict(x_axis_poly)

#fig7 = plt.figure(figsize=(8,6))
#Plot the actual data 
plt.scatter(x_train, y_train, color='b', label='Train data')
plt.scatter(x_test, y_test, color='g', label='Test data')
#Plot the predicted model
plt.plot(x_axis, y_pred_for_x_axis_value, color = 'r', label='Training predictions')
plt.xlabel('Popularity')
plt.ylabel('Followers')
plt.legend()
#plt.show()



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


#si on utilise le graphe de la frequence des followers, faire un nettoyage des trop grande données