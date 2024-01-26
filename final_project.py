import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st

st.header("Programming for Data Science - Final Project")
st.subheader("Explore the dataset part")
st.write("We'll start by exploring the dataset, finding correlations between attributes and finding some interesting aspects that justify the next parts of the analysis.")


#DATASET EXPLORATION


#Load the data
artists_df = pd.read_csv("Artists.csv")

#we look at the number of observations(rows) and features(columns) in the dataset
print("shape: ",artists_df.shape)
print("There are 9488 observations and 9 features in our dataset")
st.write("The shape of the dataset is: ",artists_df.shape,". We 9488 observations and 9 features in our dataset")

#View the data
print("\nhead: \n", artists_df.head())
print("\ntail: \n", artists_df.tail())
st.write("Here is the dataset: ")
st.write(artists_df)

#basic information about the dataset
artists_df.info()
print("Thanks to this informations, we know that: \n-Gender, Country have missing values. \n-Name, Gender, Country, Genres, URI are object datatype. \n-Age, Popularity, Followers are int64")

#display all the columns
print("\nColumn: \n", artists_df.columns)
col_1, col_2 = st.columns(2)
with col_1:
    st.write("The column are: ",artists_df.columns)

#we drop columns which don't add value to our analysis. (columns that we never use)
artists_df = artists_df.drop(['Genres','URI'], axis = 1)
#display all the columns after dropping
print("\nColumn: \n", artists_df.columns)
with col_2:
    st.write("However, some colonne don't add value to our analysis. So we drop them: ",artists_df.columns)

#Check for the duplicates values
print("\nDuplicate values: \n", artists_df.duplicated().sum()) 
print("the function returned 0 --> not a single duplicate value")
st.write("Let's check the number of duplicated value: ",artists_df.duplicated().sum()," We don't have a single duplicate value !")

#Check the unique value of each column
print("\nNumber of unique value for each column: \n", artists_df.nunique())
print("The column Name have one duplicate value, it's not normal")
st.write("Now, let's see the number of unique value for each column: ", artists_df.nunique(), " The column name have one duplicate value, it's not normal")

#there is 1 duplicated value in Name, it's not normal so lets see which one is it and let's delete the weird one
print(artists_df.loc[artists_df.duplicated(subset=['Name'])])
st.write("Let's see which one is it",artists_df.loc[artists_df.duplicated(subset=['Name'])], " And delete it")
print(artists_df.query('Name== "HANA"'))
artists_df = artists_df.drop(8575)
#print(artists_df.query('Name== "HANA"'))
print(artists_df.loc[artists_df.duplicated(subset=['Name'])])
st.write("So now when we look for a duplicated value in Name column, we get: ",artists_df.loc[artists_df.duplicated(subset=['Name'])])


#Check for null values
print("\nNumber of null values for each column: \n", artists_df.isnull().sum())
col1, col2 = st.columns(2)
with col1:
    st.write("The number of null values for each column: ",artists_df.isnull().sum())

percentage_missing_value = (artists_df.isnull().sum()/(len(artists_df)))*100 
print("\nPercentage of missing values in each column: \n", percentage_missing_value)
print("There are some null value in column Gender(17%) and Contry(34%). But there are too much to just delete it so let's remplace by 'unknow' value" )
with col2:
    st.write("And as a percentage: ",percentage_missing_value)

st.write("We can see that there are a lot of null value in colum Gender(17%) and Country(34%). There are too much null value to just delete it so let's remplace by 'unknow' value")

#print("\nNull values for Gender: \n", artists_df[artists_df.Gender.isnull()])
#print("\nNull values for Country: \n", artists_df[artists_df.Country.isnull()])

artists_df.fillna("unknow", inplace = True)
st.write("So now we don't have anymore null value: ", artists_df.isnull().sum())




#DATASET CLEANING
st.subheader("Clean up the dataset part")


#print the description of the dataset
print("\ndescrition of the dataset artists_df (1): \n", artists_df.describe().T)
print("The value of Age begin at 0, its weird and finish at 149 impossible !")
#huge difference between the mean of followers and the maximum, maybe the maximum can be removed and the mean value is after the 75%
print("\nLet's clean it")
st.write("The description of the dataset is: ", artists_df.describe().T)
st.write("We can see that the value of the column age begin at 0, its weird because we can't sing if we're 0. More over, its finish at 149, impossible. Let's clean it !")



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
plt.show()

st.write("I tried 2 methods to clean it: ")
st.write("Replace all the outliers value by the mean of the clean dataset and we get this new description of the dataset: ",mean_clean_artists_df.describe().T)
st.write("Replace the problematic values with samples from a normal distribution and we get this new description of the dataset: ",normal_clean_artist_df.describe().T)
st.write("We can observe the difference by using these histogram: ",fig_0)


st.subheader("Correlation matrix")
#Let's look at the correlation matrix to see which numerical values are related.
#lets take only numerical value (to avoid error)
numerical_columns = normal_clean_artist_df.select_dtypes(include=['int64']).columns
artists_corr_df = normal_clean_artist_df[numerical_columns].corr()
print("\ncorrelation matrix to see which numerical values are related (1 is really related, 0 is none): \n",artists_corr_df)

#display the correlation matrix as a plot
fig_1 = plt.figure(figsize=(8,6))
sb.heatmap(artists_corr_df, annot=True)
plt.show()
st.write("Now let's see the correlation between each numerical column with the correlation matrix: ",fig_1)
st.write("You can see that most of the columns don't have much to do with each other. But the ones with the most links together are the 'Followers' and 'Popularity' columns.")




#PLOTS
st.subheader("Plot")


fig_2, axes = plt.subplots(3, 3, figsize=(10,8))

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
print("we can see that people who consider themselves neither male nor female have a higher popularity rating than others, followed by men who have a slightly higher rating than women and mixed but for the 3 of them, its almost the same. ")
print("However, when we look at the total number of followers, we see that men are much higher than the others (due to the fact that there are more of them) and that the others have almost none compared to the other categories, ")
print("and when we see the 3rd graph which gives the median number of followers for each gender, we see that it's the 'other' who have a higher median number of followers, followed by women. ")
print("This is because there are very few people in the 'other' category, but they're generally artists with a strong following.")


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
print("We note that for the top 30 countries there are only one or two popularity levels below 40, while for the 30 countries with the fewest artists listened to in the USA, almost half have a popularity level below 40.")


#Plot scatter to show the popularity by number of followers
axes[2,2].scatter(normal_clean_artist_df.Popularity, normal_clean_artist_df.Followers) 
axes[2,2].set_title('Popularity by number of followers')
axes[2,2].set_xlabel('Popularity rate')
axes[2,2].set_ylabel('Number of Followers')


#axes[0,0].legend()
plt.tight_layout()
plt.show()
st.write("Now her you can see some interesting plots: ",fig_2)
st.write("For the **plots 4, 5 and 6**, we can see that people who consider themselves neither male nor female have a higher popularity rating than others, followed by men who have a slightly higher rating than women and mixed but for the 3 of them, its almost the same.")
st.write('''However, when we look at the total number of followers, we see that men are much higher than the others (due to the fact that there are more of them) and that the others have almost none compared to the other categories, 
and when we see the 3rd graph which gives the median number of followers for each gender, we see that it's the "other" who have a higher median number of followers, followed by women.''')
st.write("This is because there are very few people in the 'other' category, but they're generally artists with a strong following.")
st.write("For the **plots 7 and 8**, we note that for the top 30 countries there are only one or two popularity levels below 40, while for the 30 countries with the fewest artists listened to in the USA, almost half have a popularity level below 40.")
         



st.subheader("Plot pairwise relationships in a dataset")

fig_3 = sb.pairplot(normal_clean_artist_df, hue='Gender')
fig_4 = sb.relplot(x="Popularity",y="Followers", hue='Gender', data = normal_clean_artist_df)
plt.show()
st.pyplot(fig_3)
st.write("Let's take a look at the 8th plot")
st.pyplot(fig_4)
st.write("We can see that the data follow a pattern. Let's do a regression model to predict these continuous values.")





# REGRESSION MODEL
st.subheader("The regression model")


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
st.write("The coefficient of the model and the intercept are: ")
st.write("coefficient: ",model.coef_," intercept: ",model.intercept_)


#PREDICTION 

#The model is used to predict values on the test set, 
y_pred = model.predict(X_poly_test)
#calculing the mean of the absolute values of the errors between a model's predictions and the true values (to assess the model's performance on the test set).
mae = mean_absolute_error(y_test, y_pred)
print("\nMean absolute error: ", mae)
print("In my case the Mean Absolute Error is 1 808 622 which is relatively high. However, the followers value range is 115 998 928, so this MAE can be considered acceptable.")
st.write("The **Mean Absolute Error** between a model's predictions and the true values for the test set is: ",mae)
st.write("In my case the Mean Absolute Error is 1 808 622 which is relatively high. However, the followers value range is 115 998 928, so this MAE can be considered acceptable.")


#Lets do the same thing with our training set
y_pred_train = model.predict(X_poly_train)
mae2 =mean_absolute_error(y_train, y_pred_train)
print("Mean absolute error for the training set: ", mae2)
print("The Mean Absolute Error is 1 719 868, we have almost the same than our test set so our model is well train (no overfitting)")
st.write("And for the training set we have: ",mae2)
st.write("The Mean Absolute Error is 1 719 868, we have almost the same than our test set so our model is well train (no overfitting)")


#To predict the number of followers for a specific popularity rate(e.g. 90)
popularity_to_predict = 90
predicted_followers = model.predict(poly.fit_transform([[popularity_to_predict]])) #poly.transform est utilisé pour transformer la nouvelle valeur de popularité en caractéristiques polynomiales compatibles avec le modèle qui a été ajusté précédemment
print("\nPrediction for popularity  ",popularity_to_predict,": ", predicted_followers[0])

slider_value  = st.slider("Choose your popularity rank",6, 100)
predicted_followers2 = model.predict(poly.fit_transform([[slider_value]]))
st.write("The prediction for the number of followers is: ",predicted_followers2)

#PREDICTION PLOT

# Calculating curve values for the x-axis
x_axis = np.arange(6,105,0.1)
#response = intercept + coefficient[1]* x_axis + coefficient[2]*x_axis**2 + coefficient[3]*x_axis**3

#Transforming polynomial features for the entire x-axis and Predicting y values for the entire x-axis using the trained model
x_axis_poly = poly.fit_transform(x_axis.reshape(-1, 1))
y_pred_for_x_axis_value = model.predict(x_axis_poly)

fig_5 = plt.figure(figsize=(8,6))
#Plot the actual data 
plt.scatter(x_train, y_train, color='b', label='Train data')
plt.scatter(x_test, y_test, color='g', label='Test data')
#Plot the predicted model
plt.plot(x_axis, y_pred_for_x_axis_value, color = 'r', label='Training predictions')
plt.xlabel('Popularity')
plt.ylabel('Followers')
plt.legend()
plt.show()

st.write(fig_5)


