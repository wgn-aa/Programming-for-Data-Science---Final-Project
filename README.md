# Programming-for-Data-Science---Final-Project

The aim of the project is to perform a data analysis, which includes exploring the dataset, cleaning it, displaying interesting plots and the regression model for trying to predict continuous values.

## Run the project

There are two ways to run the project: 

You can use the **python final_project.py** command to run the main program and see all the important information in the terminal, as well as any interesting plots.

You can also run **streamlit run final_project.py** to get a better presentation of the data analysis.

## The Dataset

The dataset used is 'Artist', this dataset from Kaggle provides data on the 10,000 most listened-to artists in the USA. It is composed of 9 columns/features:
* the artist's name: normally a unique value,
* his identifier, a unique value,
* gender, which can be male, female, other for those who consider themselves neither male nor female and mixed for music groups,
* the artist's age,
* the country he comes from,
* the musical genre,
* popularity rated from 0 to 100,
* number of followers,
* URI

## Projet guideline

### Dataset exploration

The project is divided into several parts. After loading the dataset, we explore the data, looking at:
* the shape,
* what does the data look like (by using head() and tail()),
* dataset info (type, number of null values),
* duplicate values,
* the number of unique values for each column,
* number and percentage of null values per column

### Dataset cleaning

During this exploration, we treat undesirable or odd data.

Indeed, we notice that some columns contain a lot of null values. Since there are too many to just delete the rows with null values, we replace them with the default 'unknow' value.

We also notice that the Name column has a duplicate value, which, given the description of the database, is not normal, so after find the duplicated name, we delete this data.

Some columns don't add value to our analysis, so we delete them from the dataset.

The Age column contains outlier and incoherent data, so we clean it up. Two methods are tried:

1- Delete the outliers, save the mean of this age column, return to the real dataset and replace the outliers with the mean of the previous (cleaned) dataset.

2- Replace the problematic values with samples from a normal distribution

Solution n°2 has been selected

For better visualization, histograms of each step are plotted and displayed.


### Correlation between column

The correlation matrix between each numerical column in the dataset is displayed, along with the correlation graph, so that you can see which columns are most closely related.

### Interesting Plot

Then we plot the graphs that look interesting, some of which are commented on in the terminal.
The different graphs displayed are:
* the distribution of genres in the dataset,
* the 30 countries with the most artists listened-to in the US and their number of artists listened-to in the US,
* the age distribution of artists after the selected cleaning,
* the popularity rate by genre,
* total number of followers for each genre,
* the mean number of followers for each genre, because as seen with the gender distribution, since males have more artists, the total number of followers is biased,
* the average popularity rate for the 30 countries with the most artists listened to in the USA,
* the average popularity rate for the 30 countries with the fewest artists listened to in the USA,
* Popularity rate by number of followers

### The model that explains the data

In our case, we want to predict continuous values, so we've chosen the regression model.

To predict a value (the number of followers as a function of popularity (from 0 to 100)), you can change the popularity_to_predict value to the value you want to predict.

In the terminal, you can observe the Mean absolute error of the test set, which has a rather acceptable value compared with our range of values. 

We also display the Mean absolute error of the training set and see that it's almost identical to that of the test set. This means that our model is well trained (not too much and not too little).

Next, we plot our prediction curve overlayed on the dataset data.