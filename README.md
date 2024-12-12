# 1. Introduction

This project aims to predict the California region (i.e. Superior Counties, Bay Area Counties, Central Counties, Southern Counties, Los Angeles Counties) in which an establishment is based using features such as number of establishments, establishment sector, average weekly wages, and average monthly employment present in the [Quarterly Census of Employment and Wages (QCEW) dataset](https://catalog.data.gov/dataset/quarterly-census-of-employment-and-wages-qcew-a6fea). We will employ machine learning algorithms and specialized models in order to accomplish this task. Our model will be a classification model doing supervised learning since California counties, which can be mapped to regions, are included in the QCEW dataset. This model could highlight which establishment sectors are popular in any given California region. We chose this project because inferring the region in which an establishment is based using features of that establishment seemed interesting and it helps us understand which establishments are popular and unpopular in any California region. Knowing where certain establishment sectors are popular or unpopular enables entrepreneurs to find untapped markets where there may be little competition or markets to stay away from with low chances for success.

# 2. Methods

## Data Exploration

Here are the steps we took to explore our data: 
- Print a heatmap for non-categorical data
- Print the columns of the dataset
- Print the mean, count, standard deviation, minimum, 25% quartile, 50% quartile, 75% quartile, and maximum for every column in the dataset
- Print the number of datapoints from years 2018 - 2019
- Print all unique values in the *industry names* column
- Print all null values in each column
- Print the datatypes of each column
- Perform shapiro wilks test on non-categorical data
- Print all unique values in the *Area Types* column
- Print the number of datapoints where *Area Type* is county
- Print a pairplot that only includes obeservations where *Area Type* is county, *Year* is either 2019 or 2018, and *Quarter* is annual
- Print a heatmap for non-categorical data

## Preprocessing

Here are the steps we took to preprocess our data: 
- Drop the columns *Ownership*, *1st Month Emp*, *2nd Month Emp*, *3rd Month Emp*, and *Total Wages (All Workers)*
- Drop columns *Area Type* and *Quarter*
- Drop observations from 2016 and 2017
- Drop outliers using IQR
- Create the column *Sector* which is just a remapping of *NAICS*
- Replace columns *NAICS Level*, *NAICS Code*, and *Industry Name* with *Sector*
- Create the *log_weekly_wages* column which is just a log transformation of the *Average Weekly Wages* column
- Create the *log_monthly_employment* column which is just a log tranformation of the *Average Monthly Wages* column
- Create a dataframe that one hot encodes the *Sector* feature
- Create a dataframe that label encodes the *Sector* feature
- Create a dataframe that drops the *Sector* feature

## Model 1

Here are the steps we took to create our [first model](https://github.com/ericstratford/predictable-residence/blob/Milestone3/CA_Residence_Prediction.ipynb): 
- Split the data into train and test using the one hot encoded dataframe
- Train and test our data using a decision tree model
- Train and test our data using a random forest model
- Train and test our data using a k-nearest neighbors model

## Model 2

Here are the steps we took to create our [second model](https://github.com/ericstratford/predictable-residence/blob/Milestone4/CA_Residence_Prediction.ipynb): 
- Map counties to their respective regions
- Standardize X_train and X_test using normalization
- Get the average k-fold cross validation accuracy for decision tree model with k = 10
- Train and test our data using a decision tree model 
- Get the average k-fold cross validation accuracy for k-nearest neighbors model with k = 10
- Train and test our data using a k nearest neighbors model
- Calculate the accuracy, true positive rate, true negative rate, false positive rate, and false negative rate across all classes for k-nearest neighbors

# 3. Results

## Data Exploration

Results from data exploration: 
- The columns of the dataset are *Area Type*, *Area Name*, *Year*, *Quarter*, *Ownership*, *NAICS Level*, *NAICS Code*, *Industry Name*, *Establishments*, *Average Monthly Employment*, *1st Month Emp*, *2nd Month Emp*, *3rd Month Emp*, *Total Wages (All Workers)*, and *Average Weekly Wages*.
- The mean, count, standard deviation, minimum, 25% quartile, 50% quartile, 75% quartile, and maximum for every column in the dataset are shown in **Figure 1**

![Fig 1](./CSE151A_fig1.png)
<center>
<b>Figure 1.</b> Mean, count, standard deviation, minimum, 25% quartile, 50% quartile, 75% quartile, and maximum for each column
</center><br>

- The number of datapoints from 2018-2019 is 506913.
- There are too many industry names to list (1000+ unique industry names).
- There are no null values in the dataset.
- The datatypes of each column are shown in **Figure 2**

![Fig 2](./CSE151A_fig2.png)
<center>
<b>Figure 2.</b> Columns and their associated datatypes
</center><br>

- Shapiro Wilks test failed because we have too many observations.
- The following are all the unique Area Types: 'County', 'California - Statewide', 'United States'
- 441540 datapoints are just counties.
- The pairplot that only includes obeservations where *Area Type* is county, *Year* is either 2019 or 2018, and *Quarter* is annual is shown in **Figure 3**

![Fig 3](./CSE151A_fig3.png)
<center>
<b>Figure 3.</b> Pairplot that only includes observations where <i>Area Type</i> is county, <i>Year</i> is either 2019 or 2018, and <i>Quarter</i> is Annual
</center><br>

- The heatmap for non-categorical data is shown in **Figure 4**

![Fig 4](./CSE151A_fig4.png)
<center>
<b>Figure 4.</b> Heatmap for non-categorical data
</center><br>

## Preprocessing

Results from preprocessing:

- The resulting dataframe after performing all of the preprocessing steps outlined in the methods section is shown in **Figure 5**

![Fig 5](./CSE151A_fig5.png)
<center> 
<b>Figure 5.</b> Resulting dataframe after performing preprocessing
</center><br>

## Model 1

Results from Model 1:
- Decision Tree Classification Model:
    - Training accuracy: 0.9995052425263078
    - Testing accuracy: 0.6148576647891613
- Random Forest Classification Model:
    - Training accuracy: 0.9995052425263078
    - Testing accuracy: 0.6089968031663876
- K-nearest Neighbors Classification Model:
    - Training accuracy: 0.4934254343399745
    - Testing accuracy: 0.1996498706043538

## Model 2

Results from Model 2:
- The resulting dataframe after mapping counties to their respective regions is shown in **Figure 6**

![Fig 6](./CSE151A_fig6.png)
<center> 
<b>Figure 6.</b> Resulting dataframe after mapping counties to their respective regions 
</center><br>

- Decision Tree Classification Model:
    - Cross-validation accuracy: 0.7119750017016482
    - Testing accuracy: 0.7351195006850357
- K-nearest Neighbors classification Model:
    - Mean accuracy: 0.7141
    - Testing accuracy: 0.7598569036383012
- Combined rates across all classes:
    - Accuracy/Correct Rate: 0.7598569036383012
    - True Positive Rate: 0.7598569036383012
    - True Negative Rate: 0.919952301212767
    - False Positive Rate: 0.08004769878723296
    - False Negative Rate: 0.2401430963616989

# 4. Discussion

## Data Exploration

Prior to preprocessing and model implementation, extensive data exploration was done to understand the structure and characteristics of the QCEW dataset. This step was important for identifying potential issues such as missing values, data distributions, and relevant features for analysis. Key exploratory steps were:
- Heatmap Analysis: A heatmap was generated for non-categorical data to identify potential correlations between numerical variables.
- Descriptive Statistics: Summary statistics, including mean, standard deviation, quartiles, and maximum/minimum values, were calculated for all columns to assess central tendency and variability.
- Null Value Check: The dataset was checked for missing values in each column, enabling us to determine whether or not data cleaning would be required.
- Data Filtering: Specific subsets of data were analyzed, such as observations where Area Type was county and Year was 2018 or 2019, to ensure that there would be enough relevant data points to use.
- Normality Test: A Shapiro-Wilk test was performed on non-categorical data to normality.
- Pairplot Generation: Visualizations, such as pairplots, were generated to explore relationships between variables.

These exploratory steps provided insights into the dataset's structure, ensuring that the features selected for modeling were relevant and that potential data issues could be addressed when moving forward. For example, identifying the number of older data points helped us to focus on the more meaningful subsets of the data. Additionally, the use of pairplots and heatmaps allowed us to visualize relationships and refine feature selection.

## Preprocessing

During our preprocessing step, we dropped many of the columns from the original dataset. We felt like this would help with the accuracy and real-world applicability of our models. For starters, we dropped the data from 2016 and 2017 from our dataset since having dates from more recent years would better reflect current conditions. The dataset is quite large so splitting it in half would also reduce the overall computational cost. We also decided to only retain rows that contained annual data since we do not want quarterly and annual information for the same establishments mixed together as that may introduce noise. Another step we took was creating a new column called *Sector* which would use the NAICS code mapping to derive broader encompassing sectors to categorize each data point. This dramatically reduced the number of *Industries* we had, helping us keep dimensionality low when encoding. We also dropped any other columns that we did not believe would help with the predictive power of the model based on our data exploration to further reduce the dimensionality. One thing that we dropped during our preprocessing that may have been beneficial to keep however was *Ownership* since knowing if an establishment was owned federally or by a private party may have helped with the accuracy of our models. For example, the city of San Diego was established due to the Navy's large presence, so it makes sense that there are more federally owned establishments here. This is something we should have taken into consideration before making the decision to drop that column.

## Model 1

For our first model, we initialy tested three different models to get an understanding of what would work best with our data. We decided to go with a decision tree model because it was yielding the best test accuracy. The decision tree model was also appealing since our dataset has both numerical variables, such as income levels, and categorical variables, such as industry type. Decision trees can easily work with different data types without requiring extensive preprocessing, such as scaling or encoding. Furthermore, the simplicity of decision trees makes them computationally efficient, which was an advantage during the exploratory phase of our analysis. The use of the Gini impurity ensured that the model focused on maximizing similarities in each node, which is ideal for a classification task. However, decision trees are prone to overfitting, and this was an issue that we ran into. Unfortunately, our training accuracy was roughly 20-30% higher than our testing accuracy. We attempted to tune the model by pruning it, but dealt with sharp drops in model performance; we struggled with balancing the issue of overfiting and the accuracy of our model.

## Model 2

Building on our expereince from Model 1 we implemented a second decision tree model, but  this time we incorporated cross-validation to address the overfitting issue observed in the first model. Cross-validation helped evaluate the model's performance on multiple subsets of the dataset and enabled us to fine-tune hyperparameters, such as maximum depth and minimum samples per split, to achieve better performance. By using cross-validation, we were able to slightly increase the accuracy of the second model. We also fine-tuned our KNN model by reducing the number of neighbors and specifying that neighbors of closer distance bear greater influence on the predictions. Through this fine tuning we were able to drastically improve that other model's performance. However, the KNN model still did not perform as well as the Decision Tree model, so we went with the latter. Despite the improvement to the model, it's accuracy was still lower than we would like and the model is struggling to fully capture the underlying relationship in our data.

# 5. Conclusion

### Model 1

### Model 2

# 6. Statement of Collaboration

Name: Eric Stratford\
Title: Team Leader\
Contribution: Created the group repository and discord. Assisted with coding and milestone writeups.

Name: Kesler Anderson\
Title: Project Manager\
Contribution: Managed project deadlines by making Gradescope submissions and creating milestone branches. Assisted with coding and miletone writeups.

Name: Syed Usmani\
Title: Coder/Writer\
Contribution: Assited in pre-processing, tuning models, adding cross-validation, and writing.

Name: Maasilan Kumaraguru\
Title: Coder/Writer\
Contribution: Assisted in coding and writing.

Name: Viet Tran\
Title: Coder/Writer\
Contribution: Assited in coding and writing.