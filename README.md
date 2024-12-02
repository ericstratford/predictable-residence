# predictable-residence
This project (tentatively) aims to predict the residence of Californians based on demographic and characteristic features.

# Milestone 2: Pre-processing Plans

## Pre-processing Plans:
### How will you preprocess your data?
Based on the inital data exploration we've done, we plan to drop all rows that are from 2016 and 2017, keeping only those from 2018 and 2019. We are doing this because the dataset is quite large, and since we are dealing with economic data, we want to use the most recent information. We also plan to use the NAICS Code column to create a new column with the economic sector, something that is built into the NAICS code structure. This is because we have 1586 unique industry names, and many of them are very specific, so we want to bring this number down by mapping the rows to broader encompassing sectors. This will make the encoding much easier, which is something that we'll need to do since the columns are categorical variables with no intrinsic ordering. Due to the nature of the columns, we believe that binary encoding would be the best to use. We also plan to drop '1st Month Emp', '2nd Month Emp', '3rd Month Emp', and 'Total Wages (All Workers)' since we already have Average Monthly Employment capturing similar data and we also don't need to know the total wages since we are more concerned with individuals. We also want to remove rows whose 'Quarter' column is not "Annual" because keeping Quarterly data means that there may be overlap between whats captured in the "Annual" rows and the quarterly information. We also want to filter out rows whose 'Area Type' column is anything but "County" since that is what we are trying to predict. After removing those rows, we can remove the columns 'Area Type' and 'Quarter'. There are no null values to drop or replace in any of the columns; however, the standard deviation of many of the columns is quite high, so we will have to remove many outliers. Perhaps we could remove outliers using IQR. We can't use the shapiro wilks test because we have too much data, so we will have to use another method to determine whether our numerical data is normally distributed or we will have to rely on MinMaxStandardization.

## Notebook Link:
[Milestone 2 Notebook](https://github.com/ericstratford/predictable-residence/blob/Milestone2/CA_Residence_Prediction.ipynb)

# Milestone 3: Pre-Processing

## Updates:
We began major preprocessing by dropping data that we thought was irrelevant or redundant. We dropped data from 2016 and 2017 because we felt it was too outdated and would not help our model make accurate predictions. We also dropped the total wages, individual months of employement, and quarter because there was overlap between these columns and other that we kept. Lastly, we dropped area type and business ownership because we felt that these were irrelevant for our purposes.

Next, we dropped outliers using IQR and used NAICS code to generalize industry names into broader categories. Once this was finished, we tried out both one-hot encoding and label encoding for the 'Sector' feature. For both encodings, we trained decision tree models with different depths (30 and 50) and splits (5 and 2) and compared the test accuracy results. We found that the highest test accuracy came from using one-hot encoding with a decision tree with a depth of 50 and split of 2.

In preparation for the next milestone, we also trained a random forest model with the same two encodings and found that the highest test accuracy came from using label encoding with a depth of 50 and a split of 3.

## Model Evaluation:
### Where does your model fit in the fitting graph?
![model fit](MST3_graph.jpg)
The model has much higher error on test data than it does on training data.
### What are the next models you are thinking of and why?
The next models we are thinking about using are Naïve Bayes, a support vector machine, k-nearest neighbors, or random forest. Specifically, we are thinking about using Naïve Bayes because if our data is conditionally independent, Naïve Bayes should work well. This is significant because it does seem like our data should be conditionally independent by cursory glance. Additionally, Naïve Bayes supports label encoding, whereas models like SVM don't support label encoding. This is significant because using label encoding could help us avoid the curse of dimensionality that could arise from one-hot and binary encoding. However, using an SVM with a soft margin would help us avoid overfitting to the training data which is occuring now with our decision tree model.

## Conclusions:
### What is the conclusion of your 1st model? 
The conclusion of our first model is a very high training accuracy of ~0.9995 and an abysmal testing accuracy of ~0.6149. In other words, our decision tree model overfits to the training data and doesn't generalize well.
### What can be done to possibly improve it?
To improve our first model we could make use of k-fold cross validation.

## Notebook Link:
[Milestone 3 Notebook](https://github.com/ericstratford/predictable-residence/blob/Milestone3/CA_Residence_Prediction.ipynb)

# Milestone 4:

## Model Evaluation:
### Where does your model fit in the fitting graph?
![model fit](MST4_graph.jpg)
The model has a lower accuracy for test data than for training data. There is roughly a 25% difference between the two.
### What are the next models you are thinking of and why?
The next models we are thinking about using SVM since it works well as a classifier for non-linear decision boundaries, and we can pair this with PCA to reduce the dimensionality of the dataset. Another option we are considering is a neural network using a multilayer perceptron model, which may work well since the dataset is large and complex. All of our models so far have had low accuracy, indicating that the relationship between the features and the target may be difficult to capture, an MLP model may be able to capture the relationship given the size of the dataset.
## Conclusions:
In conclusion, after grouping counties into broader regional categories and using cross-fold validation to evaluate our model fitting, overfitting is no longer a problem but accuracy remains less than ideal, at around 70-75% for all models trained.
### What is the conclusion of your 2nd model? 
The conclusion of our second model is that it is not a more precise model than our first one. Improvements in accuracy came from additional data preprocessing and were not due to the use of KNN. 
### What can be done to possibly improve it?
We could apply PCA to our test and train sets prior to fitting the model.

## Notebook Link:
[Milestone 3 Notebook](https://github.com/ericstratford/predictable-residence/blob/Milestone3/CA_Residence_Prediction.ipynb)
