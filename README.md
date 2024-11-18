# predictable-residence
This project (tentatively) aims to predict the residence of Californians based on demographic and characteristic features.

# Milestone 3: Pre-Processing

## Updates:
For this milestone we finished major preprocessing by scaling the data, using encoding, and

## Model Evaluation:
### Where does your model fit in the fitting graph?
![model fit](MST3_graph.jpg)
The model has much higher error on test data than it does on training data.
### What are the next models you are thinking of and why?
The next models we are thinking about using are Naïve Bayes, a support vector machine, k-nearest neighbors, or random forest. Specifically, we are thinking about using Naïve Bayes because if our data is conditionally independent, Naïve Bayes should work well. This is significant because it does seem like our data should be conditionally independent by cursory glance. Additionally, Naïve Bayes supports label encoding, whereas models like SVM don't support label encoding. This is significant because using label encoding could help us avoid the curse of dimensionality that could arise from one hot encoding. However, using an SVM with a soft margin would help us avoid overfitting to the training data which is occuring now with our decision tree model.

## Conclusions:
### What is the conclusion of your 1st model? 
The conclusion of our first model is a very high training accuracy of ~0.9995 and an abysmal testing accuracy of ~0.6149. In other words, our decision tree model overfits to the training data and doesn't generalize well.
### What can be done to possibly improve it?
To improve our first model we could make use of K-fold cross validation.

# Notebook Link
[Milestone 3 Notebook](https://github.com/ericstratford/predictable-residence/blob/Milestone3/CA_Residence_Prediction.ipynb)