# Group 3: Project 4 Report Template

## Project Goal

* The main objective is to develop an accurate and reliable predictive model that can anticipate the probability of a heart attack in patients, using medical data and relevant risk factors.
  
## Data Sources
* The data was sourced from Kaggle, where 5 different datasets were combined to make one comprehensive overview of heart diseases
* There are 11 features present in this dataset
* The final dataset before cleaning had a count of 918
* After cleaning, the count became 917
* Link to Data: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ 

## Data Cleaning
* This dataset was relatively clean and comprehensive after checking its size, duplicates, and null values
* However, there was one data point which did not make sense, where resting blood pressure was 0
* This data point was taken out of the final dataset


## Models
* Neural Network
  * We trained a neural network model to learn from some data and make predictions to see if it gave the most optimized result or not.
  * We trained it to a model with 2 hidden layers each with 20 neurons which are the same number as the input variables.
  * When the neurons were increased the accuracy decreased stating there is a chance of overfitting
  * The final model is trained on 50 epochs but initially started with 30 epochs.
  * The increasing number of epochs got the increase in accuracy. But when increased to 100 epochs the accuracy dropped
  * The loss is a measure of how much error my model makes, and the accuracy is a measure of how many correct predictions my model makes.
  * Our aim was to have a low loss and a high accuracy for the model. The accuracy is 0.84 which is high but when comparing with the other models is less
  * However the neural network did not turn out an ideal model as the count of the data is less causing the model to over fit very easily.


* Logistic Regression (Original)
  * The precision the ratio of true positives to all predicted positives, i.e., TP / (TP + FP).
  * The recall is the ratio of true positives to all actual positives, i.e., TP / (TP + FN).
  * The table shows that the model has a higher precision (0.89) and f1-score for class 1 (0.86) than for class , which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 0(0.86) than for class 1(0.83), which means that it is more complete in finding class 0 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.84 which is the accuracy for this model.


* Logistic Regression (Resampled)
  * The precision the ratio of true positives to all predicted positives, i.e., TP / (TP + FP).]
  * The recall is the ratio of true positives to all actual positives, i.e., TP / (TP + FN).
  * The table shows that the model has a higher precision(0.92) and f1-score for class 1 (0.7) than for class 0(0.84), which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 0(0.90) than for class 1(0.83), which means that it is more complete in finding class 0 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.86 which is the accuracy for this model.
  * The resampling increased the accuracy from the previous model
  * Reason for using this method -It can improve the accuracy and generalization of machine learning models by reducing the effects of imbalanced class distribution, overfitting, and high variance
  * It helps in creating new synthetic datasets for training machine learning models and to estimate the properties of a dataset when the dataset is unknown, difficult to estimate, or when the sample size of the dataset is small1.


* Random Forest
  * We opted for Random forest model because Random forest can handle imbalanced datasets better, while logistic regression can produce biased predictions for rare classes as might be the case which we saw in the previous 2 models.
  * Random forest can handle high-dimensional data better, while logistic regression can suffer from overfitting and multicollinearity
  * Random forest works good on mixed data and very effective for categorical data
  * As a decision tree algorithm, Random Forests are less influenced by outliers than other algorithm
  * The table shows that the model has a higher precision(0.89) and f1-score for class 1 (0.88) than for class 0(0.84), which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 1(0.87) than for class 1(0.86), which means that it is more complete in finding class 1 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.87 which is the accuracy for this model.
  * In general, if the precision, recall, and the F1 score are higher, it means that the model is performing better which is in this case
  * Based on the classification report, the model has a high precision and recall for both classes, which means that it can correctly identify most of the true positives and true negatives, and avoid false positives


## Summary


