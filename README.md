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
  * .



* Logistic Regression (Resampled)
  * .


* Random Forest
  * .


## Summary


