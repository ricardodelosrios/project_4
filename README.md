# Group 3: Project 4 Report Template


<div style="display: inline_block"><br/>
  <img align="center" alt="jupyter" src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" />
<div style="display: inline_block"><br/>
  <img align="center" alt="visual studio" src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" />
<div style="display: inline_block"><br/>
  <img align="center" alt="python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<div style="display: inline_block"><br/>
  <img align="center" alt="html" src="https://img.shields.io/badge/HTML-239120?style=for-the-badge&logo=html5&logoColor=white" />
<div style="display: inline_block"><br/>
  <img align="center" alt="css" src="https://img.shields.io/badge/CSS-239120?&style=for-the-badge&logo=css3&logoColor=white" />
<div style="display: inline_block"><br/>
  <img align="center" alt="js" src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" />



## Project Goal

* The main objective is to develop an accurate and reliable predictive model that can anticipate the probability of a heart attack in patients, using medical data and relevant risk factors.

## Libraries

`Pandas`: Used for efficient manipulation and analysis of tabular data. It allows loading the dataset from a CSV file, data manipulation, and exporting clean data.

`NumPy`: Provides functionalities for working with arrays and matrices, essential for efficient numerical operations in data analysis and modeling.

`Matplotlib and Seaborn`: Crucial for data visualization. Matplotlib is used for static plots, while Seaborn, built on Matplotlib, offers a high-level interface for more attractive visualizations.

`Plotly Express`: Offers interactive plots that allow detailed exploration of patterns and trends in the data.

`Scikit-learn`: Provides essential tools for machine learning modeling. In this project, it is used to split data, build logistic regression and random forest models, and evaluate model performance.

`Imbalanced-learn`: Used to address class imbalance issues in the data, specifically through oversampling techniques.

`TensorFlow`: Framework for building and training neural network models. It is used to define and train a neural network to predict heart diseases.

`Flask`: A web framework for Python used to build the web application.

`Joblib`: Utilized for model persistence and loading.

`Pickle`: Used for serializing and deserializing Python objects, particularly for saving and loading the machine learning model.

`Flask-CORS`: Necessary for handling Cross-Origin Resource Sharing, allowing the web application to make requests to the Flask API from a different domain.

  
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

### Neural Network

Construction and training of a neural network for heart diseases prediction.

`Training`: A neural network with two hidden layers and an output layer is defined and trained on the scaled training data.

`Evaluation`: The trained neural network is evaluated on the scaled testing set, and loss and accuracy metrics are reported.

  * We trained a neural network model to learn from some data and make predictions to see if it gave the most optimized result or not.
  * We trained it to a model with 2 hidden layers each with 20 neurons which are the same number as the input variables.
  * When the neurons were increased the accuracy decreased stating there is a chance of overfitting
  * The final model is trained on 50 epochs but initially started with 30 epochs.
  * The increasing number of epochs got the increase in accuracy. But when increased to 100 epochs the accuracy dropped
  * The loss is a measure of how much error my model makes, and the accuracy is a measure of how many correct predictions my model makes.
  * Our aim was to have a low loss and a high accuracy for the model. The accuracy is 0.84 which is high but when comparing with the other models is less
  * However the neural network did not turn out an ideal model as the count of the data is less causing the model to over fit very easily.

### Logistic Regression (Original)

Construction of a logistic regression model to predict heart diseases and evaluation of performance.

`Training`: The dataset is split into training and testing sets. The logistic regression model is then trained on the training set.

`Evaluation`: Model accuracy is evaluated on both the training and testing sets. Additionally, a confusion matrix and a classification report are generated for detailed performance metrics.

  * The precision the ratio of true positives to all predicted positives, i.e., TP / (TP + FP).
  * The recall is the ratio of true positives to all actual positives, i.e., TP / (TP + FN).
  * The table shows that the model has a higher precision (0.89) and f1-score for class 1 (0.86) than for class 0, which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 0 (0.86) than for class 1 (0.83), which means that it is more complete in finding class 0 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.84 which is the accuracy for this model.


### Logistic Regression (Resampled)

Implementation of oversampling using the RandomOverSampler to handle class imbalance in the dataset. Another logistic regression model is trained on the oversampled data.

`Training`: The logistic regression model is trained on the resampled training set.

`Evaluation`: Model performance is evaluated on the testing set using a balanced accuracy score, a confusion matrix, and a classification report.

  * The precision the ratio of true positives to all predicted positives, i.e., TP / (TP + FP).]
  * The recall is the ratio of true positives to all actual positives, i.e., TP / (TP + FN).
  * The table shows that the model has a higher precision(0.92) and f1-score for class 1 (0.7) than for class 0 (0.84), which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 0 (0.90) than for class 1 (0.83), which means that it is more complete in finding class 0 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.86 which is the accuracy for this model.
  * The resampling increased the accuracy from the previous model
  * Reason for using this method:
    * It can improve the accuracy and generalization of machine learning models by reducing the effects of imbalanced class distribution, overfitting, and high variance
    * It helps in creating new synthetic datasets for training machine learning models and to estimate the properties of a dataset when the dataset is unknown, difficult to estimate, or when the sample size of the dataset is small1.


### Random Forest

Construction and evaluation of a random forest model to predict heart diseases.

`Training`: A random forest classifier is created and trained on the scaled training data.

`Evaluation`: Model performance is evaluated on the scaled testing set using a balanced accuracy score, a confusion matrix, and a classification report.

  * We opted for Random forest model because Random forest can handle imbalanced datasets better, while logistic regression can produce biased predictions for rare classes as might be the case which we saw in the previous 2 models.
  * Random forest can handle high-dimensional data better, while logistic regression can suffer from overfitting and multicollinearity
  * Random forest works good on mixed data and very effective for categorical data
  * As a decision tree algorithm, Random Forests are less influenced by outliers than other algorithm
  * The table shows that the model has a higher precision (0.89) and f1-score for class 1 (0.88) than for class 0 (0.84), which means that it is more accurate and balanced in predicting class 1 instances.
  * However, the model has a higher recall for class 1 (0.87) than for class 0 (0.86), which means that it is more complete in finding class 1 instances.
  * The average values indicate that the model has a good overall performance, with an average precision, recall and f1-score of 0.87 which is the accuracy for this model.
  * In general, if the precision, recall, and the F1 score are higher, it means that the model is performing better which is in this case
  * Based on the classification report, the model has a high precision and recall for both classes, which means that it can correctly identify most of the true positives and true negatives, and avoid false positives

## Application best model (Random Forest)

We created a `Flask` web application that hosts a machine learning model trained to predict heart disease. The model, a Random Forest classifier, is loaded into the Flask application and exposed through API endpoints. Users can interact with the model by inputting various parameters related to heart health through a web interface.

### Features

* `Home Route (/)`: Renders the home page of the web application.

* `Predict Route (/predict)`: Accepts user inputs through a form, sends the data to the machine learning model, and displays the predicted result on the web page.

## HTML

### Home page

The web application has a home page displaying the title "Heart Disease Monitor" and a brief description of the project's objective. The main objective is to develop an accurate and reliable predictive model that anticipates the probability of a heart attack in patients using medical data and relevant risk factors. The project employs a machine learning model, specifically a Random Forest classifier, with an 87% accuracy rate.

Users interact with the model through a form on the web page, where they input details such as age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, and the slope of the peak exercise ST segment. Upon submitting the form, the model processes the input and returns a prediction.

#### Features
* `Prediction Form`: Allows users to input various medical parameters and risk factors to obtain a predictive result.

* `Responsive Design`: The web application is designed to be accessible and user-friendly on various devices.

* `Integration with Machine Learning`: Utilizes a pre-trained Random Forest classifier to provide accurate predictions.

### Predict page

The web application has a page with results section that dynamically displays predictions. If the model predicts a high probability of heart disease, an alert is shown; otherwise, a message indicating a low probability is displayed. Users can navigate back to the home page using the "Back to Home" button.

#### Features

* `Prediction Results`: The template dynamically displays predictions based on the model's output, providing users with an indication of their risk of heart disease.

* `Back to Home Button`: Allows users to return to the home page for additional predictions.

`GitHub Link`: A link to the GitHub repository, providing users access to the project's source code and potentially additional information.

## Javascript

The JavaScript function is a crucial component of the web application that facilitates the prediction of heart disease probability based on user input. It collects data such as age, blood pressure, cholesterol level, and other relevant factors, validates the inputs within specific ranges, and sends the data to the backend server for prediction.

The predictions are made by a machine learning model on the server, and the result is displayed on the web page.

### Functionality

* `Input Collection`: Gathers user input for various medical parameters and risk factors.

* `Data Formatting`: Organizes the collected data into a JSON format suitable for sending to the server.

* `HTTP Request`: Utilizes the Fetch API to send the data to the server for prediction.

* `Result Display`: Updates the web page with the prediction result obtained from the server.

## CSS

CSS style create a responsive and visually engaging web page. The styles cover various aspects, including the website title, main container, text area, form elements, buttons, contact icons, and the footer.

 * `Body and Container Styles`: The body of the page is configured to have a flex layout, allowing for dynamic responsiveness. The main container is styled with a gradient background and specific font choices, enhancing the overall aesthetic.

* `Text Area and Form Styles` : The text area, where users input their data, is styled for clarity and readability. Form elements, including text inputs and selects, have defined widths, paddings, and borders for a consistent and user-friendly appearance.

* `Button Styles`: The predict button and back button have distinct styles with background colors, borders, and box shadows, providing a clear visual indication of their purpose. Hover effects enhance user interactivity.

* `Result Styles`: The result section is styled to display prediction outcomes with specific colors indicating risk levels (red for high risk and green for low risk).

* `Footer Styles`: The footer has its own gradient background, and the contact icon is styled for visual appeal. The footer description is presented in a smaller font size for a balanced layout.

## How Users Can Use It

* `Clone the Repository`: Download or clone the repository to your local environment.

* `Install Dependencies`: Ensure you have the necessary libraries installed. You can do this by running pip install -r requirements.txt.

* `Run the Flask Application`: Execute the Flask application by running the provided script (app.py).

* `Access the Web Application`: Open your web browser to access the home page.

* `Submit Predictions`: Navigate to the prediction page and submit inputs through the provided form to get predictions.



## Summary
  * Overall, the random forest model best predicts the likelihood of heart disease with an 87% average accuracy, recall, and f1 score.
  * Also, a Flask app was created for users to input their statistics and have our random forest model predict their likelihood of heart disease.
  * Visualizations are also presented.



