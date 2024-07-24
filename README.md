# deep-learning-challenge

## Module 21 Homework Deep Learning Challenge
### Description:
This is the Module 21 Homework deep-learning-challenge. It is using scikit-learn (Reference: https://scikit-learn.org/stable/) for Python along with tensorflow (https://www.tensorflow.org/) for deep machine learning. It involves using various combinations of data inputs, neurons in hidden layers and activations (as noted below). Its aim is to calculate the probability of a correct prediction of successful funding applications. Jupyter Lab has been used as the editor, including pandas libraries. 

### Getting started:
This challenge has 1 project called: Deep Learning Challenge using a Jupyter Notebook file called Starter Code.ipynb, which contains a URL link to access a csv file  file called charity_data.csv

### Installations required:
This project has been run in a Windows 11 GitBash environment, using Python 3.10.13 together with Jupyter Lab as the editor.  There was also a requirement to import scikit-learn and tensorflow libraries, together with an installation of the keras-tuner for tensorflow.  (https://www.tensorflow.org/tutorials/keras/keras_tuner)

### Executing Homework Challenge Requirements:
In the Deep_Learning_Challenge folder, please see the Starter_Code.ipynb file for the script which performs all code for Steps 1 and 2 of the project. These steps involved (1) preprocessing the data, and (2) compiling, training and evaluating the model. I have written code in the Starter_Code.ipynb file to achieve outputs that match the requirements given. 

In the folder named 'Optimsation Attempts using class strategies from PDF', I made 3 attempts at optimization using techniques learnt in class as per the PDF located at:  https://git.bootcampcontent.com/Monash-University/MONU-VIRT-DATA-PT-02-2024-U-LOLC/-/tree/main/01-Lesson-Plans/21-Neural-Networks-Deep-Learning/2?ref_type=heads: 

* (1) using 'tanh' as an activation function in a hidden layer, instead of 'relu', to try io increase the accuracy of the model to over 75%.

* (2) using 200 epochs instead of 100 epochs to try to reduce the loss and increase the accuracy of the model of over 75%.

* (3) using a third hidden layer to try to find additional relationships between the data, and get the accuracy of the data to over 75%.

Given that none, of the above optimization attempts increased the accuracy to over 75%, I tried using a keras_tuner to again reduce loss, and increase accuracy of the model.  

See the folder named 'Optimization Attempts using Keras Tuner', where I tried 2 models, one with far less neurons ('AlphabetSoupCharity_Optimisation.ipynb') than that original starter_code.ipynb file, and one 50 to 80 neurons as the units for the first layer ('AlphabetSoupCharity_Optimisation_Final_More_Neurons.ipynb') 


## Overview of the Analysis - Deep Learning Report:

### Purpose:
The purpose of the analysis in the deep learning project was to determine how accurate the deep learning  model was for identifying successful funding applications.  

### Data Preprocessing
Deep Learning models have the following main variables determined during preprocessing: 

* (1) A target outcome, which for this project is named "IS_SUCCESSFUL" (represented in the model by 'y'), and
*
* (2) features, which for this project are the remaining 43 'encoded' fields of data (inputs) (represented by X).

Having grouped potenital outliers in the data, and then encoded the data, the features are now all of numeric values which can be correlated to determine the probability of correctly predicting the outcome.  

In this project, the target for each data record was that it represented either a successful application where the funding was used efficiently (indicted by 0) or an unsuccessful application where the funding was not used efficiently (indicated by 1).  

Any non-numeric data, which is not relevant to the determination of the success of the project, such as the EIN and NAME columns which are for project identification only, where removed from the input data.

### Compiling, Training and Evaluating the Model
The starter_code provided, indicated the following model was to be used 

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 80)                  │           3,520 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 30)                  │           2,430 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │              31 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 5,981 (23.36 KB)
 Trainable params: 5,981 (23.36 KB)
 Non-trainable params: 0 (0.00 B)
 
### Data Preprocessing### Data Preprocessing

### Summary Results



### Financial Loan input data for predictions
The input data in the lending_data.csv file had a loan_status column to indicate if the loan was healthy (indicated by 0), or high-risk (indicated by 1).  Logisitc Regression is a form of supervised learning, and as is the case with supervised learning models, we know the answer to the question we are analysing (that is we know the loan_status as either a 0 or 1), and we are determining how well the Logistic Regression model predicts the correct outcome, to enable accurate classification of the data into the healthy and high-risk loans.

### Data preparation and machine learning steps used in the analysis
To create an X and y variable, the loan_status was separated from the lending_data.csv file, with this single column of loan status data being used as the y variable, which represents the 'target' or 'labels' for the two possible outcomes (healthy loans and high-risk loans.)  The remaining columns of data in the lending_data.csv file form the X variable 'features' of the loans and include items such as the loan size, interest rate, borrower income, debt to income ratio, and total debt.  

The scikit-learn train_test_split function, then uses the X and y variables to split the data into a train dataset that will teach the Logistic Regression model, and a test dataset that will test the Logistic Regression model after it is trained, with the aim of seeing how accurately the model predicts the status of the loans. 

The scikit-learn Logistic Regression function is then called and uses the training data for the X and y axis along with other functions such as the default solver 'lbfgs' and a maximum of 200 interations to control the model's performance and the .fit function to process the data in the Logistic Regression model.  

The resulting trained Logistic Regression model, is then used to process the test data for the X and y access to predict which records and healthy loans and which records are high risk loans.  The resulting predictions are compared to the actual loan status to see how accurate the predictions are.

If this project, is would appear theat the model was very accurate for the healthy loans and also reasonably accurate for the high-risk loans, as further explained in the Results section below.

To further test the accuracy of the model, the predictions and actual loan status data were analysed with the Confusion Matrix and the Classification Report functions from scikit learn. 


## Results Analysis using precision and recall scores and accuracy measures

### Machine Learning Logisitc Regression Model:
 
* Using the classification report, we see that this logistic regression model did a very good job at predicting the healthy loans, with the model achieving 100% precision and 100% recall for healthly loans.
* While the model does have 99% accuracy overall and a weighted average accuracy of 99%, these very high percentages, are potentially skewed by the input data have many more healthy loan records, than high-risk loan records.
* The macro average accuracy of 94%, and the precision and recall percentages of 87% and 89% respectively for high-risk loans, are indicative of the logistic regression model being slightly less accurate for predicting high-risk loans. So to improve the model further, attention would need to be paid to identifying features that help determine high-risk loans.

## Summary

In summary, the Logistic Regression model did an excellent job at predicting the healthly loans as demonstrated by the accurancy, precision and recall scores noted above. The confusion matrix rows and columns both add to the total of 19384 records in the model, which is another indicator that the model is accurate.  The classification report further indicated that the model does a good job predicting high-risk loan between 88% and 94% of the time.
This model can be recommended to predict the outcome for loans.  However, further analysis of high-risk home loans is needed, potentially with a more advanced machine learning model, considering more features that indicate tendency for a loan to become high-risk. 

The performance of the model is important, particularly in correctly identifying loan status = 1 for the loans that are high-risk, so that measures can be taken to further analyse characteristics of why the loans are high risk, and to avoid approval of such loans in the further. It would also be useful to further train the model to try to eliminate false positve and false negative predictions.  This would help to ensure that loans are correctly identified as healthy or high-risk, so that appropriate actions can be taken by loan approvers. 
