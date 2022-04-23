# Neural Network CharityAnalysis
Module 19 challenge prepared by Hannah Wikum - April 2022
___
## Resources
Data Source: charity_data.csv (provided)

Software: Jupyter Notebook, Python 3.7.11 with Pandas, Scikit-learn, TensorFlow

___
## Overview
### Background
This analysis was completed for Alphabet Soup, a nonprofit foundation dedicated to funding organizations with environmental, health, or world peace initiatives. The project consisted of creating a deep learning neural network from datapoints on 34,000 orgnaizations that have been funded by Alphabet Soup in the past. This data included information like the name of the organization, industry affiliation, gorvernment classification, organization type, use for the funding, income classification, speacial considerations, amount requested, and if the money was used effectively. The goal is to use the deep learning model to be able to predict if a charitable venture that applies in the future will be successful or not if it was funded by the foundation.

### Analysis
To build my deep learning neural network, I followed the following steps:
  1. Loaded data from the CSV into a dataframe and pre-processed the data, which included dropping irrelevant columns, binning long tail values into an 'other' bucket for columns with more than 10 unique values, and encoding categorical (text) variables using a one-hot encoder before merging and dropping the original categorical columns
  2. Defined y (target variable = IS_SUCCESSFUL) and X (all other columns), split into training and testing datasets, and scaled the X data
  3. Defined the deep neural network model, including number of input features, number of hidden layers and neurons per layer, and the type of activation to use in each layer
  4. Compiled and trained the model with a feature that saved a checkpoint after every five epochs
  5. Evaluated the model to view loss and check accuracy
  6. Evaluated results and saved final results to a .h5 file

After my initial model was built, I went back and tested various modifications to try to achieve an accuracy over 75%. (My initial model came in at 72.28%.) The modifications I tested included dropping additional columns that I thought could be adding noise, increasing the amount of hidden layers, the number of neurons per hidden layer, and the number of epochs, and changing the activation function within the hidden layers from ReLU to sigmoid.
___
## Results
The information below describes my thought process during pre-processing, building the intial model, and testing to improve.

_Variables by Type in Data Pre-Processing_
 * Using the 11 columns in the charity_data.csv, the data in the IS_SUCCESSFUL column was clearly the target variable because that is what we are trying to predict. 
 * Columns EIN and NAME contain identification data that will not help with predicting the success of a future venture, so they should be dropped.
 * All other columns (APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT) can be used for input variables to train the model.

_Compiling, Training, and Evaluating the Model_
 * **Number of Neurons** - When building my model I used 80 neurons in the first hidden layer and 30 in the second. The original 80 neurons in the first layer felt like a good starting quantity because it is best to use 2-3x the amount of neurons as the amount of input variables. I had 43 inputs in the original model and later 40 after testing with dropping the STATUS and SPECIAL_CONSIDERATIONS columns. I later tested with 120 in the first layer (on the higher end of the suggested range) to try improve the accuracy of the model.

 * **Number of Layers** - My original neural network model had two hidden layers. I tested with adding a third to try to improve the efficacy with the idea that a third layer would train on data that has already been analyzed and could better catch nonlinear patterns.
 
 * **Activation Functions** - My first model used ReLU functions in the two hidden layers and a sigmoid activation in the output layer. I tested with only using sigmoid functions in the hidden layers and output to try to improve the accuracy because the sigmoid model is ideal for binary classification. (The target variable of successful/unsuccessful is a binary outcome.)

_Testing to Improve the Model to over 75% Accuracy_

 1. In the first attempt, I focused on pre-processing. I dropped the STATUS and SPECIAL_CONSIDERATIONS columns during the pre-processing stage to try to reduce noise and reran everything else in the model the same as the original model. The model accuracy was 0.7250, which was higher than the original 0.7228, but did not meet the 75% goal.

  _Test 1 Results: Drop Extra Columns_
  
  ![image](https://user-images.githubusercontent.com/93058069/164945544-2e1d850b-aac2-4bab-9e83-beacf2ea3b54.png)
  
 2. My second attempt involved building off the first test with dropping two extra columns, plus adding a third hidden layer, and changing the activation function in all hidden layers from ReLU to sigmoid. This model had an accuracy score of 0.7280, which was improvement from the first test and original model, but still not above the 75% hurdle.

  _Test 2 Results: Drop Extra Columns, Add a Layer, and Change Activation Functions_

  ![image](https://user-images.githubusercontent.com/93058069/164945699-bd52890c-f9b5-41e8-aaaa-f5582eba1883.png)

 3. My third attempt involved dropping the extra columns from the first test, increasing the number of neurons in the three layers from the second test, and then increasing the number of epochs from 50 to 100. Instead of using 80 neurons in the first layer (approximately 2x more than the number of input variables), I increased it to 120 (3x more) to be on the higher end of the rule of thumb. I also upped the neurons in the second layer from 30 to 50 and third layer from 10 to 20. I kept the third layer from the second test and also continued to use sigmoid activation function for all hidden and output layers. The accuracy results were 0.7269, which is actually worse than my second attempt.

  _Test 3 Results: Drop Extra Columns, Add a Layer, Change Activation Function, Increase the Number of Neurons, and Use More Epochs_
 
  ![image](https://user-images.githubusercontent.com/93058069/164946031-080f15a9-d692-470b-a01d-6e48dcf819eb.png)

___
## Summary
In conclusion, the best model I created could predict whether an application would be successful with 72.8% accuracy. Although this is not a medical prediction that would require a very high accuracy rate, it would still be good to most effectively distribute money to charitable causes if the accuracy was higher. As an alternative, I would recommend using a logistic regression model, which is a supervised machine learning model. The reason I am suggesting a logistic regression is because we are trying to determine a binary classification where the target variable (success) is known. In addition, neural networks can be overcomplicated or overfit to the training data, which is less of a risk with the logistic regression model.
