# deep-learning-challenge

## Module 21 Homework Deep Learning Challenge

### Purpose:
The purpose of the deep learning challenge was to determine how accurate a deep learning model was for predicting that the money granted for the application was used efficiently, resulting in a successful project. 

### Installations required:
This project has been run in a Windows 11 GitBash environment, using Python 3.10.13 together with Jupyter Lab as the editor.  There was also a requirement to:
1. Use scikit-learn (Reference: https://scikit-learn.org/stable/),
2. Use tensorflow (https://www.tensorflow.org/),
3. Install and use the keras_tuner for tensorflow (https://www.tensorflow.org/tutorials/keras/keras_tuner)
4. Used the pandas library for Python

## Overview of the Analysis - Deep Learning Report:
#### Data Preprocessing

In the Deep_Learning_Challenge folder, please see the Starter_Code.ipynb file for the script which performs all code for Steps 1 and 2 of the project. This file contains a URL link to access a csv file called charity_data.csv.  This file is used to load the source data for this project.

This source data was grouped, to try to reduce the influence of any outlier data, and then encoded to ensure the data was in a numerical format, which can be correlated to determine the probability of correctly predicting the outcome.  <br> 
In this project, the target for each data record was that it represented either a successful application where the funding was used efficiently (indicted by 0) or an unsuccessful application where the funding was not used efficiently (indicated by 1).  <br>
Any non-numeric data, which is not relevant to the determination of the success of the project, such as the EIN and NAME columns which are for project identification only, where removed from the input data.

Deep Learning models have the following main variables determined during preprocessing: 

* (1) A target outcome, which for this project is named "IS_SUCCESSFUL" (represented in the model by 'y'), and <br>
* (2) features, which for this project are the remaining 43 'encoded' fields of data (inputs) (represented by X).

--------------------------------------------------------------------------
###### Split our preprocessed data into our features and target arrays
y = encoded_application_df.IS_SUCCESSFUL.values <br>
X = encoded_application_df.drop(columns = 'IS_SUCCESSFUL').values

--------------------------------------------------------------------------
#### Compiling, Training and Evaluating the Model
The X and y data is then divided into a training dataset (75% of the data) and a testing dataset (25% of the data) <br>
The data in these datasets is scaled and fitted to a Sequential data model, for which the following paramenters were given as to what the shape of the starter_code model should look like:

-------------------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------------------

I determined from the above table there should be 80 neurons (as per the Output Shape) in the first hidden layer of the model, to process the 43 features input as per the X dataset for training the model<br>

So to start training the model, we have: <br>
    43 data inputs X 80 neurons in the first hidden layer + 80 'biases' (one for each neuron) = 3,520 Params, as shown in the above table. See dense layer in table<br> The activation specified for this layer is the Rectified Linear Unit or 'relu', which is the most commonly used activation.

Moving to the second hidden layer, we have:
    80 inputs from the neurons in the first hidden layer X 30 neurons in the second hidden layer (as per the Output Shape) + 30 'biases' (one for each new neuron) = 2430 Params, as show in the above table. See dense_1 layer in table<br>    Again, the 'relu' activation is used

Finally we move to the outer layer, where we have:
    30 inputs from the neurons in the second hidden layer X 1 output (as per the Output Shape) + 1 'bias' for the single output = 31 as per the above table. See dense_2 later in table. <br>  The activation specified for this layer is the Sigmoid Function, used for outputs relating the probability of predictions.
     
Having defined the inputs and neurons and outputs in the model, I compiled the model. <br> 

Once compiled, the number of epochs (iterations for the model is defined) and the fit process that trains the model start iterating through the number of epochs defined, which in this scenario was 100.    As the iterations occur, the model is adjusting the weighted values it applies to the input data, to try to correctly train the model.

-----------------------------------------------------------------------------------------------------

Output of the Epoch iterations, showing accuracy of the predictive model

Epoch 1/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.6213 - loss: 0.6677
Epoch 2/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7205 - loss: 0.5715
Epoch 3/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7256 - loss: 0.5596
Epoch 4/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7332 - loss: 0.5515
Epoch 5/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7305 - loss: 0.5520
Epoch 6/100
804/804 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7353 - l

etc

----------------------------------------------------------------------------------------------------
Reference: https://blog.metaphysic.ai/weights-in-machine-learning/#:~:text=During%20the%20training%20process%2C%20a,weighted%20sum%20that's%20been%20calculated.0o### <br> 

### Summary Results for Steps 1 and 2
Once thew 100 epochs have run, the test dataset is used to evaluate the accuracy of the model and it's ability to predict a successful funding application.  It also takes into account the amount of data loss occurring in the model. <br>

Using the model as indicated in the Starter_code.ipynb file, we only get an accuracy of about 72.33%, as per this screen print from the evaluation step in the code.<br>

---------------------------------------------------------------------
268/268 - 0s - 2ms/step - accuracy: 0.7233 - loss: 0.5662
Loss: 0.5661863088607788, Accuracy: 0.7232652902603149

---------------------------------------------------------------------<br>
The resulting model is stored in the AlphabetSoupCharity.h5 <br>
This result is very close to the 72.63% result in the example given in the starter_code to indicate an approximate expected outcome.<br>


### Optimizing the Model
Step 3 of the project was to try optimising the model to get the probability of predicting the funding application outcome to above 75% (which is relatively low success rate for a predictability)

#### 'Manual' optimization techniques as learnt in class (PDF file)ts given. 

In the folder named 'Optimsation Attempts using class strategies from PDF', I made 3 attempts at optimization using techniques learnt in class as per the PDF located at:  https://git.bootcampcontent.com/Monash-University/MONU-VIRT-DATA-PT-02-2024-U-LOLC/-/tree/main/01-Lesson-Plans/21-Neural-Networks-Deep-Learning/2?ref_type=heads:   <br>

* (1) using 'tanh' as an activation function in a hidden layer, instead of 'relu', to try io increase the accuracy of the model to over 75%.

* (2) using 200 epochs instead of 100 epochs to try to reduce the loss and increase the accuracy of the model of over 75%.

* (3) using a third hidden layer to try to find additional relationships between the data, and get the accuracy of the data to over 75%.

Given that none, of the above optimization attempts increased the accuracy to over 75%, I tried using a keras_tuner to again reduce loss, and increase accuracy of the model.<br>

#### Keras-tuner optimization techniquese model.  

See the folder named 'Optimization Attempts using Keras Tuner', where I tried 2 models, one with far less neurons ('AlphabetSoupCharity_Optimisation.ipynb') than that original starter_code.ipynb file, and one 50 to 80 neurons as the units for the first layer ('AlphabetSoupCharity_Optimisation_Final_More_Neurons

### Summary Results for Step 3
As per the details below, I tried both 'manual' techiniques suggested in class exercises, and also the more automated tuning process of the keras tuner model.   However, I was not able to get any models to reach 75% or more in accuracy.  This highest achieved was still only 0.7278 accuracy, and the rate of loss really never slowed in any of the techniques attempted. .pprovers. 

#### References:
https://stackoverflow.com/questions/33191744/how-to-add-new-line-in-markdown-presentation

https://www.markdownguide.org/basic-syntax/