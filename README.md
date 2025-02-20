# ANN-
This code implements an Artificial Neural Network (ANN) using TensorFlow/Keras to predict customer churn (whether a customer will leave the bank or not) based on the Churn_Modelling dataset. Below is a detailed breakdown of the code:

Step 1: Data Preprocessing
Loading the Dataset:

The dataset is loaded from a CSV file (Churn_Modelling.csv).

Features (x) are selected from columns 3 to the second-to-last column.

The target variable (y) is the last column, which indicates whether the customer left the bank (1) or not (0).

Encoding Categorical Data:

Label Encoding: The Gender column (index 2) is encoded into numerical values (0 for Female, 1 for Male).

One-Hot Encoding: The Geography column (index 1) is one-hot encoded to convert categorical data into binary vectors (e.g., France = [1, 0, 0], Spain = [0, 1, 0], Germany = [0, 0, 1]).

Splitting the Dataset:

The dataset is split into training (80%) and testing (20%) sets using train_test_split.

Feature Scaling:

Features are scaled using StandardScaler to normalize the data, which is crucial for ANN performance.

Step 2: Building the ANN
Initializing the ANN:

A sequential model (A_nn) is created using TensorFlow/Keras.

Adding Layers:

Input Layer and First Hidden Layer: A dense layer with 6 neurons and ReLU activation function is added.

Second Hidden Layer: Another dense layer with 6 neurons and ReLU activation function is added.

Output Layer: A dense layer with 1 neuron and sigmoid activation function is added (since this is a binary classification problem).

Step 3: Training the ANN
Compiling the ANN:

The model is compiled using the Adam optimizer and binary cross-entropy loss function (suitable for binary classification).

Accuracy is used as the evaluation metric.

Training the Model:

The model is trained on the training data (x_train, y_train) with a batch size of 32 and 100 epochs.

Step 4: Making Predictions and Evaluating the Model
Single Prediction:

The model predicts whether a specific customer will leave the bank or not. 



will leave).

Test Set Predictions:

The model predicts the outcomes for the test set (x_test).

Predictions are thresholded at 0.5 and compared with the actual values (y_test).

Confusion Matrix and Accuracy:

A confusion matrix is generated to evaluate the model's performance.

Accuracy is calculated to measure the proportion of correct predictions.

Key Points
Dataset: The dataset contains customer information such as credit score, geography, gender, age, tenure, balance, number of products, credit card status, and estimated salary.

Objective: Predict whether a customer will leave the bank (churn) or not.

Model: A simple ANN with 2 hidden layers (6 neurons each) and an output layer with 1 neuron.

Evaluation: The model's performance is evaluated using a confusion matrix and accuracy score
