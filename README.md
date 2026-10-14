# Student-Exam-Grade-Predictor
A supervised machine learning project that predicts student exam scores based on given study hours and attendance data. This machine learning model is implemented by using linear regression and gradient descent.
# Overview
This project is a simple machine learning model, built from scratch, using NumPY to predict new students' exam scores based on their study hours and attendance rate. This project demonstrates how a model can learn and predict new outcomes (given new incomes) based on already given incomes and outcomes.

# Before getting started:
1. Dataset: Each student has two piece of information that can be used to predict their exam score, which are study hours (1) and attendance rate (2). Another dataset is also provided, which consists of the ACTUAL exam scores that these students score.
2. Standardization: Standardize the student data using their means and standard deviation.
3. Prediction: Use linear regression to predict scores.
4. Error: After each prediction is made, do (prediction - actual) to calculate the error.
5. Gradient: Using the error obtained, we use it to train and update the model's weights and bias.
6. After running for a certain amount of loops (for this project, 100 loops), return the final weights and bias such that MSE is minimized. Then, pass on new students data but without their actual exam score. Using the train model's weights, bias and standardization of data, the model makes prediction of the exam result of the new students.

Formulas:

1. Linear Regression Formula:

$$
\text{Predicted Grade} = w_1x_1 + w_2x_2 + b
$$

2. Update of bias:

$$
b_{\text{new}} = b_{\text{old}} - \alpha \times \text{Bias Gradient}
$$

3. Update of weights:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \times \text{Weight Gradient}
$$

## Gradient Descent

The partial derivative with respect to the weight is:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
$$

The partial derivative with respect to the bias is:

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})
$$
