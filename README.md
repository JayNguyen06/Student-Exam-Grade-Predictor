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

# Formulas:

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

4. Standardization of x:

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

5. MSE (Mean Squared Error) :

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$


Where:

$$
\alpha = \text{Learning Rate}
$$

$$
\text{Bias Gradient} = 2 \times \text{Mean}(\text{Error})
$$

$$
\text{Weight Gradient} = 2 \times \text{Mean}(\text{Error} \times X)
$$

# Process:
1. First set both weights and bias equal to 0. We also set the learning rate to be 0.1
2. Because our weight_gradient can return a huge value, possibly resulting in a huge shift (unwanted) because of our current x data, we standardized x
3. Calculate the prediction using our linear regression formula, then calculate the error. Use this error to find MSE.
4. Calculate the weight_gradient as well as the bias_gradient, and update both values based on the functions given above.
5. Loop this process 100 times to minimize our MSE and find a suitable weights and bias.
6. After obtaining the final value for both weights and bias, use these values to predict exam scores for new students' data.

## Result:

| Parameter | Value |
|---|---|
| Weight 1 | 5.7282 |
| Weight 2 | 5.7282 |
| Bias | 72.5 |
| Training MSE | ≈ 0 |


## Example Predictions

| Study Hours | Attendance | Predicted Exam Score |
|---|---|---|
| 4.5 | 72.5% | 67.5 |
| 6.5 | 82.5% | 77.5 |
| 8.5 | 92.5% | 87.5 |

# Possible Improvements:
1. This model uses perfectly correlated training data between the study hours, attendance rate and exam scores. In real life, this model has not been tested, and might not work as well.
2. This model also uses a small set of data, and can be improved by training a larger set of data.
3. Compare the implementation with scikit-learn's linear regression model.

# Techonology Used:
- Python
- NumPy

