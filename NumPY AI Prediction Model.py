import numpy as np

# x is the data of study hours, grade obtained by students.
# y is the actual result that the student obtained.
x = np.array([[2, 60],
     [3, 65],
     [4, 70],
     [5, 75],
     [6, 80],
     [7, 85],
     [8, 90],
     [9, 95]])
y = np.array([55, 60, 65, 70, 75, 80, 85, 90])

print (x.shape) # (8,2)
print (y.shape) # (8,)
print (np.mean(y)) # 72.5

weights = np.array([3, 0.5]) #Assuming that we have a fixed weight and bias.
bias = 15

# prediction function
y_pred = (x @ weights) + bias 
print(y_pred)
print (y_pred.shape) # (8,)

# finding the error, MSE and MAE
error = y_pred - y
print (error)

MSE = np.mean( error**2 )
print (MSE)

MAE = np.mean( np.abs(error))
print(MAE)

# looping through a list of biases values to determine which one has the smallest MSE
candidate_biases = np.array([15, 16, 17, 18, 19, 20])
MSE_values = []
for b in candidate_biases:
    prediction = (x @ weights) + b
    error = prediction - y
    MSE = np.mean( error ** 2 )
    MSE_values.append(MSE)
    print (b, MSE)

xs_val_index = np.argmin (MSE_values)
print (xs_val_index, candidate_biases[xs_val_index])

# TRAINING BIAS
# instead of doing it manually, we use gradient descent to find the best value to minimize MSE.
bias = 15
learning_rate = 0.1
prediction = (x @ weights) + bias
error = prediction - y
gradient = 2 * np.mean(error)
bias = bias - (learning_rate * gradient)
print (bias)

# now we can loop over this to repeat this process.
bias = 15
learning_rate = 0.1
for i in range(20):
    prediction = (x @ weights) + bias
    error = prediction - y
    gradient = 2 * np.mean(error)
    bias = bias - (learning_rate * gradient)

final_prediction = (x @ weights) + bias
error = final_prediction - y
MSE = np.mean( error ** 2)
print (bias, MSE)

#TRAINING WEIGHTS
# After training bias, we can train weights too
weights = np.array([3.0, 0.5])
bias = 15.0

prediction = (x @ weights) + bias
error = prediction - y
n = len(y)
w_gradient = (2/n)*(x.T @ error) #[ -19.5 -322.5]
print (w_gradient, w_gradient.shape)

# note that the gradient is very large, so we need our learning_rate to be small to avoid big jumps. To do this, standardize our x value.
x_scaled =  (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
print (x_scaled.shape, np.mean(x_scaled, axis = 0))

# Combining the two together (both bias and weight), and we create a model:
weights = np.zeros(2)
bias = 0.0
learning_rate = 0.1
x_scaled = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
for i in range(100):
    prediction = (x_scaled @ weights) + bias
    error = prediction - y
    error_sq = error ** 2
    
    weight_gradient = (2 / len(y)) * (x_scaled.T @ error)
    bias_gradient = 2 * np.mean(error)

    bias = bias - (learning_rate * bias_gradient)
    weights = weights - (learning_rate * weight_gradient)

f_prediction = (x_scaled @ weights) + bias
error = f_prediction - y
MSE = np.mean(error ** 2)
print (weights, bias, MSE)

# Predicting scores for new students with new study hours and attendance rate:
new_students = np.array([[4.5, 72.5], [6.5, 82.5], [8.5, 92.5]])
new_stu_standard = (new_students - np.mean(x, axis = 0)) / np.std(x, axis = 0)
predicted_exam_score = (new_stu_standard @ weights) + bias
print (predicted_exam_score)
    


        
