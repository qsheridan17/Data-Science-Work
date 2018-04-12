# This tutorial will teach you how to:
#       How to estimate linear regression coefficients using stochastic gradient descent
#       How to make predictions for multivariate linear regression
#       How to implement linear regression with stochastic gradient descent to make predictions on new data

# Multivariate Linear Regression:
# -      Linear regression is a technique for prediction a real value
# -      Linear regression is a technique where a straight line is used to model the relationship
#        between input and output values. In more than two dimensions, the straight line may be
#        thought of as a plane or hyperplane.
# -      Predictions are made as a combination of input values predict the output value
# -      Each attribute (x) is weighted using coefficient (b) and the goal of the learning
#        algorithm is to discover a set of coefficients that results in good predictions (y)\\

# Stochastic Gradient Descent:
# -      Coefficients can be found using gradient descent
# -      Gradient Descent is the process of minimizing a function by following the gradients
#        of the cost function.
# -      This involves knowing the cost as well as the derivative so that from a given point
#        you know the gradient can move in that direction (downhill torwards the min value),
# -      The model makes a prediction for a training instance as show to the model one at a time
#        The model makes a prediction for the training instance, the error is calculated and the
#        model is updated in order to reduce the error for the next prediction. This is repeated
#        for a fixed number of iterations.




# Gradient Descent

# b = b - learning_rate * error * x
# Where b is the coefficient or weight being optimized
# learning rate is a learning rate you must configure
# error is the prediction error of the model on the training data attributed to the weight
# x is the input value

# Tutorial is broken down into 3 parts
# 1) Making predictions
# 2) Estimating Coefficients
# 3) Wine Quality Prediction


# 1 MAKING PREDICTION

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

dataset = [[1,1], [2,3], [4,3], [3,2], [5,5]]
coef = [0.4, 0.8]
for row in dataset:
    yhat = predict(row, coef)
    #print "Expected=%.f, Predicted=%.1f" % (row[-1], yhat)

# Expected=1.000, Predicted=1.200
# Expected=3.000, Predicted=2.000
# Expected=3.000, Predicted=3.600
# Expected=2.000, Predicted=2.800
# Expected=5.000, Predicted=4.400

# These are the outputs that are reasonably close to the expected output values (y)




# 2 ESTIMATING COEFFICIENTS

# Stochastic Gradient Descent requires two parameters
# Learning rate : Used to limit the amount each coefficient is corrected each time it is updated
# Epochs : The number of times to run through the training data while updating the coefficients

# Coefficients are based on the error the model made. The error is calculated as:
# error = prediction - expected

# There is one coefficient to weight each attribute and they are updated in a consistent way
# b1(t + 1) = b1(t) - learning_rate * error(t) * x1(t)

# The special coefficient at the beginning of the list is called the intercept or the bias
# It is updated in a similar way, except without an input as it is not associated with a specific value
# b0(t + 1) = b0(t) - learning_rate * error(t)

# Below is a function called coefficients_sgd() that calculates coefficient values for a
# training data set using stochastic descent

def coefficients_sgd(train, l_rate, n_epoch):
    '''Estimate linear regression coefficients using stochastic gradient descent'''
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef) # provides prediction 
            error = yhat - row[-1] # calculates error. coefficients are based on the error the model made
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i] # updates the coefficient
        print '>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error)
    return coef

# testing the function:
# dataset same as above
l_rate = 0.001 # Used to limit the amount each coefficient is corrected each time it is updated
n_epoch = 50 # Number of times to run through the training data while updating the coefficients
coef = coefficients_sgd(dataset, l_rate, n_epoch)         
print coef
# coef = [0.229, 0.801]



# 3 Wine Quality Prediction

# In this section we will train a linear regression model using stochastic gradient descent on
# the wine quality data set.

