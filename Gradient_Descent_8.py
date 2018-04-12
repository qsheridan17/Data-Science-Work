import matplotlib.pyplot as plt
from functools import partial
import random
import math

# Gradient Descent

# Gradient Descent is used to solve optimization problems



# Functions required from previous chapters:

class gradient_descent:

    def dot(self, v,w):
        '''The dot product of two vectors is the sum of their componentwise products'''
        return sum(v_i * w_i for v_i, w_i in zip(v,w))

    def sum_of_squares(self, v):
        '''v_1 * v_1 + ... + v_n * v_n
        computer a vectors sum of squares'''
        return self.dot(v, v)

    def vector_subtract(self, v, w):
        return [v_i - w_i for v_i, w_i in zip(v,w)]

    def squared_distance(self, v,w):
        '''(v_1 - w_1) ** 2 + ... (v_n - w_n) ** 2'''
        return self.sum_of_squares(self.vector_subtract(v,w))

    def distance(self, v, w):
        return math.sqrt(self.squared_distance(v,w))



    # The Idea Behind Gradient Descent

    '''def sum_of_squares(v):'''
    '''computes the sum of squared elements in v'''
    '''   return sum(x**2 for x in v)'''

    # We will frequently need to maximize or minimize functions like sum_of_squares. To find the input v that produces the largest
    # or smallest possible value.

    # The Gradient gives the input direction in which the function most quickly increases

    # One approach to maximizing the gradient is to pick a random starting point,
    # Compute the gradient,
    # Take a small step in the direction of the gradient (the direction that causes the function to increase the most),
    # Repeat with the new starting point.





    # Estimating the Gradient

    # If f is a function of one variable, its derivative at point x measures how f(x) changes when we make a very small change to x.
    # It is defined as the limit of the difference quotients

    def difference_quotient(self, f, x, h):
        return (f(x + h) - f(x))/h

    # Example of how easy it is to calculate derivatives:

    def square(self, x):
        return x*x

    # has the derivative:

    def derivative(self, x):
        return x*2

    # Estimate derivatives by evaluating the difference quotient for a very small e:

    '''derivative_estimate = partial(difference_quotient, square, h=0.00001)'''

    # When f is a function of many variables it has multiple partial derivatives, each indicating how f changes when we make small
    # Changes in just one of the input variables

    # We calculate the partial derivative by treating it as a function of just its ith variable, holding the other variables fixed:

    def partial_difference_quotient(self,f,v,i,h):
        '''compute the ith partial difference quotient of f at v'''
        w = [v_j + (h if j == i else 0) # add h to just the ith element of v
             for j, v_j in enumerate(v)]
        return (f(w) - f(v)) / h

    # after which we can estimate the gradient in the same way

    def estimate_gradient(self, f, v, h = 0.00001):
        return [self.partial_difference_quotient(f,v,i,h) for i,_ in enumerate(v)]




    # Using the Gradient

    # Let's use gradients to find the minimum among all three-dimensional vectors
    # We pick a random starting point
    # Then take steps in the opposite direction of the gradient until we reach a point we're the gradient is very small

    def step(self, v, direction, step_size):
        '''move step size in the direction from z'''
        return [v_i + step_size + direction_i for v_i, direction_i in zip(v, direction)]

    def sum_of_squares_gradient(self, v):
        return [2* v_i for v_i in v]


    # If you run this you'll get a v thats very close to [0,0,0] the smaller you make the tolerance the closer you'll get





    # Choosing the right step size

    # Method of choosing the right step size includes:
    # - Using a fixed step size
    # - Gradually shrinking the step size over time
    # - At each step choosing the step size that minimizes the value of the objective function (very costly computation)

    step_sizes = [100, 10, 1, .01, .001, .0001, .00001]

    # It is possible that the step size will return invalid inputs so we need to include a 'safe apply' function

    def safe(self, f):
        '''return a new function thats the same as f, except that it
        outputs infinity whenever f produces an error'''
        def safe_f(self, *args, **kwargs):
            try:
                return self.f(*args, **kwargs)
            except:
                return float('inf')     # means infinity in python
        return safe_f





    # Putting it all together

    # In the general case we have some target_fn we want to minimize and we have its gradient_fn.
    # We have chosen a starting value for the paramaters theta_0.

    def minimize_batch(self, target_fn, gradient_fn, theta_0, tolerance = 0.000001):
        '''use gradient descent to find theta that minimizes target function'''

        step_sizes = [100, 10, 1, .01, .001, .0001, .00001]

        theta = theta_0                 # Set theta to initial value
        target_fn = self.safe(target_fn)     # Safe version of target_fn
        value = target_fn(theta)        # Value we are minimizing

        while True:
            gradient = gradient_fn(theta)
            next_thetas = [self.step(theta, gradient, -step_size) for step_size in step_sizes]

            # Choose the one that minimizes the error function
            next_theta = min(next_thetas, key = target_fn)
            next_value = target_fn(next_theta)

            # Stop if we're converging
            if abs(value - next_value) < tolerance:
                return theta
            else:
                theta, value = next_theta, next_value

    # It is called minimize batch because for each gradient step it looks at the entire data set.

    # Sometimes we will want to maximize a function which we will do by minimizing its negative

    def negate(self, f):
        '''return a function that for any input x returns -f(x)'''
        return lambda *args, **kwargs: -self.f(*args, **kwargs)

    def negate_all(self, f):
        '''the same when f returns a list of numbers'''
        return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

    def maxmize_batch(self, target_fn, gradient_fn, theta_0, tolerance = 0.000001):
        return self.minimize_batch(negate(target_fn),
                              negate_all(gradient_fn),
                              theta_0,
                              tolerance)





    # Stochastic Gradient Descent

    # Stochastic Gradient Descent computes the gradient for only one point at a time. It cycles over the data repeatedly until it
    # reaches a stopping point

    # During each cycle we'll want to iterate through our data in a random order

    def in_random_order(self, data):
        '''generator that returns the elements of data in a random order'''
        indexes = [i for i, _ in enumerate(data)]   # creates a list of indexes
        random.shuffle(indexes)                     # shuffles them
        for i in indexes:                           # return the data in that order 
            yield data[i]   

    # We will want to take a gradient step for each data point.
    # This approach leaves the possibility that we might circle around a minimum forever so whenever we stop getting improvements
    # We'll decrease the step size and eventually quit.

    def minimize_stochastic(self, target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

        data = zip(x,y)
        theta = theta_0                                 # the initial guess
        alpha = alpha_0                                 # the initial step size
        min_theta, min_value = None, float("inf")       # the minimum so far
        iterations_with_no_improvement = 0

        # if we go 100 iterations with no improvement stop
        while iterations_with_no_improvement < 100:
            value = sum(self.target_fn(x_i, y_i, theta) for x_i, y_i in data)

            if value < min_value:
                # if we've found a new min remember it
                # and go back to the original step size
                min_theta, min_value = theta, value
                iterations_with_no_improvement = 0
                alpha = alpha_0
            else:
                # if otherwise not improving try shrinking the step size
                iterations_with_no_improvement += 1
                alpha *= .09

            # and take a gradient step for each of the data points
            for x_i, y_i in self.in_random_order(data):
                gradient_i = gradient_fn(x_i, y_i, theta)
                theta = self.vector_subtract(theta, scalar_multiply(alpha, gradient_i))

        return min_theta

    # this function will maximize 

    def maximize_stochastic(self, target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
        return self.minimize_stochastic(self.negate(target_fn),
                                   self.negate_all(gradient_fn),
                                   x, y, theta_0, alpha_0)

                    
    










