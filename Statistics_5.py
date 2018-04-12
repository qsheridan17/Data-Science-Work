from __future__ import division
from matplotlib import pyplot as plt
from collections import Counter
import math


# Statistics


# Imported Functions

class stats:

    def sum_of_squares(self, v):
        '''v_1 * v_1 + ... + v_n * v_n
        computer a vectors sum of squares'''
        return self.dot(v, v)

    def dot(self,v,w):
        '''The dot product of two vectors is the sum of their componentwise products
            dot sums up the products of corresponding pairs of elements'''
        return sum(v_i * w_i for v_i, w_i in zip(v,w))


    # Histogram of Number of Friends
    '''
    num_friends = [100, 49, 47, 67, 10, 12, 12, 54, 54, 36, 84, 89, 22, 56, 20, 21, 1, 2, 88,
                   90, 61, 71, 66, 63, 45, 37, 41, 9, 8, 11, 54, 45, 33, 32, 27, 31, 60,
                   36, 36, 37, 29, 50, 49, 80, 98, 73, 64, 35, 45, 55, 15, 23, 38]
    friend_counts = Counter(num_friends)'''
    ''' 
    xs = range(101)     # Largest value is 100
    ys = [friend_counts[x] for x in xs]     # Height is the number of friends
    plt.bar(xs, ys)
    plt.axis([0, 101,0,5])
    plt.xlabel("Number of Friends")
    plt.ylabel("Number of People")
    plt.title("Distribution of Friends")
    plt.show() 
    '''

    # Basic Statistics on Histoigram
    '''
    number_of_points = len(num_friends)
    largest_value = max(num_friends)
    smallest_value = min(num_friends)
    sorted_values = sorted(num_friends)
    second_smallest_val = sorted_values[1]'''

    # Central Tendencies

    def mean(self, x):
        return sum(x)/ len(x)

    def median(self, n):
        size = len(n)
        sorted_list = sorted(n)
        midpoint = size//2
        if size % 2 == 1:
            return sorted_list[midpoint]
        else:
            lo = midpoint - 1
            hi = midpoint
            return (sorted_list[lo] + sorted_list[hi])/2
    

    def quantile(self, x, p):
        '''returns the pth-percentile value in x'''
        p_index = int(p * len(x))
        return sorted(x)[p_index]
    
    '''
    quantile(num_friends, .1) # 11
    quantile(num_friends, .9) # 84
    quantile(num_friends, .5) # 45 - same as the mean
    '''

    def mode(self, n):
        '''returns a list of the most common values'''
        counts = Counter(n) # creates a list of tuples with the count of each time each number appears
        max_count = max(counts.values()) # max amount of times a value appears
        return [x for x, y in counts.iteritems() if y == max_count] # use iteritems() to make the count tuple list iterable


    # Dispersion
    # Dispersion refers to measures of how spread out our data is.
    # Zero signifies not spread out at all and large values signify very spread out

    def data_range(self, n):
        return max(n) - min(n)

    # A more complex version of dispersion is variance

    def de_mean(self, n):
        ''' returns a list of every item subtracted by the mean'''
        n_bar = self.mean(n)
        return [n_i - n_bar for n_i in n]

    # variance measures how a single variable deviates from the mean 
    def variance(self, n):
        length = len(n)
        deviations = self.de_mean(n)
        return self.sum_of_squares(deviations)/ (length - 1)

    #variance(num_friends)  # 658

    def standard_deviation(self, x):
        ''' returns the square root of the variance'''
        return math.sqrt(self.variance(x))

    #standard_deviation(num_friends) # 25.65




    # Correlation
    # The relationship between metrics
    # Covariance measures how to variables vary in tandem from their means

    #num_list1 = [34 , 634 , 56 , 43 ,13 ,34 ,123 ,32]
    #num_list2 = [ 23, 23, 87, 1, 45, 876, 1000, 43, 22]

    def covariance(self, x, y):
        ''' When there is a large positive covariance x is large when y is large and small when y is small.
            A large negative covariance means the opposite - that x tends to be small when y is large and large when
            y is small.
            However it can be hard to determine what counts as a large covariance so it is more common to use
            correlation'''
        length = len(x)
        return self.dot(self.de_mean(x), self.de_mean(y)) / (length - 1)

    def correlation(self, x, y):
        '''correlation divides out the standard deviations of both variables
            The correlation is unitless and always lies between -1 (perfect anti-correlation) and 1
            (perfect correlation). A number like 0.25 represents a relatively weak positive correlation'''
        stdev_x = self.standard_deviation(x)
        stdev_y = self.standard_deviation(y)
        if stdev_x > 0 and stdev_y > 0:
            return self.covariance(x, y) / stdev_x / stdev_y
        else:
            return 0 # if no variation correlation is 0 '''

    # Correlations can be heavily affected by outliers, its often good to remove them if they don't reflect your data



# Simpsons Paradox
# Simpsons Paradox is when correlations can be misleading and confounding variables are ignored.
# For example imagine your comparing the salaries of east coast and west coast college graduates. You find that
# on average the west coast schools have way highers starting salaries. However among closer look you realize that computer science majors make
# a significantly higher starting salary on east coast and west coast, and the west coast has a higher number of computer science majors.
# The only way to avoid this is by knowing your data and making sure you've checked for possible confounding variables.

# A correlation of zero means there is no linear relationship between the datasets however it is possible there are other correlations.
# Also correlation tells you nothing about how large the relationship is
# [ -2, -1, 0, 1, 2]
# [ 99.98, 99.99, 100, 100.01, 100.02] or it could be possible that they are perfectly correlated but the relationship isnt that interesting like here.

# Remember that "correlation does not mean causation" there are often third parties that we fail to consider
