from __future__ import division
import random
import math
from matplotlib import pyplot as plt
from collections import Counter



# Probability



# Dependence - we know that two events E and F are dependent if knowing something about whether E happens
# gives us information on whether F happens.
# Otherwise they are independent.

# Mathematically we say that two events are independent if the probability that they both happen is the
# product of the probabilities that each one happens
# P(E,F) = P(E)P(F)

# Conditional Probability
# If two events E and F are necessarily independent (and if the probability of X is not zero) then we define the probability of 
# E "conditional on F" as:
# P(E|F) = P(E,F)/P(F) or rewritten as P(E,F) = P(E|F)P(F)
# This is the probability that E happens given that we know that F happens

# If E and F are independent: P(E|F) = P(E)

# What is the probability that a family has "Both children are girls and at least one of the children is a girl"
# We can check this by generating a lot of families:

def random_kid():
    return random.choice(['boy', 'girl'])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == 'girl':
        older_girl += 1
    if older == 'girl' and younger == 'girl':
        both_girls += 1
    if older == 'girl' or younger == 'girl':
        either_girl += 1

both_girls / older_girl # P(both | older): .514 (so basically 1/2)
both_girls / either_girl # P(both | either): .3415 (so basically 1/3)


# Bayes Theorem
# Bayes Theorem is a way of reversing conditional proabilities
# If we need to know the probability of some event E conditional on other event F occuring
# But we only have information about the probability of F conditional on E occurring.
# Using the definition of conditional probability tell us that:

# P(E|F) = P(E,F)/P(F) = P(E|F)P(E)/P(F)

# The event F can be split into the two mutually exclusive events "F and E" and "F and not E"

# P(F) = P(F,E) + P(F,¬E)

# So that:

# P(E|F) = P(F|E)P(E)/[P(F|E)P(E) + P(F|¬E)P(¬E)]





# Random Variables
# Expected value of a random variable is the average of its values weighted by their probabilities
# We can condition random variables



# Continuous Distribution
# A coinflip corresponds to a discrete distribution - one that associates positive probability with discrete outcomes.
# Because there are infinitely many numbers between 0 and 1, this means that the weight it assigns to individual points must necessarily be zero.
# For this reason we represent a continuous distribution with a PROBABILITY DENSITY FUNCTION (PDF) such as the probability of seeing a value in a certain
# interval requals the integral of the density function over the integral.
class prob:
    
    def uniform_PDF(self, x):
        '''density function for uniform distribution'''
        if x >= 0 and x < 1:
            return 1
        else:
            return 0
    
    def uniform_CDF(self, x):
        '''returns the probability that a uniform variable is <= x'''
        if x < 0: return 0
        elif x < 1: return x
        else: return 1



    # Normal Distribution
    # The normal distribution is the king of distributions. It is the classic bell curved shaped distribution and is completely determined by two parameters
    # its means u (mu) and its standard deviation o (sigma). The mean indicates where the bell is centered and the standard deviation how 'wide' it is.
    # It has the probability density function.

    def normal_PDF(self, x, mu=0, sigma=1):
        sqrt_two_pi = math.sqrt(2 * math.pi)
        return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

    # This graphs various normal PDFs (Probability Density Function)
    '''
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_PDF(x, sigma = 1) for x in xs], '-', label = 'mu = 0, sigma = 1')
    plt.plot(xs, [normal_PDF(x, sigma = 2) for x in xs], '--', label = 'mu = 0, sigma = 2')
    plt.plot(xs, [normal_PDF(x, sigma = 0.5) for x in xs], ':', label = 'mu = 0, sigma = .5')
    plt.plot(xs, [normal_PDF(x, sigma = -1) for x in xs], '-.', label = 'mu = -1, sigma = 1')
    plt.plot(xs, [normal_PDF(x, mu = -1) for x in xs], '.', label = 'mu = -1, sigma = 1')
    plt.legend()
    plt.title('Various Normal PDFs')
    plt.show()'''


    # When u (mu) = 0 and o (sigma) = 1, its called the standard normal distribution.

    # The cumulative distribution function for the normal distribution can be written using Pythons math.erf

    def normal_CDF(self, x, mu = 0, sigma =1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

    # This graphs various normal CDFs (Cumulative Distribution Function)
    '''
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_CDF(x, sigma = 1) for x in xs], '-', label = 'mu = 0, sigma = 1')
    plt.plot(xs, [normal_CDF(x, sigma = 2) for x in xs], '--', label = 'mu = 0, sigma = 2')
    plt.plot(xs, [normal_CDF(x, sigma = 0.5) for x in xs], ':', label = 'mu = 0, sigma = .5')
    plt.plot(xs, [normal_CDF(x, mu = -1) for x in xs], '.', label = 'mu = -1, sigma = 1')
    plt.legend()
    plt.title('Various Normal CDFs')
    plt.show()'''
    
    # Sometimes we will need to invert normal_CDF to find the value corresponding to a specified probability

    def inverse_normal_CDF(self, p, mu = 0, sigma = 1, tolerance = 0.00001):
        '''find appropriate inverse using binary search.
        This function repeatedly bisects intervals until it narrows in on a Z
        thats close enough to the desired probability'''

        # if not standard, compute standard and rescale
    
        if mu != 0 or sigma != 1:
            return mu + sigma * self.inverse_normal_CDF(p, tolerance = tolerance)

        low_z = - 10.0      # normal_CDF(-10) is very close to 0
        high_z = 10.0       # normal_CDF(10) is very close to 1
        while high_z - low_z > tolerance:
            mid_z = (low_z + high_z) / 2    # consider the midpoint
            mid_p = self.normal_CDF(mid_z)       # and the CDFs value there 
            if mid_p < p:
                # midpoint is still too low, search above it 
                low_z = mid_z
            elif mid_p > p:
                # midpoint is still too high, search below it
                high_z = mid_z
            else:
                break

        return mid_z


    # Central Limit Theorem
    # The Central Limit Theorem says that a random variable defined as the average of a large number of indpendent and identically
    # distributed random variables is itself approximately normally distributed.

    # If there are x1, x2, x3... xn random variables with mean u and standard deviation o and if n is large then
    # (1/n)(x1 + x2 + ... xn) is approximately normally distributed with mean u and standard deviation o/sqrt(n)
    # Also represented as ((x1 + x2 + .. xn) - (u)(n)) / o(sqrt(n))
    # Is approximately normally distributed with mean 0 and standard deviation 1.

    # One way to illustrate this is by looking at binomial random variables, which have two parameters n and p.
    # A Binomial(n, p) random variable is the sum of n independent Bernoulli(p) random variables, each of which equals 1 with probability
    # p and 0 with probability 1 - p.

    def bernoulli_trial(self, p):
        return 1 if random.random() < p else 0

    def binomial(self, n, p):
        return sum(self.bernoulli_trial(p) for _ in range(n))

    # The mean of the bernoulli(p) variable is p and it standard deviation is sqrt(p(1 - p)).
    # The central limit theorem says that as n gets large, a binomial(n,p) variable is approximately a normal random variable
    # with mean u = n*p and standard deviation 0 = sqrt(n*p(1-p)).
    # If we plot both we can see the resemblance.

    def make_histogram(self, p, n, num_points):

        data = [self.binomial(n, p) for _ in range(num_points)]
    
        # Use a bar chart to show the actual binomial examples
        histogram = Counter(data)

        plt.bar([x - 0.4  for x in histogram.keys()],
                [v / num_points for v in histogram.values()],
                0.8,
                color = '0.75')
    
        mu = p * n
        sigma = math.sqrt(n * p * (1 - p))

        # use a line to show the normal approximation

        xs = range(min(data), max(data) + 1)
        ys = [self.normal_CDF(i + 0.5, mu, sigma) - self.normal_CDF(i + 0.5, mu, sigma)
              for i in xs]
        plt.plot(xs, ys)
        plt.title('Binomial Distribution vs. Normal Approximation')
        plt.show()


    # make_histogram(.75, 100, 10000)

    # The basis of this approximation is that if you want to know the probability that for example a fair coin turns up more than 60 heads
    # in 100 flips you can estimate it as the probability that Normal(50, 5) is greater than 60, which is easier than computing the binomial(100, 0.5) cdf.

