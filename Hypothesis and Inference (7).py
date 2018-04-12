from __future__ import division
import math

# Hypothesis and Inference

# Data science often involves forming and testing hypthesis about our data and the processes that generate it

# Statistical Hypothesis Testing
# In a classical setup we have a null hypothesis H0 that represents some default position and some alternative hypothesis H1
# that we'd like to compare it with.
# We use statistics to decide whether we can reject H0 as false or not.

# Example - Flipping a coin
# We have a coin and want to test if its fair
# We will make the assumption that the coin has some probability p of landing heads, so our null hypothesis is that p = 0.5.
# The alternative hypothesis is that p != 0.5.
# The test will involve flipping a coin some number of times and counting the number of heads x.
# Each coin flip is a Bernoulli trial
# In statistics a Bernoulli trial is a random trial with exactly two possible outcomes, success or failure in which the probability
# of success is the same every time the experiment is conducted

# function from previous chapter:

# Sometimes we will need to invert normal_CDF to find the value corresponding to a specified probability

class hypothesis:

    def inverse_normal_CDF(self, p, mu = 0, sigma = 1, tolerance = 0.00001):
        '''find appropriate inverse using binary search.
        This function repeatedly bisects intervals until it narrows in on a Z
        thats close enough to the desired probability'''

        # if not standard, compute standard and rescale
    
        if mu != 0 or sigma != 1:
            return mu + sigma * self.inverse_normal_CDF(p, tolerance = tolerance)

        low_z = -10.0      # normal_CDF(-10) is very close to 0
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


    # this function returns mu and sigma
    # mu is the average
    # sigma is the standard deviation
    def normal_approximation_to_binomial(self, n, p):
        ''' finds mu and sigma corresponding to a binomial(n, p)
            n represents the number of times the probably is tested (such as # of coin flips)
            p represents the probability (such as 0.5 for a coin flip)'''
        mu = p * n
        sigma = math.sqrt(p * (1 - p) * n)
        return mu, sigma

    # normal_distribution is the classic bell shaped curve distirbution and is determined by two parameters
    # its mean u (mu) and its standard deviation (sigma)
    # CDF describes the distribution of a discrete random variable
    def normal_CDF(self, x, mu = 0, sigma =1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


    # The normal_CDF is the probability that the variable is below a threshold
    '''normal_probability_below = normal_CDF'''

    # Its above the threshold if its not below the threshold
    def normal_probability_above(self, lo, mu=0, sigma=1):
        return 1 - self.normal_CDF(lo, mu, sigma)

    # If its less than hi but not less than lo
    def normal_probability_between(self, lo, hi, mu=0, sigma=1):
        return self.normal_CDF(hi, mu, sigma) - self.normal_CDF(lo, mu, sigma)

    # Its outside if its not between
    def normal_probability_outside(self, lo, hi, mu=0, sigma=1):
        return 1 - self.normal_probability_between(lo, hi, mu, sigma)

    # We can also do the reverse - find either the nontail region or the (symmetric) interval around the mean that accounts for a
    # certain level of likelihood.

    def normal_upper_bound(self, probability, mu=0, sigma=1):
        '''returns the z for which P(Z <= z) = probability'''
        return self.inverse_normal_CDF(probability, mu, sigma)

    def normal_lower_bound(self, probability, mu=0, sigma=1):
        '''returns the z for which P(Z >= z) = probability)'''
        return self.inverse_normal_CDF(1 - probability, mu, sigma)

    def normal_two_sided_bounds(self, probability, mu=0, sigma=1):
        '''returns symmetric about the mean bounds that contain the specified probability'''
        tail_probability = (1 - probability)/2

        #upper bound should have the tail probability above it
        upper_bound = self.normal_lower_bound(tail_probability, mu, sigma)

        # lower bound should have tail probability below it
        lower_bound = self.normal_upper_bound(tail_probability, mu, sigma)

        return lower_bound, upper_bound



    # If we choose to flip a coin about 1000 times (n = 1000) X should be distiributed approximately noramlly wtih mean 500 and standard
    # deviation 15.8

    '''mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)'''

    # We need to make a decision about significance, how willing we are to make type 1 error 'false positive' in which we reject H0 even
    # though its true. This willingness is often set at %5 or 1%. We will chose %5.

    '''self.normal_two_sided_bounds(0.95, mu_0, sigma_0) # returns (469, 531)'''

    # Assuming P really equals 0.5 there is just a 5% chance we observe an X that lies outside this interval, which is the exact
    # significance we wanted.
    # In other words if H0 is true then approximately 19 out of 20 times the test will give the correct result

    # We are also interested in the power of a test which is the probability of making a type 2 error.
    # This is when we fail to reject H0 even though its false.
    # In order to measure this we have to sepcify what exactly H0 being false means.
    # In particular lets see what happens if p is really 0.55 and the coin is slightly biased torwards heads.

    # We calculate the power of the test:

    # 95% bounds based on the assumption that p is 0.5
    '''lo, hi = self.normal_two_sided_bounds(.95, mu_0, sigma_0)'''

    # actual mu and sigma is based on p = 0.55
    '''mu_1, sigma_1 = self.normal_approximation_to_binomial(1000, 0.55)'''

    # a type 2 error means we fail to reject the null hypothesis, which will happen when x is still in our original interval
    '''type_2_probability = self.normal_probability_between(lo, hi, mu_1, sigma_1)
    power = 1 - type_2_probability'''
    # power = .887

    # Imagine that our null hypothesis is that the coin is not biased torwards heads or that p <= 0.5.
    # In this case we want a one sided test that rejects the null hypothesis when X is much larger than 500 but not
    # when X is smaller than 500.

    # The significance test uses normal_probability_below to find the cutof below which 95% of the probability lies

    '''hi = self.normal_upper_bound(0.95, mu_0, sigma_0)'''
    # 526 (< 531 since we need more probability in the upper tail)

    '''type_2_probability = self.normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type_2_probability # 0.936'''

    # This is a much more powerful test because it no longer rejects H0 when X is below 469 (which is very unlikely to happen) and instead
    # rejects H0 when X is between 526 and 531 (which is somewhat likely to happen if H1 is true).





    # P-Values

    # An alternative way of thinking about the proceeding test involves using p-values. Instead of using bounds based on a
    # probability cut-off we compute the probability assuming H0 is true - that we would see a value as extreme as the one we actually
    # observed.

    def two_sided_p_value(self, x, mu=0, sigma=1):
        if x >= mu:
            # if x is greater than or equal to the mean, that tail is whats greater than x.
            return 2* self.normal_probability_above(x, mu, sigma)
        else:
            # if x is less than the mean, the tail is whats less than x.
            return 2* self.normal_probability_below(x, mu, sigma)

    # If we were to see 530 heads we would compute
    self.two_sided_p_value(529.5, mu_0, sigma_0) # returns 0.062
    # Because .062 is greater than 5% we don't reject our null hypothesis 
    # If the result was 0.042 than the we would reject the null hypothesis because it is greater than 5%

    # We used 529.5 because of whats called a continuity correction. It reflects the fact that normal_probability_between(529.5, 530.5, mu_0, sigma_0)
    # is a better estimate of seeing the probability of of seeing 530 heads.

    # Make sure your data is normally distributed before useing normal_probability_above to compute p-values




    # Confidence Intervals

    # We've been testing about the value of the heads probability p, which is a parameter of the unknown heads distribution.
    # A 3rd approach is to construct a confidence interval around the value of the observed parameter.





    # P-Hacking

    # A procedure that erroneously rejects the null hypothesis only 5% of the time will by definition 5% of the time erroneously
    # reject the null hypothesis

    # This means if your setting out to find significant results you typically can.




    # Running an A/B Test

    # Example: Test between two ads to see which one performs better
    # We determine this using statistical inference

    # Lets say NA people see ad A and nA of them click on it.
    # Each add can be viewed as a Bernoulli Trial where PA is the probability somone click add A.
    # If NA is large, we know that nA/NA is approximately a normal random variable with mean PA and standard deviation sqrt(PA(1-PA)/NA)
    # Similarly nB/NB is a normal random variable with mean PB and standard deviation sqrt(PA(1-PA)/NA)

    def estimated_parameters(self, N, n):
        p = n/N
        sigma = math.sqrt(p*(1-p)/N)
        return p, sigma

    # If we assume those two normals are independent then their difference should also be normal with mean PB-PA and standard deviation
    # sqrt((oA^2)+(oB^2))

    # We are testing the null hypothesis that PA and PB are the same (That is PA - PB = 0) using this statistic:

    def a_b_test_statistic(self, N_A, n_A, N_B, n_B):
        p_A, sigma_A = self.estimated_parameters(N_A, n_A)
        p_B, sigma_B = self.estimated_parameters(N_B, n_B)
        return (p_B - p_A) / math.sqrt(sigma_A**2 + sigma_B**2)

    self.a_b_test_statistic(1000,200,1000,180) #-1.14

    self.two_sided_p_value(-1.14) # 0.254, which is large enough to conclude there is not much of a difference

    # On the other hand if the other add got 150 clicks we'd have
    
    self.a_b_test_statistic(1000,200,1000,150) #-2.95

    self.two_sided_p_value(-2.95) # .003, which means theres only a .003 probability you'd see such a large difference if the ads were equally effective





    # Bayesian Inference

    # The procedures above have provided probability statements our tests.
    # For instance: There's only a 3% chance you'd observe such an extreme statistic if our null hypothesis were true

    # An Alternative Approach:
    # - Treats all the unknown parameters as random variables
    # - The analyst starts with a prior distribution for the parameters
    # - Then uses the observed data and Bayes's Theorem to get an updated posterior distribution for the parameters

    # The outcome is rather than making judgements about the tests your making judgement about the parameters themselves

    # When the unknown parameter is a probability we use Beta distribution

    def B(self, alpha, beta):
        '''normalizing a constant so the entire probability is 1'''
        return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

    def beta_PDF(self, x, alpha, beta):
        if x <= 0 or x >= 1: # no weight outside [0,1]
            return 0
        else:
            return x ** (alpha - 1) * (1 -x) ** (beta - 1) / B(alpha, beta)

    # if alpha = beta the uniform distribution is .5 
    # if alpha > beta the uniform distribution is near 1 
    # if beta > alpha the uniform distribution is near 0 

    # Bayesian Inference allows us to make probability statements about hypotheses
    # For instance "Based on the observed data, there is only a 5% likelihood the coin's head probability is between 49% and 51%"





