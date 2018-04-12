# Linear Algebra
# Linear Algebra is the branch of mathematics that deals with vector spaces
# Vector spaces are objects that can be added together (to form new vectors) and
# that can be multiplied by scalars (numbers) to form new vectors
# Vectors are points in some finite dimensional space. They are a good way to represent numeric data

# For instance if you have heights, weights and ages of a large number of people you can treat
# your data as three dimensional vectors (height, weight, age)

import math

height_weight_age = [70, # inches,
                     170, # pounds,
                     40] # years

grades = [95, # exam 1 
          80, # exam 2
          75, # exam 3
          62] # exam 4

# Need to create arithmetic functions for vectors

class Linear_algebra:

    def __init__(self):
        self = self 
    def vector_add(v, w):
        '''adds corresponding elements'''
        return [v_i + w_i for v_i, w_i in zip(v,w)]

    def vector_subtract(v, w):
        return [v_i - w_i for v_i, w_i in zip(v,w)]

    def vector_sum(vectors):
        return reduce(vector_add, vectors)

    def scalar_multiply(c, v):
        return [c * v_i for v_i in v]

    def vector_mean(vectors):
        total = len(vectors)
        return vector_sum(vectors)/n

    def vector_mean2(vectors):
        total = len(vectors)
        return scalar_multiply(1/total, vector_sum(vectors))

    def dot(v,w):
        '''The dot product of two vectors is the sum of their componentwise products'''
        return sum(v_i * w_i for v_i, w_i in zip(v,w))

    # if w has magnitude 1, the dot product measures how far the vector v extends in the w direction.
    # For example if w = [1, 0] then dot(v, w) is just the first component of v. Its the length of the vector you'd
    # get if you projected v onto w.

    def sum_of_squares(v):
        '''v_1 * v_1 + ... + v_n * v_n
        computer a vectors sum of squares'''
        return dot(v, v)

    def magnitude(v):
        return math.sqrt(sum_of_squares(v))

    # Now we have all the tools we need to compute the distance between two vectors

    def squared_distance(v,w):
        '''(v_1 - w_1) ** 2 + ... (v_n - w_n) ** 2'''
        return sum_of_squares(vector_subtract(v,w))

    def distance(v, w):
        return math.sqrt(squared_distance(v,w))




# Matrices
# Matrices are two dimensional collections of numbers
# Matrices are represented as lists of lists with the inner list representing a row

A = [[1,2,3],[4,5,6]] # Has 2 rows and 3 columns
B = [[1,2],[3,4],[5,6]] # Has 3 rows and 2 columns

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in J]

# Create a matrix given its shape and a function for generating its elements. We can do this with nested
# list comprehension

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i,j)                  # given i, create a list
             for j in range(num_cols)]      # [entry(i,0),...
             for i in range(num_rows)]      # create one list for each i

def is_diagonal(i, j):
    '''1s on the diagonal 0s everywhere else'''
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal) # creates 5 by 5 matrix with 1s going diagonal

# Matrices are important because:
# 1) They can be used to represent a data set consisting of multiple vectors by considering each vector as a row on the matrix
# 2) We can use a N x K matrix to represnt a linear function that maps k-dimensional vectors to n-dimensional vectors
# 3) Matrices can be used to represent binary relationships such as edges on a network

friendships = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

def is_friend(i, j):
    return 1 if (i,j) in friendships else 0

friend_matrix = make_matrix(9,9, is_friend) # creates friend matrix


















