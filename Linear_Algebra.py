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

    def vector_add(self, v, w):
        '''adds corresponding elements'''
        return [v_i + w_i for v_i, w_i in zip(v,w)]

    def vector_subtract(self, v, w):
        return [v_i - w_i for v_i, w_i in zip(v,w)]

    def vector_sum(self, vectors):
        return reduce(self.vector_add, vectors)

    def scalar_multiply(self, c, v):
        ''' c must be an int or float, not a list'''
        return [c * v_i for v_i in v]

    def vector_mean(self, vectors):
        total = len(vectors)
        vector_sum = self.vector_sum(vectors)
        return [x/total for x in vector_sum]


    '''def vector_mean2(self, vectors):
        total = len(vectors)
        return self.scalar_multiply(float(1/total), self.vector_sum(vectors))'''

    def dot(self, v,w):
        '''The dot product of two vectors is the sum of their componentwise products'''
        return sum(v_i * w_i for v_i, w_i in zip(v,w))

    # in mathematic magnitude is the size of a mathematical object which determines if it is smaller or larger than other objects of the same kind

    # if w has magnitude 1, the dot product measures how far the vector v extends in the w direction.
    # For example if w = [1, 0] then dot(v, w) is just the first component of v. Its the length of the vector you'd
    # get if you projected v onto w.

    def sum_of_squares(self, v):
        '''v_1 * v_1 + ... + v_n * v_n
        computer a vectors sum of squares'''
        return self.dot(v, v)

    def magnitude(self, v):
        return math.sqrt(self.sum_of_squares(v))

    # Now we have all the tools we need to compute the distance between two vectors

    def squared_distance(self, v,w):
        '''(v_1 - w_1) ** 2 + ... (v_n - w_n) ** 2'''
        return self.sum_of_squares(self.vector_subtract(v,w))

    def distance(self, v, w):
        return math.sqrt(self.squared_distance(v,w))




# Matrices
# Matrices are two dimensional collections of numbers
# Matrices are represented as lists of lists with the inner list representing a row

A = [[1,2,3],[4,5,6]] # Has 2 rows and 3 columns
B = [[1,2],[3,4],[5,6]] # Has 3 rows and 2 columns

class Matrix:

    def shape(self, A):
        num_rows = len(A)
        num_cols = len(A[0]) if A else 0
        return num_rows, num_cols

    def get_row(self, A, i):
        return A[i]

    def get_column(self, A, j): # fixed function
        num_columns = len(A)
        return [A[i][j] for i in range(num_columns)]

    # Create a matrix given its shape and a function for generating its elements. We can do this with nested
    # list comprehension

    def make_matrix(self, num_rows, num_cols, entry_fn):
        return [[entry_fn(i,j)                  # given i, create a list
                 for j in range(num_cols)]      # [entry(i,0),...
                 for i in range(num_rows)]      # create one list for each i

    def is_diagonal(self, i, j):
        '''1s on the diagonal 0s everywhere else'''
        return 1 if i == j else 0

#identity_matrix = make_matrix(5, 5, is_diagonal) # creates 5 by 5 matrix with 1s going diagonal

# Matrices are important because:
# 1) They can be used to represent a data set consisting of multiple vectors by considering each vector as a row on the matrix
# 2) We can use a N x K matrix to represnt a linear function that maps k-dimensional vectors to n-dimensional vectors
# 3) Matrices can be used to represent binary relationships such as edges on a network

'''
friendships = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

def is_friend(i, j):
    return 1 if (i,j) in friendships else 0

friend_matrix = make_matrix(9,9, is_friend) # creates friend matrix'''


















