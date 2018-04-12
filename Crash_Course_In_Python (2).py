from __future__ import division

5/2   #division has decimal returns now 

#create multiline strings

multi_line_string = """this is
the first
of mmany
linesssssss"""

# Exceptions
"""
try:
    print 0/0
except ZeroDivisionError:
    print "cannot divide by zero"
"""

# to get the last element of list use [-1]

int_list = [1,2,3,4,5,6,7,8,9,10]
int_list[-1] #gets the last element of the list

# slicing lists

first_four = int_list[:4] # gives the first four elements of the list 

fourth_to_last = int_list[4:] # gives the fifth to last element 

one_to_four = int_list[1:5] # prints second to fifth elements in a list

last_four = int_list[-4:] # prinst last four items in a list 

without_second_and_second_to_last = int_list[2:-2]

all_items = int_list[:]

# concatenate lists together

int_list.extend([11,12,13,14]) # extends the list by these four elements

int_list.append(15) # adds one item at a time

# assign variables to lists

new_list = [1,2]
x,y = new_list

# Tuples

my_tuple = (1,3)

# Tuples are immutable
'''
try:
    my_tuple[1] = 4
except TypeError:
    print "cannot change a tuple"
    '''

# Dictionaries

new_dict = {'Quinn' : 1, "Annie" : 2}

'''
try:
    new_dict["Ralph"]
except KeyError:
    print "no key called ralph in dict"
'''

# Dictionaries have a get method that returns a default value (instead of raising
# an exception when you look up a key thats not in the dictionary.

key_value = new_dict.get("Quinn", 0) # returns value of key
key_val = new_dict.get("Ralph", 0) # returns default value of zero if no key

new_dict.keys() #list of keys 
new_dict.values() # list of values 
new_dict.items() # list of (key, val) tuples

from collections import defaultdict #very useful dictionary tool

from collections import Counter
c = Counter([0,1,2,0])

# say you had a count of the most common words in a text file
# you could use counters most_common method to print them

# for word, count in word_counts.most_common(10):
#   print word, count

# Sets is a data structure which represents a collection of distinct elements

s = set()
s.add(1)
s.add(2)

# its very fast to search sets for an element

# Control Flow

x = 2
parity = "is even" if x % 2 == 0 else "is odd"


# Booleans in python work as in other languages except they are capitalized
# and instead of null they use None to represent no value

# Python lets you use any value where it expects a boolean. The following are
# falsy:
#   False
#   None
#   [] empty list
#   {} empty dict
#   ""
#   set()
#   0
#   0.0

all([False, set(), 0.0]) # Python has an all function which returns True when
                        # every object is truthy, otherwise falsy

all([1, [1], True, 'A']) # prints true

# Sorting

y = [1,19,8,3,25,14,-3]
x = sorted(y) 
y.sort() # both sort the list to [-3, 1, 3, 8, 14, 19, 25]

# To sort from largest to smallest

x = sorted(y, cmp = None, key = None, reverse = True) # does a reverse sort

# List Comprehensions

odd_numbers = [x for x in range(10) if x % 2 == 1]
odd_number_squares = [x * x for x in odd_numbers]

zeroes = [0 for _ in odd_numbers] # prints five zeros

# list comprehension can use multiple for loops

odd_number_pairs = [(x,y)
                    for x in odd_numbers
                    for y in odd_number_squares] # prints a list of every pair value

# A generator is something you can iterate over but whose values are
# only produced as needed (lazily)

def lazy_range(n):
    """ A lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1

for i in lazy_range(10): # the loop will consume the yielded values one at a time until none our left 
    i 

# With lazy sequences you can create an infinite sequence however you can
# only iterate over it once.

lazy_odds_below_20 = (x for x in lazy_range(20) if x % 2 == 1)

# Randomness

import random

five_randoms = [random.random() for _ in range(5)]
random.random() # creates numbers between 0 and 1

# The random module produces psuedorandom numbers based on an internal state
# that you can set with random.seed if you want to get reproducable results

random.seed(12)
random.random()
random.seed(12)
random.random() # produces .4745706... both times

# We can use random.randrange to return an element from the corresponding range

random.randrange(10) # choose randomly from list betweeen [0-9]
random.randrange(4,7) # chhose randomly from range [4,5,6]

# to pick an element randomly from a list you can use randomchoice

random.choice(['a','b','c'])

# you can use random.sample to avoid duplicates and random.replacement
# to include duplicates

# Regular Expression
# They provide a way of searching text

import re

re.match("a","cat") # cat does not start with a 
re.search("a","cat") # cat has an a in it 
re.sub("[0-9]", "-", "R2D2") # Replaces digits with dahes

# Creating a class in Python
'''
class Set:

    def __init__(self, values = none):
        s1 = set()
        s2 = set([1,2,2,3])
        self.dict = {}
        for values is not none:
            for value in values:
                self.add

    def __repr__(self):
        return "Set: " + str(self.dict.keys())
'''

# Functional Tools
# This is used to partially apply (or curry) functions to create new functions

def exp(base, power):
    return base ** power

def two_to_the(power):
    return exp(2, power)

from functools import partial
two_to_the = partial(exp, 2)

square_of = partial(exp, power = 2)
square_of(3) # prints 9

# Using map, reduce and filter for functional alternatives

def triple(x):
    return x * 3

lst = [1,2,3,4]
twice_list = [triple(x) for x in lst]
twice_list = map(triple, lst)
double_list = partial(map, triple)
twice_list = double_list(lst)

# you can use map to create multiple argument functions

def square(x, y):
    return x**y

square_list = map(square, [2,4,6,4],[4,5,7,9]) # prints [16, 1024, 279936, 262144]

# filter does a lot of list comprehension

def is_even(x):
    if x % 2 == 0:
        return True

x_even = [is_even(x) for x in [1,2,3,4,5,6]] # returns [none, true, none, true, none, true]
x_evens = [x for x in [1,2,3,4,5,6] if is_even(x)] # returns [2,4,6]
x_even = filter(is_even, [1,2,3,4,5,6]) # returns [2,4,6]

# reduce combines the elements of a list

def multiply(x,y):
    return x * y

num_list = [1,2,3,4,5,6]
x_reduce = reduce(multiply, num_list) # 720 

# Enumerate
# Used to iterate over a list - using both its elements and their indexes

pet_list = ['dog', 'cat', 'fish', 'gerbil', 'lizard']
list(enumerate(pet_list)) # prints [(0, 'dog'), (1, 'cat'), (2, 'fish'), (3, 'gerbil'), (4, 'lizard')]

# Zip and Argument Unpacking
# Zip transforms multiple lists into a single list of tuples of corresponding elements

list1 = ['Annie', 'Quinn', 'Dash']
list2 = [21, 23, 11]
zipped = zip(list1, list2) # Prints [('Annie', 21), ('Quinn', 23), ('Dash', 11)]

# You can unzip
names, ages = zip(*zipped)

# You perform argument unpacking
unzipped = zip(*zipped) # Prints [('Annie', 'Quinn', 'Dash'), (21, 23, 11)]
# The * performs argument unpacking which uses the elements of pairs as individual arguents

# The * initiliazes it to a tuple
# The ** initiliazes it to a dictionary

# Args and Kwargs

def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

def f1(x):
    return x + 1

g = doubler(f1)
g(3) # prints (3 + 1) * 2 = 8

def f2(x,y):
    return x + y

g = doubler(f2),
# g(1, 2) This would give a typeError because it takes two arguments and one was given

def magic(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs

magic( 1, 2, key="word", key2= "word2")
# Prints - unnamed args (1,2)
# keyword args: {'key2', 'word2', 'key': 'word'}

# args is a tuple of its unnmaed arguments
# kwargs is a dictionary of its named arguments

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1,2]
z_dict = {'z':3}
other_way_magic(*x_y_list, **z_dict)

def doubler_correct(f):
    '''works no matter what inputs x expects'''
    def g(*args, **kwargs):
        '''return whatever arguments g is supplied and pass them through f'''
        return 2 * f(*args, **kwargs)
    return g 

# Args and Kwargs let you pass a variable amount of arguments to a function.
# Args is used to send a non-keyworded variable length argument list to the function
# arg('1','2','3')
# Kwargs is used to send a keyworded variable length argument list to the function
# Kwarg(name = 'yasoob', name = 'dadoood') 
