from __future__ import division
from functools import partial
import math
import csv
import random
from matplotlib import pyplot as plt
from collections import Counter
import sys


# importing special class
folder_name = '/Users/quinnsheridan/Desktop/Data Science From Scratch'
name = '/Users/quinnsheridan/Desktop/Data Science From Scratch/Datasets/winequality-red.csv'
sys.path.insert(0, folder_name)
from Gradient_Descent_8 import gradient_descent
gd = gradient_descent()


# opening dataset 
def remove_quotes(string):      # removes quotiation marks from data
    return string.replace('"', '')
readCSV = csv.reader(open(name, 'rU'), dialect=csv.excel_tab, delimiter = ';')
counter = 0


# lists of wine values
fixed_acidity = [] 
volatile_acidity = [] 
free_sulfur_dioxide = []
free_sulfur_dioxide2  = []


# populating wine lists from dataset 
for row in readCSV: 
    if counter == 0: 
        headers = row[0].split(';') 
        headers = map(remove_quotes, headers) # creates list with headers 
        #print headers
    if 0 < counter < 101:    
        data = [float(item) for item in row] # creates list of data points for each wine
        fixed_acidity.append(data[0])
        volatile_acidity.append(data[1])
        free_sulfur_dioxide.append(data[5])

        
    counter += 1












