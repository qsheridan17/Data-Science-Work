# Two primary uses of data visualization - to explore data and to communicate data
from __future__ import division

# Making a simple plot 
from matplotlib import pyplot as plt
'''
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# to create aline chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color = 'green', marker = 'o', linestyle = 'solid')

# add a title
plt.title('Nominal GDP')

# add a lable to the Y and X axis
plt.ylabel("Billions of Dollars")
plt.xlabel("Period")

plt.show()
'''


# Making a Bar Chart
'''
movies = ['Annie Hall', 'Ben-Hur', 'Casablanca', 'Ghandi', 'West Side Story']
num_oscars = [5, 11, 3, 8, 10]

# Bars are default width 0.8 so we'll add 0.1 to the left coordinates so each bar is centered
xs = [i + 0.1 for i, _ in enumerate(movies)]
       
plt.bar(xs, num_oscars)

plt.ylabel("Number of Oscars")
plt.xlabel("Movie Title")
plt.title("Movies by Oscars")

# label x-axis with move names at bar centers
plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)

#plt.show()
'''


# Making a Histogram
'''
from collections import Counter

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)

plt.bar([x - 4 for x in histogram.keys()] # shifts each bar to the left by 4
        , histogram.values(), 8)          # gives each bar the corect height and each bar a width of 8

plt.axis([-5, 105, 0, 5]) # sets the x and y axis, always have the y axis start at 0

plt.xticks([10 * i for i in range(11)]) # x-axis labels at 10, 20..
plt.xlabel('Decile')
plt.ylabel('# of Students')
plt.title('Distribution of Grades')
plt.show()'''

# Bar Chart 2 

'''
mentions = [500, 505]
years = [2013, 2014]

plt.bar([2012.6, 2013.6], mentions, .8)
plt.xticks(years)
plt.ylabel("Number of times I heard someone say Data Science")
plt.ticklabel_format(useOffset = False)
plt.axis([2012.5, 2014.5, 499, 506])
plt.title("Look at the Yuuuuge increase!")
plt.show()'''

# Line Chart
'''
variance = [2 ** x for x in range(9)]
bias_squared = [256/(2**x) for x in range(9)]
total_error = [x + y for x,y in zip(variance, bias_squared)] # prints [256, 130, 68, 40, 68, 130, 257] 
xs = [i for i, _ in enumerate(variance)] # prints [0,1,2,3,4,5,6,7,8]

# You can make multiple calls to plt.plot
plt.plot(xs, variance, 'g-', label = 'variance') # green solid line
plt.plot(xs, bias_squared, 'r-.', label = 'bias^2') # red dot dashed line
plt.plot(xs, total_error, 'b:', label = 'total error') # blue dotted line

# Because we have assinged label for each line we have to create the legend
# loc =9 means top center 
plt.legend(loc=9)
plt.xlabel('model complexity')
plt.title('The Bias-Variance Tradeoff')
plt.show()'''

# Scatter Plot
'''
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy = (friend_count, minute_count), # put the label with its point
                 xytext = (5, -5), # slightly offset 
                 textcoords = 'offset points')

plt.title('Daily Minutes vs. Number of Friends')
plt.xlabel('# of Friends')
plt.ylabel('Daily minutes spent online')
plt.show()
'''
# Scatter Plot 2 

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes aren't comparable")
plt.xlabel('Test 1 Grade')
plt.ylabel('Test 2 Grade')
#plt.show()

# If you're scattering comparable variables you might get a misleading picture if you let matplotlib use the scale
# to fix this

plt.axis("equal")
plt.show()




