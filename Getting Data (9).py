from collections import Counter
import sys, re

# Getting Data


# stdin and stdout
# if you run python scripts through the command line you can pipe data through them using sys.stdin and sys.stdout

# here is a script that reads in lines of text and spits back out the ones that match a regular expression

# egrep.py

# sys.argv is the list of command-line arguments
# sys.argv[0] is the name of the program itself
# sys.argv[1] will be the regex specified at the command line
'''
regex = sys.argv[1]

# for every line passed into the script
for line in sys.stdin:

    if re.search(regex, line): # if it matches the regex, write to stdout
        sys.stdout.write(line)
'''

# line_count.py

'''
count = 0
for line in sys.stdin:
    count += 1

print count
'''

# If you wanted to count how many lines of a file included a windows you'd use

''' type someFile.txt | python egrep.py "[0-9]" | python line_count.py '''

# The | is a pipe character which means use the output of the left command as the input of the right command

# Here's a script that counts the words in its input and writes out the most common ones:

# pass in number of words as first argument

'''
try:
    num_words = int(sys.argv[1])
except:
    print "usage: most_common_words.py num_words"
    sys.exit(1)     # non zero exit code contains error

counter = Counter(word.lower()                          # lowercase words 
                  for line in sys.stdin                 
                  for word in line.strip().split()      # split on spaces
                  if word)                              # skip empty words

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")
'''

# Raw Input asks the user for a string of data and simply returns the string
'''
x = raw_input("What is your name? : ")
print "your name is %s" % x
'''
# Input()
# uses raw_input to read a string of data, evaluates it as if it was a Python program, then returns the values
# that results
'''
x = input("What are the first 10 perfect squares? ")
print [z*z for z in x]
'''
# Standard File Objects
'''
for line in sys.stdin:
    print line
# every line you write into the compiler gets printed

sys.stdout.write("Hello")
# prints hello

print "This is the name of the script: ", sys.argv[0]
print "Number of arguments in: ", len(sys.argv)
print "The arguments are: ", str(sys.argv)

prints 
This is the name of the script:  /Users/quinnsheridan/Desktop/Data Science From Scratch/Getting Data (9).py
Number of arguments in:  1
The arguments are:  ['/Users/quinnsheridan/Desktop/Data Science From Scratch/Getting Data (9).py']
'''

print sys.argv[1:]




