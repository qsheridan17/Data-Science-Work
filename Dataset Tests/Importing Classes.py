''' To import a class'''

folder_name '/Users/quinnsheridan/Desktop/Data Science From Scratch' # identify a path to the folder the file is in

sys.path.insert(0, folder_name) # add that folder to the top of sys path

from Linear_Algebra import Linear_algebra # from file name import class

la = Linear_algebra() # create instance of class

x = [1,2,3]
y = [4,5,6]

la.vector_add(x,y) # utlize functions from class


'''
When creating a class make sure to put self as a value of every function in the class

for instance:

class school:

    def gradents(self, test_scores, participation):
        return (test_score + participation)/2

or else it will return an error stating that the function takes exactly 2 arguments (3 given)
'''
