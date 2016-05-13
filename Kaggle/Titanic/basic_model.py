import csv
import numpy as np

# Open the csv file containing the training set
csv_object = csv.reader(open('data/train.csv','rb'))
csv_object.next()  # Skip the header

data = []
for row in csv_object:
    data.append(row)

data = np.array(data)  # convert the list into an array
# print data[0, 3]  # print 1st row, 4th column

# Global analysis
number_passengers = np.size(data[0:, 1].astype(np.float))
number_survivors = np.sum(data[0:, 1].astype(np.float))
proportion_survivors = number_survivors/number_passengers
# print proportion_survivors

# Gender analysis
women_only = data[0:, 4] == 'female'
men_only = data[0:, 4] != 'female'
women_on_board = data[women_only,1]
men_on_board = data[men_only,1]

proportion_women_survived = np.sum(women_on_board.astype(np.float))/np.size(women_on_board.astype(np.float))
proportion_men_survived = np.sum(men_on_board.astype(np.float))/np.size(men_on_board.astype(np.float))

# print proportion_men_survived,proportion_women_survived

# Open the csv file containing the test set
test_set_open = open('data/test.csv','rb')
test_set = csv.reader(test_set_open)
test_set.next()

# Write our predictions for the test set into a new .csv file
results_open = open('GenderBasedModel.csv','wb')
results = csv.writer(results_open)

results.writerow(['PassengerID','Survived'])
for row in test_set:
    if row[3] == 'female':
        results.writerow([row[0],'1'])  # Survived if the passenger is a female
    else:
        results.writerow([row[0],'0'])   # Didn't survived if the passenger is a male

test_set_open.close()
results_open.close()









