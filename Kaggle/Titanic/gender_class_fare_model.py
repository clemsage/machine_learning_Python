import csv
import numpy as np

# Open the csv file containing the training set
csv_object = csv.reader(open('data/train.csv', 'rb'))
csv_object.next()  # Skip the header

data = []
for row in csv_object:
    data.append(row)

data = np.array(data)  # convert the list into an array

fare_celling = 40
data[data[0:, 9].astype(np.float) >= fare_celling, 9] = fare_celling - 1

fare_bracket_size = 10
number_of_price_brackets = fare_celling / fare_bracket_size

number_of_classes = len(np.unique(data[0:, 2]))

# Initialize the survival table with zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        women_only = data[(data[0:, 4] == "female")
                          & (data[0:, 2].astype(np.float) == i + 1)
                          & (data[0:, 9].astype(np.float) >= j * fare_bracket_size)
                          & (data[0:, 9].astype(np.float) < (j + 1) * fare_bracket_size)
        , 1]

        men_only = data[(data[0:, 4] == 'male')
                        & (data[0:, 2].astype(np.float) == i + 1)
                        & (data[0:, 9].astype(np.float) >= j * fare_bracket_size)
                        & (data[0:, 9].astype(np.float) < (j + 1) * fare_bracket_size)
        , 1]

        survival_table[0, i, j] = np.mean(women_only.astype(np.float))
        survival_table[1, i, j] = np.mean(men_only.astype(np.float))

# Convert the nan value to 0 (corresponding to empty classes of survival_table)
survival_table[survival_table != survival_table] = 0

# Final predictions (threshold : 50 %)
survival_table[survival_table > 0.5] = 1
survival_table[survival_table <= 0.5] = 0

# print survival_table

# Write our predictions in a new .csv file
test_open = open("data/test.csv", 'rb')
test = csv.reader(test_open)
test.next()

results_open = open("GenderClassModel.csv", 'wb')
results = csv.writer(results_open)
results.writerow(['PassengerID', 'Survived'])

for row in test:
    for j in xrange(number_of_price_brackets):

        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] >= fare_celling:
            bin_fare = number_of_price_brackets - 1
            break
        elif j * fare_bracket_size <= row[8] < (j + 1) * fare_bracket_size:
            bin_fare = j
            break

    if row[3] == 'female':
        results.writerow([row[0], int(survival_table[0, int(row[1])-1, bin_fare])])
    else:
        results.writerow([row[0], int(survival_table[1, int(row[1])-1, bin_fare])])

results_open.close()
test_open.close()