import pandas_visualization, csv
from sklearn.ensemble import RandomForestClassifier

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators=10)

# Fit the training data to the Survived Label and create the decision trees
forest = forest.fit(pandas_visualization.train_data[0:,1:],pandas_visualization.train_data[0:,0])

# Take the same decision trees and run it on the test data
output = forest.predict(pandas_visualization.test_data)

# Write the output in a .csv file
with open('random_forests_model.csv','wb') as results_open :
    results = csv.writer(results_open)
    results.writerow(['PassengerId','Survived'])

    for i, survived in enumerate(output):
        results.writerow([range(892,1310)[i],int(survived)])


