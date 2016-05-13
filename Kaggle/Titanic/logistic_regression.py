import pandas_visualization, csv
from sklearn import linear_model

# Create the random forest object which will include all the parameters for the fit
logreg = linear_model.LogisticRegression(C=1e5)

# Fit the training data to the Survived Label and create the decision trees
logreg = logreg.fit(pandas_visualization.train_data[0:,1:],pandas_visualization.train_data[0:,0])

# Take the same decision trees and run it on the test data
output = logreg.predict(pandas_visualization.test_data)

# Write the output in a .csv file
with open('logistic_regression_model.csv','wb') as results_open :
    results = csv.writer(results_open)
    results.writerow(['PassengerId','Survived'])

    for i, survived in enumerate(output):
        results.writerow([range(892,1310)[i],int(survived)])