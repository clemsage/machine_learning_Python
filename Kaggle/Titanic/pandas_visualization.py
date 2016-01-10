import pandas as pd
import numpy as np
import pylab

def pretreatment(df):
    '''
    :param df:
    :return: data :
    '''

    # ====== Pandas DataFrame =======
    # print df  # All the data
    # print df.head(3)  # Only the first three rows
    # print df.dtypes  # Type of each column
    # print df.info()  # Number of rows, type and number of non-null elements for each column
    # print df.describe()  # Statistics about each column (count, mean, std, min, max...)

    # ====== Data Munging =====
    # print df['Age']  # Age column or df.Age
    # print df['Age'][0:10]  # Age column (10 first columns)
    # print df.Age.median()  # same for mean or other statistical functions
    # print df[['Age','Sex']] # Print some of the columns
    # print df[df.Age>60]  # Print the rows corresponding to passengers who are 60 years old or more
    # print df[df.Age > 60][['Age', 'Sex', 'Pclass']]  # Combine the two last tips
    # print df[df.Age.isnull()][['Sex','Age','Pclass']]  # Display the columns for which ages is not given
    # for i in range(1,4):  # Multiple criteria (operators:  'OR' : |,'AND': & ...)
    #    print u"Number of men in class no. %d : %d" % (i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)]))
    # df['Age'].hist()  # Histogram of the age column
    # df['Age'].dropna().hist(bins=16, range=(0, 80), alpha=0.5)  # dropna() : drop the missing values
    # pylab.xlabel('Age')
    # pylab.ylabel('Number of passengers')
    # pylab.suptitle('Histogram')
    # pylab.show()


    # ======== Data cleaning ========
    # df['Gender'] = 4  # Add a column 'Gender' with values 4 for each row
    # df['Gender'] = df['Sex'].map(lambda x: x[0].upper())  # 'M' for 'male' , 'F' for 'female'
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # em_dict = {}
    # for i, em in enumerate(df['Embarked'].dropna().unique()):
    #    em_dict[em] = i
    # df['int_Embarked'] = df['Embarked'].map(em_dict)
    median_ages = np.zeros((2, 3))  # 2 by 3 array
    df['AgeFill'] = df['Age']  # copy the age column
    for i in range(2):  # Loop over genders
        for j in range(3):  # Loop over Pclass
            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
            df.loc[(df['Age'].isnull()) & (df['Gender'] == i) & (df['Pclass'] == j+1), 'AgeFill'] = median_ages[i, j]

    df['AgeisNull'] = df['Age'].isnull().astype(int)
    # print df.describe()

    # ========== Feature engineering ============
    df['FamilySize'] = df['Parch'] + df['SibSp']
    df['Age*Pclass'] = df['AgeFill'] * df['Pclass']
    # df['Age*Pclass'].hist()
    # pylab.show()

    # ========== Final preparation ===========
    # Most ML algorithms require the data to be in array with non string elements
    # print df.dtypes[df.dtypes.map(lambda x: x == 'object')]  # object type means that it contains strings
    df = df.drop(['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId','Fare'], axis=1)

    # print df.info()

    data = df.values  # convert into a NumPy array
    # print data
    return data

train_data = pretreatment(pd.read_csv('data/train.csv', header=0))
test_data = pretreatment(pd.read_csv('data/test.csv', header=0))