import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer #for the age values 

data = pd.read_csv("Titanic-Dataset.csv")

# First, we start off by removing columns that we do not need
df = pd.DataFrame(data)
df.drop(columns=['PassengerId', 'Ticket', 'Name'], inplace=True) # Ticket column may have hidden information within it

#Now we convert necessary columns into one-hot encodings
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# To fill in the NaN values, I first need to check how probable it is that someone from 1st, 2nd, or 3rd class left from the place given their fare price

# Now, we one-hot encode the Embarked column
df = pd.get_dummies(df, columns=['Embarked']) 

# Then we fill in empty values with nan/unknown values
df['Cabin'] = df['Cabin'].fillna('unknown')
# My response to Timo's suggestion to simply average the age
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
# Now the Age is made to be dependent on ticket class and sex - but only the missing values are filled, while the originals are left untouched


# Here, we want to see if there are any relationships between Embarked, Pclass and Survival
df.groupby(['Embarked', 'Pclass'])['Survived'].mean()


#using the imputer
#imputer = KNNImputer(n_neighbors=5) # apparently risk of overfitting with more neighbors but we will see how it goes
#age_after = imputer.fit_transform(df[["Age"]])
#df["Age"] = age_after
#print(age_after)

#using the median of sex, class and age to fill in the values for the ages
#df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.mean()))
#print(df["Age"])

#print(len(df[(df["Sex"] == "female") & (df["Survived"] == 1)]))