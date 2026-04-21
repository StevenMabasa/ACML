import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer #for the age values 

data = pd.read_csv("Titanic-Dataset.csv")


df = pd.DataFrame(data)
missing = pd.isnull(df)
print(missing.sum()) # data before using the imputer


#using the imputer
#imputer = KNNImputer(n_neighbors=5) # apparently risk of overfitting with more neighbors but we will see how it goes
#age_after = imputer.fit_transform(df[["Age"]])
#df["Age"] = age_after
#print(age_after)

#using the median of sex, class and age to fill in the values for the ages
#df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.mean()))
#print(df["Age"])

#print(len(df[(df["Sex"] == "female") & (df["Survived"] == 1)]))