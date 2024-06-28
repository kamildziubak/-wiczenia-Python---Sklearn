import numpy as np
import pandas as pd
słowniczek={
    "Kraj": ["Argentyna", "Anglia", "Francja", "Niemcy", "Kolumbia", "USA", "Chiny"],
    "Płeć": ["Kobieta", "Mężczyzna", "Profesor", "Kobieta", "Mężczyzna", "Profesor", np.nan],
    "Auto": ["Mercedes", "Audi", "Skoda","Jeep","Audi","Skoda","Opel"],
    "Dochód": [25000,np.nan,40000 ,35000,60000,np.nan,55000],
    "Zawód": ["Informatyk","Lekarz","Malarz","Pisarz","Kominiarz","Informatyk", "Piłkarz"],
    "Pochodzenie": ["Ziemskie","Aldebarańskie","Ziemskie","Aldebarańskie","Ziemskie","Aldebarańskie","Ziemskie"]
}

df = pd.DataFrame(słowniczek)
df

df1 = df[["Kraj", "Zawód", "Auto"]]

df1_kod = pd.get_dummies(df1)
df1_kod

df2 = df[["Płeć"]]
df2

from sklearn.impute import SimpleImputer
si1 = SimpleImputer(strategy  = "constant", fill_value="Kobieta" )

df2_ = si1.fit_transform(df2)
df2_

df2_bez = pd.DataFrame(df2_)
df2_bez

df2_kod = pd.get_dummies(df2_bez)
df2_kod

si2 = SimpleImputer(strategy= "mean")
df3 = df[["Dochód"]]

df3_bez= si2.fit_transform(df3)

df3_bez

df3_kod= pd.DataFrame(df3_bez, columns = ['Dochód'])
df3_kod

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df4 = df[["Pochodzenie"]]

le = LabelEncoder()
df4_kod = pd.DataFrame(le.fit_transform(df4), columns=['Pochodzenie'])
df4_kod

NR = pd.concat([df1_kod, df2_kod, df3_kod, df4_kod], axis = 1)
NR

X = NR.drop('Pochodzenie', axis = 1)

y = NR['Pochodzenie']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.25)

mmsc = MinMaxScaler ()
Xtrain = mmsc.fit_transform(Xtrain)
Xtest = mmsc.transform(Xtest)