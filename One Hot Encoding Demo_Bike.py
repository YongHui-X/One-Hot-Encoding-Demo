import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
#LabelEncoder only works on one column at a time, 1d array
df = pd.read_csv("bike_types_times.csv")

encoder = LabelEncoder()
# Fit only on the column you want to encode
labels = encoder.fit_transform(df["Road_Bike_Type"]) #includes encoder.fit

z = tf.keras.utils.to_categorical(labels)

df_onehot = pd.DataFrame(data=z,columns=encoder.classes_)

print(df_onehot.head())
