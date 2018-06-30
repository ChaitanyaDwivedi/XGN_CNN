import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

data = pickle.load(open("./Data/CMU2X_data.pickle", "rb"))
x = np.array(data['data'])
y = np.array(data['labels'])
x = x.reshape(x.shape[0], 1, 31, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

model = tf.keras.Sequential()
