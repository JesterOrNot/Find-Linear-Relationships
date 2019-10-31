import tensorflow as tf
import numpy as np
def mathRelationshipID(x_data, y_data, number, epochsNum):
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)
    model.fit(x_data, y_data, epochs=epochsNum)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model.predict([number])
