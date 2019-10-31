import tensorflow as tf
import numpy as np
def mathRelationshipID(x_data, y_data, number,epochsNum):
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.array(x_data, dtype=float)
    ys = np.array(y_data, dtype=float)
    model.fit(x_data, y_data, epochs=epochsNum)
    return model.predict(number)
print(mathRelationshipID([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],[-2.0, 1.0, 4.0, 7.0, 10.0, 13.0],100,4000))