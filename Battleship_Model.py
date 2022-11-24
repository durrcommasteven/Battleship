import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
    

class Battleship_Model(tf.keras.Model):
    def __init__(self, filters = 25):

        # Make sure to call the Layer's __init__ method
        # Without it, it will throw really strange errors  
        # that are hard to detect

        super(Battleship_Model, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters, (5, 5), activation='relu', input_shape=(10, 10, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None),
            tf.keras.layers.Reshape((10, 10)),
        ])

    def call(self, inputs):
        """AI is creating summary for call

        Args:
            input_board ([type]): [description]
        """

        # account for rotation symmetry
        transformed_inputs = [
            tf.experimental.numpy.rot90(inputs, k=rotations, axes = (-3, -2)) for rotations in range(4)
        ] + [
            tf.experimental.numpy.rot90(tf.experimental.numpy.flip(inputs, axis = -2), k=rotations, axes = (-3, -2)) for rotations in range(4)
        ]

        # collect corresponding outputs
        outputs = []
        for transformed_input in transformed_inputs:
            outputs.append(self.model(transformed_input))
        
        # un-rotate, un-flip, and combine outputs
        symmetrized_outputs = tf.reduce_mean([
            tf.experimental.numpy.rot90(output, k=-rotations, axes = (-2, -1)) for rotations, output in enumerate(outputs[:4])
            ] + [
            tf.experimental.numpy.flip(tf.experimental.numpy.rot90(output, k=-rotations, axes = (-2, -1)), axis = -1) for rotations, output in enumerate(outputs[4:], start=4)
        ], 0)

        return symmetrized_outputs


        