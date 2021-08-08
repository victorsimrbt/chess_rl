from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Model, Input
import tensorflow as tf
from board_conversion import *

class Q_model():
    def __init__(self):
        self.model = self.create_q_model()

    def create_q_model(self):
    # Network defined by the Deepmind paper
        input_layer = Input(shape=(8, 8, 12))

        # Convolutions on the frames on the screen
        x = Conv2D(filters=64,kernel_size = 2,strides = (2,2))(input_layer)
        x = Conv2D(filters=128,kernel_size=2,strides = (2,2))(x)
        x = Conv2D(filters=256,kernel_size=2,strides = (2,2))(x)
        x = Flatten()(x)

        action = Dense(4096,activation = 'softmax')(x)
        return Model(inputs=input_layer, outputs=action)
    
    def predict(self,env):
        state_tensor = tf.convert_to_tensor(env.translate_board())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action_space = filter_legal_moves(env.board,action_probs[0])
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action
    
    def explore(self,env):
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board,action_space)
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action
        