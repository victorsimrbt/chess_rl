from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D, Dense
from keras.layers import add, BatchNormalization, Flatten
from keras.losses import CategoricalCrossentropy,BinaryCrossentropy
import tensorflow as tf
from board_conversion import *
from variable_settings import *
class Q_model():
    def __init__(self,model = None):
        if model:
            print('CUSTOM MODEL SET')
            self.model = model
        else:
            self.model = self.create_q_model()

    def create_q_model(self):
        def residual_module(layer_in, n_filters):
            merge_input = layer_in
            if layer_in.shape[-1] != n_filters:
                merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
            conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
            batch_norm = BatchNormalization()(conv1)
            layer_out = add([batch_norm, merge_input])
            layer_out = Activation('relu')(layer_out)
            return layer_out

        visible = Input(shape=(17, 8, 8,12))
        layer1 = residual_module(visible, 64)
        layer2 = residual_module(layer1, 64)
        layer3 = residual_module(layer2, 64)
        flatten = Flatten()(layer3)
        p = Dense(4096,activation='softmax', name = 'p')(flatten)
        v = Dense(1,activation = 'sigmoid', name = 'v')(flatten)
        
        model = Model(inputs=visible, outputs=[p,v])
        
        model.compile(optimizer = optimizer, loss = {'p':CategoricalCrossentropy(),
                                                     'v':BinaryCrossentropy()})
        return model
    
    def predict(self,position_memory):
        input_values = generate_input(position_memory)
        state_tensor = tf.convert_to_tensor(input_values)
        state_tensor = tf.expand_dims(state_tensor,0)
        p,v = self.model(state_tensor, training=False)
        return p,v
    
    def explore(self,env):
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board,action_space)
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action