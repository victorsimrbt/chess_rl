from q_network import *
import pickle

with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
    
with open('y.pkl', 'rb') as f:
    y = pickle.load(f)
    
print(X.shape,y.shape)

model = Q_model()
model.model.fit(X,y)