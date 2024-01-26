import numpy as np 
import pickle

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    ar = np.array([[2, 5, 0, 1, 0, 2, 2, 5, 6, 6, 0, 3, 2]])
    prd = loaded_model.predict(ar)
    print(prd)
    #p = prd[0]
    #p = p.tolist()
    #print(p.index(max(p)))
