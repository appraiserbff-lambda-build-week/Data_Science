import os
import pickle
import numpy as np 

house_features = [3.0, 2.0, 1700.0, 7576.5, 241.0, 4, 39.0, 89128]

def estimate_value(house_features):
	x = np.array(house_features).reshape(1, -1)
	s = pickle.load(open('regressor.pkl', 'rb'))
	price_estimate = s.predict(x)
	return price_estimate[0]


