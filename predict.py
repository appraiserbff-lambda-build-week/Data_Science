import pickle
import numpy as np

"""
This is the trained model. It requires the following inputs as a list:

 Inputs: 

'TOTAL_LOT_AREA_SQFT', 'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 
'BATHS', 'TOTAL_ROOMS', 'ZIP', 'ASSESSED_PROPERTY_TAXES', 
'YEAR_BUILT','SQAURE_FEET_HOUSE'

 Output;

 APPRAISED_VALUE 

 Example: house_features=[ 7736.0, 261, 2015, 4, 2.0, 0, 96505, 3402.94, 1963, 1288.0]
 array([248182.87052798])

 Details: 

 TOTAL_LOT_AREA_SQFT        float64
 LAND_USE_TYPE                int64
 YEAR_ASSESSMENT              int64
 BEDS                         int64
 BATHS                      float64
 TOTAL_ROOMS                  int64
 ZIP                          int64
 ASSESSED_PROPERTY_TAXES    float64
 YEAR_BUILT                   int64
 SQAURE_FEET_HOUSE          float64
 APPRAISED_VALUE            float64

 The Model: 
 
 RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=7,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
           oob_score=False, random_state=42, verbose=0, warm_start=False)

"""


def estimate_value(house_features):
	x = np.array(house_features).reshape(1,-1)
	m = pickle.load(open('zillow_model.p', 'rb'))
	appraised_value = m.predict(x)
	return appraised_value[0]