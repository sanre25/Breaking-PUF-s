import numpy as np
from sklearn.linear_model import LogisticRegression

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    
	# challenges are in ci's so first convert c to our X's
	X = np.cumprod( np.flip( 2 * X - 1 , axis = 1 ), axis = 1 )

	# Get the indices for the upper triangle, avoiding the diagonal
	i, j = np.triu_indices(len(X[1, ]), k=1)

    # Calculate the pairwise products
	pairwise_products = X[:, i] * X[:, j]

	# Concatenate the original X with the pairwise products
	feat = np.append(X, pairwise_products, axis = 1)
	return feat


################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

  # Use this method to train your model using training CRPs
  # X_train has 32 columns containing the challeenge bits
  # y_train contains the responses
  Xtrans = my_map(X_train)
  
  model = LogisticRegression(max_iter=20000,C=100)
  # Fit the model to your data
  model.fit(Xtrans, y_train)

  # Access the weight vector (coefficients)
  w = model.coef_.flatten()
  # Access the bias term
  b = model.intercept_

  # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
  # If you do not wish to use a bias term, set it to 0
  return w, b



#if __name__ == "__main__":