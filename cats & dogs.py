import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Define settings.
useScaler = True
verbose = 0

DATADIR = "C:Users\shoekzem\Documents\TMAP literatuur\cats and dogs"
CATEGORIES = ['Dog', 'Cat']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GREYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

X = ['Dog']
y = ['Cat']

# Split the data set up into a training and a test set. We can do whatever we want with the training set, but we may only use the test set once, to check the performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data, scaling features. For most regression methods this has no influence, but for some it does.
if useScaler:
    scaler = StandardScaler() # Create a basic scaler object.
    scaler.fit(X_train) # Examine the data to find means and standard deviations.
    X_train = scaler.transform(X_train) # Normalize the data set.
    X_train = pd.DataFrame(X_train, columns=X.columns) # The scaler only returns the numbers, so we manually turn the numbers into a Pandas dataframe again, with the right column titles.

# Set up a validation set to tune the parameters on.
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Set up our own evaluation function.
def evaluateResults(title, model, Xt, Xv, yt, yv):
    pred_train = model.predict(Xt) # Predict values from the training set to determine the fit.
    pred_val = model.predict(Xv) # Predict values from the validation set to determine the fit.
    RMSE_train = np.sqrt(mean_squared_error(yt, pred_train))
    R2_train = r2_score(yt, pred_train)
    RMSE_val = np.sqrt(mean_squared_error(yv, pred_val))
    R2_val = r2_score(yv, pred_val)
    print("The performance of {} on the training and validation set is:".format(title))
    print("Training set: RMSE = {}, R2 = {}".format(RMSE_train, R2_train))
    print("Validation set: RMSE = {}, R2 = {}\n".format(RMSE_val, R2_val))

# Try linear regression.
model = LinearRegression() # Set up a regression model.
model.fit(Xt, yt) # Train the model on the training data.
evaluateResults('linear regression', model, Xt, Xv, yt, yv)
    