import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def score(name, y_true, y_pred):
    # Return RMSE, MAE and R2 for one model as a dict (one row of a results table

    return{
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }