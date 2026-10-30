import numpy as np
from src.evaluate import score
 
 
def test_perfect_predictions_have_zero_error():
    # Ensure that when predictions match the actual values perfectly, the evaluation metrics reflect absolute accuracy.
    y = np.array([1.0, 2.0, 3.0])
    # Passing the same array as both true and predicted values
    s = score("perfect", y, y)
    assert s["RMSE"] == 0 and s["MAE"] == 0 and s["R2"] == 1
 
 
def test_predicting_the_mean_gives_r2_of_zero():
    # Verify that using the mean value as a constant prediction results in an R-squared of zero (meaning the model performs no better than a horizontal baseline.
    y = np.array([1.0, 2.0, 3.0])
    # Predicting the exact mean of y for all data points
    s = score("mean", y, np.full(3, y.mean()))
    # R2 should be effectively 0
    assert abs(s["R2"]) < 1e-1