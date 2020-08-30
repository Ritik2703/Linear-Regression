import numpy as np 
import matplotlib.pyplot as plt 

#This function will be used to find the value of y intercept and slope
def coef_estimation(x, y): 
    # number of observations/points
    n = np.size(x) 
  
    # Find mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y) 
  
    # Use sum of square to find out the intercept and slope
    
    SS_xy = np.sum(y*x) - (n * y_mean * x_mean)
    SS_xx = np.sum(x*x) - (n * x_mean * x_mean)
  
    # calculating regression coefficients 
    slope = SS_xy / SS_xx 
    y_intercept = y_mean - slope*x_mean
  
    return(y_intercept, slope) 

def regression_line_plot(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b") 
  
    # calculating predicted response from y = mx + c
    y_pred = b[1]*x + b[0]
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # adding labels to the Axes
    plt.xlabel('X-Axis') 
    plt.ylabel('Y-Axis') 
  
    # function to show plot 
    plt.show() 
    
# observations 
x = np.array([1, 2, 3, 4, 5]) 
y = np.array([3, 4, 2, 4, 5]) 

# estimating coefficients 
b = coef_estimation(x, y) 
print("Estimated coefficients:\nc = {} \nm = {}".format(b[0], b[1])) 
# plotting regression line 
regression_line_plot(x, y, b)

from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5]) 
y = np.array([3, 4, 2, 4, 5]) 

n = np.arange(5)
# Calculating R2
y_pred = []
for i in n:
    y_p = b[1]*x[i] + b[0]
    y_pred.append(y_p)

print("R-Square: ",r2_score(y, y_pred))

# observations 
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 7, 8, 9, 10, 12]) 

# estimating coefficients 
b = coef_estimation(x, y) 
print("Estimated coefficients:\nc = {} \nm = {}".format(b[0], b[1])) 
# plotting regression line 
regression_line_plot(x, y, b)

from sklearn.metrics import r2_score

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 7, 8, 9, 10, 12])

n = np.arange(10)
# Calculating R2
y_pred = []
for i in n:
    y_p = b[1]*x[i] + b[0]
    y_pred.append(y_p)

print("R-Square: ",r2_score(y, y_pred))

