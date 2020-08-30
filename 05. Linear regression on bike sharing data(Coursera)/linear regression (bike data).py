
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

import numpy as np
import pandas as pd  
import seaborn as sns 
plt.rcParams['figure.figsize'] = (12, 8)


data= pd.read_csv('bike_sharing_data.txt')
data.head()

data.info()

ax= sns.scatterplot(x="Population",y="Profit",data=data)
ax.set_title("Profit in $100000 vs City Population in $10000")

def cost_function(X,y,theta):
    m= len(y)
    y_pred=X.dot(theta)
    error= (y_pred-y) ** 2
    
    return 1/(2*m) * np.sum(error)

m= data.Population.values.size
X= np.append(np.ones((m,1)), data.Population.values.reshape(m,1), axis=1)
y=data.Profit.values.reshape(m,1)
theta = np.zeros((2,1))

cost_function(X,y,theta)

def gradient_descent(X,y,theta,alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        error = np.dot(X.transpose(), (y_pred-y))
        theta = theta - alpha * 1/m * error
        costs.append(cost_function(X,y,theta))
     
    return theta, costs

theta, costs = gradient_descent(X,y, theta, alpha= 0.01, iterations = 2000)
print("h(x) = {}+ {}x1".format(str(round(theta[0,0],2)),str(round(theta[1,0],2))))


from mpl_toolkits.mplot3d import Axes3D

theta_0 = np.linspace(-10,10,100)
theta_1 = np.linspace(-1,4,100)

cost_values= np.zeros((len(theta_0),len(theta_1)))

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array([theta_0[i],theta_1[j]])
        cost_values[i, j]= cost_function(X,y,t)
        
fig = plt.figure(figsize= (12,8))
ax= fig.gca(projection = '3d')

surf =  ax.plot_surface(theta_0, theta_1, cost_values, cmap='viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30,330)
plt.show()

plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Values of the Cost Function  over Iterations of Gradient Descent");

theta.shape
theta
theta = np.squeeze(theta)
ax= sns.scatterplot(x="Population",y="Profit",data=data)

x_value= [ x for x in range(5,25)]
y_value= [(x * theta[1]+ theta [0]) for x in x_value]

sns.lineplot(x_value,y_value)

plt.xlabel("Population in 10000s")
plt.ylabel("Profit in $10,000")
plt.title("Linear Regression fit")

def predict(x,theta):
    y_pred= np.dot(theta.transpose(),x)
    return y_pred

y_pred_1 = predict(np.array([1,4]), theta) * 1000
print("for a population of 40,000 people, the model predicts a profit of $" +str(round(y_pred_1,0)))

y_pred_2= predict(np.array([1,8.3]),theta) * 10000
print("for a population of 83000 people, the model predicts a profit of $"+ str(round(y_pred_2,0)))
