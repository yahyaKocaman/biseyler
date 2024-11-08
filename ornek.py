import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import Pipeline  
import matplotlib.pyplot as plt  

X = np.array([[1, 2], [1, 4], [2, 2], [2, 4], [3, 3], [3, 5]])  
y = np.array([5, 11, 9, 17, 13, 21])  

 
poly = PolynomialFeatures(degree=2)  


model = Pipeline([('poly', poly), ('linear', LinearRegression())])  

model.fit(X, y)  


X_plot = np.linspace(0, 4, 100).reshape(-1, 2)  
y_pred = model.predict(X_plot)  
 
plt.figure(figsize=(8, 6))  
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')  
plt.plot(X_plot[:, 0], X_plot[:, 1], 'k-', label='Predicted')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')  
plt.title('Polynomial Linear Regression')  
plt.legend()  
plt.show()
