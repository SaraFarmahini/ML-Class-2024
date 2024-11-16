#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


# In[52]:


cl1 = np.random.randn(100, 2)
cl2 = np.random.randn(100, 2)
cl1 += np.array([2, 2])
cl2 += np.array([-2, -2])


# In[53]:


class1 = np.ones([100, 1])
class2 = -1 * np.ones([100,1])


# In[54]:


points = np.vstack((cl1, cl2))
classes = np.vstack((class1, class2))


# In[55]:


indices = np.arange(points.shape[0])


# In[56]:


np.random.shuffle(indices)
points = points[indices]
classes = classes[indices]


# In[57]:


data = np.hstack((points, classes))


# In[61]:


def plot_decision_boundary(points, classes, model):
    model.fit(points, classes.ravel())

    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.scatter(points[:, 0], points[:, 1], c=classes.ravel(), cmap=plt.cm.coolwarm, edgecolors='k', s=40)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Decision Boundary")
    plt.show()


# In[63]:


model = LinearSVC()

plot_decision_boundary(points, classes, model)


# In[64]:


def add_outliers(points, classes, num_outliers=8):
    outlier_indices = np.random.choice(len(classes), num_outliers, replace=False)
    classes[outlier_indices] *= -1 


# In[65]:


add_outliers(points, classes, num_outliers=8)


# In[67]:


plot_decision_boundary(points, classes, model)


# In[ ]:




