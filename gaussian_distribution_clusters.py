import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# Extract x and y
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.stats as st

n_components = 3
centers = [[0, 1], [0.5,2],[2,2]]
stds=[0.3, 0.3,0.3]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=stds, random_state=0)
#X, truth = make_blobs(n_samples=300, centers=n_components,
                      #cluster_std = [1, 1, 1],
                      #random_state=0)
fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], s=20,alpha=0.5)

plt.xlabel("X coordinates (km)")
plt.ylabel("Y coordinates (km)")
plt.grid(color='gray', alpha=0.5, linestyle='dashed', linewidth=0.5)
x = X[:, 0]
y = X[:, 1]

# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.grid(color='gray', alpha=0.5, linestyle='dashed', linewidth=0.5)
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('X coordinates (km)')
ax.set_ylabel('Y coordinates (km)')
#ax.set_zlabel(r'$f(x|\mu, \Sigma)$ (m)', rotation=180,fontsize=12)
ax.set_zlabel('$f(x|\mu,\Sigma)$ ', rotation=180,fontsize=12)
#ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.2, aspect=5) # add color bar indicating the PDF
ax.view_init(60, 35)
plt.show()
