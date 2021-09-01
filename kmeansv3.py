#after paper acceptance 
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import ConvexHull
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from collections import Counter
import math
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from gap_statistic import OptimalK
from numpy import linalg as LA
from numpy.linalg import multi_dot
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

rand_iter = 5
Npts=100
rand_initializations=3

#https://stackoverflow.com/questions/61859098/maximum-volume-inscribed-ellipsoid-in-a-polytope-set-of-points?fbclid=IwAR2DcaQSvqd368lstBayY-atAbQxCVa_EecgS7HE-mrpK6EanO9qbsX2Drg
#From: https://stackoverflow.com/a/61786434/752843

def random_point_ellipse(W,d):
  # random angle
  alpha = 2 * np.pi * np.random.random()
  # vector on that angle
  pt = np.array([np.cos(alpha),np.sin(alpha)])
  # Ellipsoidize it
  return W@pt+d

def GetRandom(dims, Nptss):
  if dims==2:
    W = sklearn.datasets.make_spd_matrix(2)
    d = np.array([2,3])
    #form clusters 2d 
    centers = [[0, 1], [1.5, 1.5], [1,1],[1,2],[2,2],[2.5,2.5],[0,2.5],[1,2.5]]
    stds = [0.6, 0.6,0.6,0.6,0.6,0.6,0.6,0.6]
    points, labels_true = make_blobs(n_samples=Npts, centers=centers, cluster_std=stds, random_state=0)
  else:
    raise Exception("dims must be 2 or 3!")
  return points

def GetHull(points):
  dim  = points.shape[1]
  hull = ConvexHull(points)
  A    = hull.equations[:,0:dim]
  b    = hull.equations[:,dim]
  return A, -b, hull #Negative moves b to the RHS of the inequality

def FindMaximumVolumeInscribedEllipsoid(points,rand_iter):
  """Find the inscribed ellipsoid of maximum volume. Return its matrix-offset form."""
  dim = points.shape[1]
  A,b,hull = GetHull(points)

  B = cp.Variable((dim,dim), PSD=True) #Ellipsoid
  d = cp.Variable(dim)                 #Center

  constraints = [cp.norm(B@A[i],2)+A[i]@d<=b[i] for i in range(len(A))]
  prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
  optval = prob.solve()
  if optval==np.inf:
    raise Exception("No solution possible!")
  #print(f"Optimal value: {optval}") 
  sum=0
  B.value,d.value,avg,inside,outside=Plot(points, hull, B.value, d.value)
  return B.value, d.value,avg,inside,outside

def Plot(points, hull, B, d):
    sum=0
    for j in range(rand_iter):
      if points.shape[1]==2:
        for j in range(rand_iter):
          outside=0
          inside=0
          for i in range(Npts):
            P = (points[i] - d).T         
            Q = (np.linalg.inv(B)).T
            R = np.linalg.inv(B)
            S = (points[i] - d)
            
            if multi_dot([P,Q,R,S])> 1:
                    inside =inside+1
                    np.array(result_t.append(points[i]))
                    np.array(label_t.append(labels_true[i]))
            elif multi_dot([P,Q,R,S])<= 1:
                    outside =outside+1
        kmeans_model = KMeans(n_clusters=3,init='k-means++', n_init=10)
        kmeans = kmeans_model.fit(np.array(result_t))
        centers = np.array(kmeans.cluster_centers_)
        # Centroid coordinates
        centroids = kmeans.cluster_centers_
        display_points = np.array([random_point_ellipse([[1,0],[0,1]],[0,0]) for i in range(Npts)])
        display_points = display_points@B+d
        #optimal clustering
        optimalK = OptimalK(parallel_backend='rust')
        n_clusters = optimalK(points, cluster_array=range(2, 15))
        optimalK.gap_df.head()
        km = KMeans(n_clusters,init='k-means++', n_init=10)
        km.fit(np.array(result_t))
        y_cluster_kmeans = km.predict(np.array(result_t))
        score = metrics.silhouette_score(np.array(result_t), y_cluster_kmeans)
        sum = sum+ score
    avg = sum/rand_iter
    #print("sscore without ellipse: ",avg)
    return B,d,avg,inside,outside

#optimal clustering using points inside only

sum2 =0
centers = [[0, 1], [1.5, 1.5], [1,1],[1,2],[2,2],[2.5,2.5],[0,2.5],[1,2.5]]
stds = [0.6, 0.6,0.6,0.6,0.6,0.6,0.6,0.6]
points, labels_true = make_blobs(n_samples=Npts,centers=centers, cluster_std=stds, random_state=0)
#print("labels_true: ",labels_true)
result_t = []
label_t = []

def WithEllipsoid(Npts,rand_iter,B,d):
# Impose the constraint that v1, ..., v? are all outside of the ellipsoid.
    #take the total sum of no rand iterations
    sum2 =0
    for j in range(rand_iter):
        if points.shape[1]==2:
            for j in range(rand_iter):
              outside=0
              inside=0
              for i in range(Npts):
                P = (points[i] - d).T         
                Q = (np.linalg.inv(B)).T
                R = np.linalg.inv(B)
                S = (points[i] - d)
                
                if multi_dot([P,Q,R,S])> 1:
                        inside =inside+1
                        np.array(result_t.append(points[i]))
                        np.array(label_t.append(labels_true[i]))
                elif multi_dot([P,Q,R,S])<= 1:
                        outside =outside+1

        optimalK = OptimalK(parallel_backend='rust')
        n_clusters2 = optimalK(np.array(result_t), cluster_array=range(2, 15))

        optimalK.gap_df.head()
        km = KMeans(n_clusters2 ,init='k-means++', n_init=10)
        km.fit(np.array(result_t))
        ellipsoid_labels=km.predict(np.array(result_t))
        score2 = metrics.silhouette_score(np.array(result_t), ellipsoid_labels)
        sum2 = sum2+ score2
        avg2 = sum2/rand_iter
    return avg2

rand_init_sum = 0
rand_init_sum_wt=0
in_pts_sum=[]
out_pts_sum=[]

for j in range(rand_initializations):
  np.random.seed(j)
  B,d,sscore_without,in_pts,out_pts = FindMaximumVolumeInscribedEllipsoid(GetRandom(dims=2, Nptss=Npts),rand_iter)
  rand_init_sum_wt =  rand_init_sum_wt + sscore_without
  result_t=[]
  rand_init_avg = WithEllipsoid(Npts,rand_iter,B,d)
  rand_init_sum =  rand_init_sum + rand_init_avg 
  #append inside points to an array
  np.array(in_pts_sum.append(in_pts))
  np.array(out_pts_sum.append(out_pts))

avg_rand_sscore_wt = rand_init_sum_wt/rand_initializations 

avg_rand_sscore = rand_init_sum/rand_initializations 

print("avg_rand_sscore_wt: ",avg_rand_sscore_wt)   

print("avg_rand_sscore: ",avg_rand_sscore,"in pts:",in_pts_sum,"out pts:",out_pts_sum)    