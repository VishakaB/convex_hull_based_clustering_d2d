#k-means
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

rand_iter = 10
Npts=1000
rand_initializations=3
maxk = 10
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
    stds=[0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8]
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
  B.value,d.value,avg=Plot(points, hull, B.value, d.value)
  return B.value, d.value,avg

def Plot(points, hull, B, d):
    sum=0
    sscore = []
    ellipse_points = []
    for j in range(rand_iter):
        #optimal clustering
        optimalK = OptimalK(parallel_backend='rust')
        n_clusters = optimalK(points, cluster_array=range(2, maxk))
        optimalK.gap_df.head()
        km = KMeans(n_clusters,init='k-means++', n_init=10)
        sum2 =0
        sscore2 = []
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
                      np.array(ellipse_points.append(points[i]))
                      np.array(label_t.append(labels_true[i]))
              elif multi_dot([P,Q,R,S])<= 1:
                      outside =outside+1
        km.fit(ellipse_points)
        y_cluster_kmeans = km.predict(ellipse_points)
        score = metrics.silhouette_score(ellipse_points, y_cluster_kmeans)
        np.array(sscore.append(score))
        #print("sscore array : ",sscore)
    avg = np.max(sscore)
    #print("avg: ",avg)
    return B,d,avg

#optimal clustering using points inside only
sum2 =0
centers = [[0, 1], [1.5, 1.5], [1,1],[1,2],[2,2],[2.5,2.5],[0,2.5],[1,2.5]]
stds=[0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8]
points, labels_true = make_blobs(n_samples=Npts,centers=centers, cluster_std=stds, random_state=0)
#print("labels_true: ",labels_true)
result_t = []
label_t = []

def WithEllipsoid(Npts,rand_iter,B,d):
# Impose the constraint that v1, ..., v? are all outside of the ellipsoid.
    #take the total sum of no rand iterations
    sum2 =0
    sscore2 = []
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
        n_clusters2 = optimalK(np.array(result_t), cluster_array=range(2, maxk))
        #print("new cluster count: ",n_clusters2)

        optimalK.gap_df.head()
        km = KMeans(n_clusters2,init='k-means++', n_init=10)
        km.fit(np.array(result_t))
        ellipsoid_labels=km.predict(np.array(result_t))
        score2 = metrics.silhouette_score(np.array(result_t), ellipsoid_labels)
        #print("silhouette_score ellipsoid",score)
        np.array(sscore2.append(score2))
    avg2 = np.max(sscore2)
    return avg2,inside,outside

rand_init_sum = 0
rand_init_sum_wt=0
in_pts_sum=[]
out_pts_sum=[]

for j in range(rand_initializations):
  np.random.seed(j)
  B,d,sscore_without = FindMaximumVolumeInscribedEllipsoid(GetRandom(dims=2, Nptss=Npts),rand_iter)
  rand_init_sum_wt =  rand_init_sum_wt + sscore_without
  
  rand_init_avg,in_pts,out_pts = WithEllipsoid(Npts,rand_iter,B,d)
  rand_init_sum =  rand_init_sum + rand_init_avg 
  #append inside points to an array
  np.array(in_pts_sum.append(in_pts))
  np.array(out_pts_sum.append(out_pts))

avg_rand_sscore_wt = rand_init_sum_wt/rand_initializations 

avg_rand_sscore = rand_init_sum/rand_initializations 

print("avg_rand_sscore_wt: ",avg_rand_sscore_wt)   

print("avg_rand_sscore: ",avg_rand_sscore,"in pts:",in_pts_sum,"out pts:",out_pts_sum)    