import numpy as np
import random
import math
from itertools import combinations
import pylab as pl
from matplotlib import collections as mc
from math import sin, cos, pi, factorial, acos, atan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from scipy.optimize import minimize

from mpl_toolkits.mplot3d import axes3d    
from mpl_toolkits.mplot3d import Axes3D

def main():
    """Creating the figure"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    """Creating the sphere"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x, y, z = sph_to_cartes_range(u, v)
    """Plotting the sphere"""
    ax.plot_surface(x, y, z, alpha=0.3, color='b')

    """ Initialize random points - The Vertices of the Network """
    nodes = random_nodes()
    R = len(nodes[0])
    n = len(nodes)
    init_coords = ([random.uniform(min([i[dim] for i in nodes]), max([i[dim] for i in nodes])) for dim in range(R)])

    """Plotting the initial guess point"""
    Init_D = steiner_distance(init_coords, nodes)
    u_i,v_i = init_coords
    x_i_point, y_i_point, z_i_point = sph_to_cartes(u_i,v_i)
    ax.scatter(x_i_point, y_i_point, z_i_point, color='blue', label = 'Initial Distance: {:2.2f}'.format(Init_D))

    """ Find the minimum of distance """
    bnds = ((0., np.pi), (0, 2*np.pi))
    optim = minimize(steiner_distance, init_coords, args=(nodes), method='SLSQP',bounds=bnds,options={'maxiter':300})
    steiner_coord = optim.x
    steiner_dist = optim.fun
    u_s,v_s = steiner_coord
    x_s_point, y_s_point, z_s_point = sph_to_cartes(u_s,v_s)

    """Plotting the Steiner point"""
    ax.scatter(x_s_point, y_s_point, z_s_point, color='r', label = 'Steiner Distance: {:2.2f}'.format(steiner_dist))

    for u,v in nodes:
        x_point, y_point, z_point = sph_to_cartes(u,v)

        # Plotting the nodes
        ax.scatter(x_point, y_point, z_point, color='yellow')
        
        # Connect each yellow point with the Initial Point      
        x,y,z = Arc3D((x_point, y_point, z_point), (x_i_point, y_i_point, z_i_point))
        ax.plot(x, y, z, color = 'blue')
        # Connect each yellow point with the Steiner Point      
        x,y,z = Arc3D((x_point, y_point, z_point), (x_s_point, y_s_point, z_s_point))
        ax.plot(x, y, z, color = 'red')

    # Setting the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Showing the plot
    plt.show()

def random_nodes():
    theta = np.pi*np.random.rand(3)
    phi = 2*np.pi*np.random.rand(3)
    coords = [(theta[i],phi[i]) for i in range(3)]
    return(coords)

def dist_sphere_azim(a, b):
    # a = (r,phi,theta) where theta==lambda longitude
    Ds = acos(sin(a[0])*sin(b[0])+cos(a[0])*cos(b[0])*cos(b[1]-a[1]))
    return (Ds)

def steiner_distance(steiner_coords, nodes):
    theta = steiner_coords[0]
    phi = steiner_coords[1]
    s = [sin(theta)*cos(phi), sin(theta)*cos(phi), cos(theta)]
    Dist = [dist_sphere_azim(coord, (theta,phi)) for coord in nodes]
    D = sum(Dist)
    return(D)

def Arc3D(p0, p1):#
    num_points=100
    alpha = acos(np.dot(p0, p1))
    d = sin(alpha)
    p0 = np.array(p0)
    p1 = np.array(p1)
    res = [(sin(alpha - t)*p0 + sin(t)*p1)/d for t in np.linspace(0,alpha, num_points)]
    x = [res[i][0] for i in range(num_points)]
    y = [res[i][1] for i in range(num_points)]
    z = [res[i][2] for i in range(num_points)]
    return x,y,z

def sph_to_cartes_range(u,v):
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def sph_to_cartes(u,v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

if __name__ == "__main__":
    main()