import numpy as np
import random
import math
from itertools import combinations
from math import sin, cos, pi, factorial, acos, atan
import matplotlib.pyplot as plt
from itertools import product, combinations
from scipy.optimize import minimize

def main():
    """Creating the figure"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    """Creating the sphere"""
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)

    x, y, z = sph_to_cartes_range(phi, theta)
    """Plotting the sphere"""
    ax.plot_surface(x, y, z, alpha=0.3, color='b')

    """ Initialize random points - The Vertices of the Network """
    nodes = random_nodes()
    print(nodes)
    R = len(nodes[0])
    n = len(nodes)
    init_coords = ([random.uniform(min([i[dim] for i in nodes]), max([i[dim] for i in nodes])) for dim in range(R)])

    """Plotting the initial guess point"""
    Init_D = steiner_distance(init_coords, nodes)
    theta_i, phi_i = init_coords
    x_i_point, y_i_point, z_i_point = sph_to_cartes(theta_i, phi_i)
    ax.scatter(x_i_point, y_i_point, z_i_point, color='blue', label = 'Initial Distance: {:2.2f}'.format(Init_D))

    """ Find the minimum of distance """
    bnds1 = ((0., np.pi), (0, 2*np.pi))
    optim1 = minimize(steiner_distance, init_coords, args=(nodes), method='SLSQP',bounds=bnds1,options={'maxiter':1000})
    steiner_coord1 = optim1.x
    steiner_dist1 = optim1.fun
    theta_s1, phi_s1 = steiner_coord1
    x_s_point1, y_s_point1, z_s_point1 = sph_to_cartes(theta_s1, phi_s1)
    ax.scatter(x_s_point1, y_s_point1, z_s_point1, marker='x',  color='r', label = 'Steiner Distance: {:2.2f}'.format(steiner_dist1))

    # bnds2 = ((0., np.pi), (0, 2*np.pi))
    # optim2 = minimize(steiner_distance, init_coords, args=(nodes), method='Nelder-Mead',bounds=bnds2,options={'maxiter':1000})
    # steiner_coord2 = optim2.x
    # steiner_dist2 = optim2.fun
    # u_s2, v_s2 = steiner_coord2
    # x_s_point2, y_s_point2, z_s_point2 = sph_to_cartes(u_s2, v_s2)
    # ax.scatter(x_s_point2, y_s_point2, z_s_point2, color='green', label = 'Steiner Distance: {:2.2f}'.format(steiner_dist2))
    """Plotting the Steiner point"""

    for theta, phi in nodes:
        x_point, y_point, z_point = sph_to_cartes(theta, phi)

        # Plotting the nodes
        ax.scatter(x_point, y_point, z_point, color='yellow')
        
        # Connect each yellow point with the Initial Point      
        x,y,z = Arc3D((x_point, y_point, z_point), (x_i_point, y_i_point, z_i_point))
        ax.plot(x, y, z, color = 'blue')
        # Connect each yellow point with the Steiner Point      
        x,y,z = Arc3D((x_point, y_point, z_point), (x_s_point1, y_s_point1, z_s_point1))
        ax.plot(x, y, z, color = 'red')

        # x,y,z = Arc3D((x_point, y_point, z_point), (x_s_point2, y_s_point2, z_s_point2))
        # ax.plot(x, y, z, color = 'green')
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
    # 
    # a = (r,phi,theta) where theta==lambda longitude
    Ds = acos(sin(np.pi/2-a[0])*sin(np.pi/2-b[0])+cos(np.pi/2-a[0])*cos(np.pi/2-b[0])*cos(b[1]-a[1]))
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

def sph_to_cartes_range(theta,phi):
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))
    return x, y, z

def sph_to_cartes(theta,phi):
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return x, y, z

if __name__ == "__main__":
    main()
