import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import norm

### Class Helper Functions

'''
Calculate displacement vector along a straight segment of curvelength s
'''
f = lambda s : np.vstack([s, np.zeros_like(s)])

'''
Calculate displacement vector along a curved segment of curvelength s, curvature k,
and tube rotation angle theta (in 2D, theta is 0 or np.pi)
'''
g = lambda theta,k,s : 1/(np.cos(theta)*k) * np.array([np.sin(np.cos(theta)*k*s),1-np.cos(np.cos(theta)*k*s)])

'''
Calculate the 2D rotation matrix of angle phi in counterclockwise direction 
'''
R = lambda phi : np.array([[np.cos(phi), -np.sin(phi)],
                             [np.sin(phi), np.cos(phi)]])

class CTR_DomStiff:
    def __init__(self, Lengths : np.ndarray, Curved_Lengths: np.ndarray, 
                 Kappas: np.ndarray, Pipe_Profile : list = None) -> None:
        ''' Initialize a CTR robot

        Lengths (ndarray) : [L3, L2, L1], total length of each tube, where L3 is the outermost tube
        Curved_Lengths (ndarray): [L3c, L2c, L1c], curved length of each tube
        Kappas (ndarray): [k3, k2, k1], curvature of each tube
        Pipe_Profile (list): Optional, [x, y1, y2] to draw the pipe on all plots
        '''
        # redefine lengths
        Lengths[1] = Lengths[0] + Lengths[1]
        Lengths[2] = Lengths[1] + Lengths[2]
        
        self.Lengths = Lengths
        self.Curved_Lengths = Curved_Lengths
        self.Kappas = Kappas
        self.Pipe_Profile = Pipe_Profile

    def forward_kin(self, kin_lengths : np.ndarray, thetas : np.ndarray, s : float = None):
        ''' Get the (x,y) coordinate of a point at arclength s on the CTR, given the kinematic inputs

        kin_lengths (ndarray): [l3, l2, l1], protrusion length of each tube, where l3 is the outermost tube
        thetas (ndarray) : [theta3, theta2, theta1], rotation angles in radian (0 or np.pi)
        s (float): 0 <= s <= sum(kin_lengths), arclength value. Default: the end of the CTR

        Return:
            (xy, normal_dir) (tuple): coordinate, normal direction of the CTR at arclength s
        '''
        if s is None:
            s = np.sum(kin_lengths)
        [k3, k2, k1] = self.Kappas
        [theta3, theta2, theta1] = thetas
        s_sections = self.get_s_sections(kin_lengths)
        [s3_s, s3_c, s2_s, s2_c, s1_s, s1_c] = self.get_s_sections_intermediate(s_sections,s)
        xy = (f(s3_s).flatten() +g(theta3,k3,s3_c) 
            + R(np.cos(theta3)*k3*s3_c)@(f(s2_s).flatten() + g(theta2,k2,s2_c)
            + R(np.cos(theta2)*k2*s2_c)@(f(s1_s).flatten() + g(theta1,k1,s1_c)))
            ) 
        normal_dir = R(np.cos(theta3)*k3*s3_c)@R(np.cos(theta2)*k2*s2_c)@R(np.cos(theta1)*k1*s1_c)@np.array([1,0])
        return (xy, normal_dir) 
    
    def inverse_kin(self, target_xy : np.ndarray, target_dir : np.ndarray, cost_inv : callable = None) -> list:
        ''' Obtain all possible kinematic inputs from inverse kinematic, given a target location and direction vector

        target_xy (ndarray): (x,y), array of shape (2,), the target xy location for the CTR's tip
        target_dir (ndarray): array of shape (2,), the target normal direction for the CTR's tip
        cost_inv (callable): Optional, a callable for cost calculation with the signature:
            cost_inv(target_xy, target_dir, tip_xy, normal_dir) -> float
        
        Return:
            results (list((kin_lengths, thetas, cost_value)): an ordered list based on the cost value of the kinematic inputs,
            each list element is a tuple of the form (kin_lengths, thetas, cost_value)
        '''
        thetas_options = [[0,0,0], [0,np.pi,0], [0,0,np.pi], [0,np.pi,np.pi], 
                          [np.pi,0,0], [np.pi,np.pi,0], [np.pi,0,np.pi], [np.pi,np.pi,np.pi]]
        results = []
        stacked_lengths_bounds = [(0,self.Lengths[0]), 
                                  (0,self.Lengths[1]), 
                                  (0,self.Lengths[2])]
        initial_kin_lengths = (1/3)*norm(target_xy)*np.array([1,2,3])
        for thetas in thetas_options:
            def objective(stacked_lengths):
                kin_lengths = self.stacked_to_kin_lengths(stacked_lengths)
                tip_xy, normal_dir = self.forward_kin(kin_lengths, thetas)
                if cost_inv is None:
                    return norm(tip_xy-target_xy)**2+ norm(normal_dir-target_dir)**2
                else:
                    return cost_inv(target_xy, target_dir, tip_xy, normal_dir)
            opt_result = minimize(objective, initial_kin_lengths, method="Nelder-Mead", 
                                  bounds=stacked_lengths_bounds)
            results.append((self.stacked_to_kin_lengths(opt_result.x), thetas, opt_result.fun))
        results = sorted(results, key=lambda x: x[2])
        return results
    
    def get_shape(self, kin_lengths : np.ndarray, thetas : np.ndarray, merge : bool = True):
        ''' Get the 2D shape of the robot
        
        kin_lengths (ndarray): [l3, l2, l1], protrusion length of each tube, where l3 is the outermost tube
        thetas (ndarray) : [theta3, theta2, theta1], rotation angles in radian (0 or np.pi)
        merge (bool): default True, returns the merged robot shape. 
            If False, returns a list of three lists of xy-coordinates, one for each tube
        '''
        size = 100
        s3 = np.linspace(0, kin_lengths[0], size)
        s2 = kin_lengths[0] + np.linspace(0, kin_lengths[1], size)
        s1 = kin_lengths[0] + kin_lengths[1] + np.linspace(0, kin_lengths[2], size)
        s_vals = [s3, s2, s1]
        [xy_s3, xy_s2, xy_s1] = [np.array([self.forward_kin(kin_lengths, thetas, s)[0] 
                                  for s in s_vals[i]]) for i in range(len(s_vals))]
        if merge:
            return np.concatenate([xy_s3, xy_s2, xy_s1], axis=0)
        else:
            return [xy_s3, xy_s2, xy_s1]
    
    def plot_forward(self, kin_lengths : np.ndarray, thetas : np.ndarray, ax_provided : bool =False, ax : plt.Axes =None):
        ''' Plot the shape of the robot given a set of kinematic inputs

        kin_lengths (ndarray): [l3, l2, l1], protrusion length of each tube, where l3 is the outermost tube
        thetas (ndarray) : [theta3, theta2, theta1], rotation angles in radian (0 or np.pi)
        ax_provided (bool): default False. If True, provide a matplotlib Axis to plot on
        ax (plt.Axes) : default None. Axis to plot, only considered if ax_provided=True
        '''        
        total_length = np.sum(kin_lengths)
        [xy_s3, xy_s2, xy_s1] = self.get_shape(kin_lengths, thetas, merge=False)
        tip_xy, tip_normal_dir = self.forward_kin(kin_lengths, thetas)

        if not ax_provided:
            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(5,5)
        ax.plot(xy_s3[:,0],xy_s3[:,1], "r", linewidth= 5)
        ax.plot(xy_s2[:,0],xy_s2[:,1], "b", linewidth= 3)
        ax.plot(xy_s1[:,0],xy_s1[:,1], "g", linewidth= 1)
        ax.arrow(tip_xy[0], tip_xy[1], 0.5*tip_normal_dir[0], 0.5*tip_normal_dir[1], color="black", ls="--", head_width=.2)
        ax.scatter([tip_xy[0]],[tip_xy[1]], edgecolors="black", facecolors='black',s=20)
        ax.set_xlim(0,20)
        ax.set_ylim(-2,12)
        ax.set_aspect('equal')
        if self.Pipe_Profile is not None:
            self.add_pipe_boundaries(ax)
        return ax
    
    def plot_inverse(self, target_xy : np.ndarray, target_dir : np.ndarray, inverse_results, num_sol_to_plot : int =1):
        ''' Plot inverse kinematic solution

        target_xy (ndarray): (x,y), array of shape (2,), the target xy location for the CTR's tip
        target_dir (ndarray): array of shape (2,), the target normal direction for the CTR's tip
        cost_inv (callable): Optional, a callable for cost calculation with the signature:
            cost_inv(tip_xy, normal_dir, target_xy, target_dir) -> float
        '''
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5,5)
        for i in range(num_sol_to_plot):
            self.plot_forward(inverse_results[i][0], inverse_results[i][1], ax_provided=True, ax=ax)
        ax.arrow(target_xy[0], target_xy[1], 1.5*target_dir[0], 1.5*target_dir[1], 
                 color="magenta", ls="--", head_width=.2)
        ax.scatter([target_xy[0]],[target_xy[1]], edgecolors="magenta", facecolors='none',s=100)
        return ax
    
    def add_pipe_boundaries(self, ax):
        x, y_upper, y_lower = self.Pipe_Profile.tube_shape()
        ax.plot(x, y_upper, c="black")
        ax.plot(x, y_lower, c="black")

    def stacked_to_kin_lengths(self, stacked_lengths):
        l3 = stacked_lengths[0]
        l2 = max([0,stacked_lengths[1]-stacked_lengths[0]])
        l1 = max([0,stacked_lengths[2]-stacked_lengths[1]-stacked_lengths[0]])
        return [l3, l2, l1]
    
    def get_s_sections(self, kin_lengths : np.ndarray) -> np.ndarray:
        ''' Get the piecewise constant curvature sections of the CTR

        kin_lengths (ndarray): [l3, l2, l1], protrusion length of each tube

        Return: 
            s_sections (ndarray): [s3_s,s3_c, s2_s,s2_c, s1_s,s1_c],
            where s(i)_s and s(i)_c is the extrusion length of the straight and curved sections of tube i
        '''
        [L3c, L2c, L1c] = self.Curved_Lengths
        [l3, l2, l1] = kin_lengths

        s3_s = max([0,l3-L3c])
        s3_c = l3 - s3_s

        s2_s = max([0,l2-L2c])
        s2_c = l2 - s2_s

        s1_s = max([0,l1-L1c])
        s1_c = l1 - s1_s

        s_sections = np.array([s3_s,s3_c, s2_s,s2_c, s1_s,s1_c],dtype=float)
        return s_sections

    def get_s_sections_intermediate(self, s_sections : np.ndarray, s : float) -> np.ndarray:
        ''' Get the piecewise constant curvature sections of the CTR, upto arc length s

        s_sections (ndarray): [s3_s,s3_c, s2_s,s2_c, s1_s,s1_c],
            where s(i)_s and s(i)_c is the extrusion length of the straight and curved sections of tube i
        s (float): a point s measured along the CTR arclength 
        '''
        s_sections_inter = np.zeros_like(s_sections)
        for i in range(len(s_sections)):
            s -= s_sections[i]
            if s > 0:
                s_sections_inter[i] = s_sections[i]
            else:
                s_sections_inter[i] = s + s_sections[i]
                break
        return s_sections_inter

    
    
        
        
    



        