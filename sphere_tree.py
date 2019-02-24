import numpy as np
import os
os.environ['NUMBA_DISABLE_PARALLEL']='1'

from clifford.g3c import *
from clifford.tools.g3c import *
from pyganja import *

root2 = np.sqrt(2)


def join_spheres(S1,S2):
    """ 
    Find the smallest sphere that encloses both spheres
    """
    s1 = -(S1/((I5*S1)|einf)[0]).normal()
    s2 = -(S2/((I5*S2)|einf)[0]).normal()
    L = ((s1*I5)^(s2*I5)^einf)(3).normal()
    pp1 = meet(s1,L)(2).normal()
    pp2 = meet(s2,L)(2).normal()
    p1 = point_pair_to_end_points(pp1)[0]
    p2 = point_pair_to_end_points(pp2)[1]
    P = up(e1)^up(e2)^up(-e2)^einf
    if (p1|(s2*I5))[0] > 0.0:
        opt_sphere = s2(4)
    elif (p2|(s1*I5))[0] > 0.0:
        opt_sphere = s1(4)
    else:
        opt_sphere = ((p1^p2)*(p1^p2^einf).normal()*I5)(4)
    return opt_sphere


def enclosing_sphere(spheres):
    nspheres = len(spheres)
    if nspheres == 1:
        return spheres[0]
    elif nspheres == 2:
        return join_spheres(spheres[0],spheres[1])
    mins = spheres[0]
    for i in range(1,nspheres):
        mins = join_spheres(mins,spheres[i])
    return mins


class SphereTree:
    def __init__(self, children=[], parent=None, sphere=None):
        if len(children) == 0:
            self.children = children
        else:
            self.children = []
            self.add_children(children)
        self.parent = parent
        self._sphere = sphere

    @property
    def sphere(self):
        if self._sphere is None:
            if len(self.children) > 0:
                self._sphere = enclosing_sphere([c.sphere for c in self.children])
                return self._sphere
            else:
                return None
        else:
            return self._sphere

    @property
    def isroot(self):
        return (self.parent is None)
    
    @property
    def isleaf(self):
        return (len(self.children) == 0)

    def add_children(self, streechildren):
        for s in streechildren:
            s.parent = self
            self.children.append(s)
        self._sphere = enclosing_sphere([c.sphere for c in self.children])
        print(self._sphere)

    @staticmethod
    def from_list(sphere_list):
        p_sphere = SphereTree()
        sphere_tree_list = [SphereTree(sphere=s) for s in sphere_list]
        p_sphere.add_children(sphere_tree_list)
        return p_sphere

    @staticmethod
    def decimate_grid(sphere_grid):
        # Go bottom up
        s_per_side = sphere_grid.shape[0]
        # Sphere tree grid
        if not isinstance(sphere_grid[0,0,0], SphereTree):
            print('Constructing sphere tree grid')
            st_grid = np.empty((s_per_side,s_per_side,s_per_side),dtype=np.object)
            for i in range(s_per_side):
                for j in range(s_per_side):
                    for k in range(s_per_side):
                        st_grid[i,j,k] = SphereTree(sphere=sphere_grid[i,j,k])
        else:
            st_grid = sphere_grid
        # Start to bin them
        st_grid_2 = np.empty((s_per_side//2,s_per_side//2,s_per_side//2),dtype=np.object)
        for i in range(s_per_side//2):
            for j in range(s_per_side//2):
                for k in range(s_per_side//2):
                    sphere_tree_list = st_grid[2*i:2*i+2,2*j:2*j+2,2*k:2*k+2].flatten()
                    p_sphere = SphereTree(children=sphere_tree_list)
                    st_grid_2[i,j,k] = p_sphere
        return st_grid_2

def construct_sphere_grid(s_per_side, side_length, ndims=3):
    sphere_grade = ndims + 1
    sphere_radius = root2*0.5*side_length/(s_per_side-1)
    vspaces = np.linspace(-side_length/2, side_length/2, s_per_side)
    XX,YY,ZZ = np.meshgrid(*[vspaces for i in range(ndims)])
    centers = np.stack([XX,YY,ZZ],axis=-1)
    sphere_grid = np.empty((s_per_side,s_per_side,s_per_side),dtype=np.object)
    for i in range(s_per_side):
        for j in range(s_per_side):
            for k in range(s_per_side):
                dual_sphere_val = np.zeros(32)
                dual_sphere_val[1:ndims+1] = centers[i,j,k]
                dual_sphere = layout.MultiVector(value=dual_sphere_val)
                dual_sphere = up(dual_sphere)
                dual_sphere = dual_sphere - 0.5*sphere_radius**2*layout.einf
                sphere = dual_sphere.dual()
                sphere_grid[i,j,k] = sphere
    return sphere_grid


def test_join_spheres():
    s1 = (up(5*e1 + 2*e1)^up(5*e1 - 2*e1)^up(5*e1 + 2*e2)^up(5*e1 + 2*e3)).normal()
    s2 = (up(-5*e1 + e1)^up(-5*e1 - e1)^up(-5*e1 + e2)^up(-5*e1 + e3)).normal()
    s3 = join_spheres(s1,s2)
    draw([s1,s2,s3], browser_window=True, scale=0.1)


def test_enclosing_spheres():
    s1 = (up(5*e1 + 2*e1)^up(5*e1 - 2*e1)^up(5*e1 + 2*e2)^up(5*e1 + 2*e3)).normal()
    s2 = (up(-5*e1 + e1)^up(-5*e1 - e1)^up(-5*e1 + e2)^up(-5*e1 + e3)).normal()
    s3 = (up(-5*e2 + e1)^up(-5*e2 - e1)^up(-5*e2 + e2)^up(-5*e2 + e3)).normal()
    s4 = enclosing_sphere([s1,s2,s3]).normal()
    P = up(e1)^up(e2)^up(-e2)^einf
    draw([meet(s4,P),meet(s1,P),meet(s2,P),meet(s3,P)], browser_window=True, scale=0.1)


def test_in_sphere():
    C = up(e1)
    rho = 2
    rho2 = rho**2
    ds = C - 0.5*rho2*einf
    for a in np.linspace(-4.0,4.0,50):
        p1 = up(a*e1)
        print(a, p1|ds)


def test_construct_sphere_grid():
    grid = construct_sphere_grid(10,4)
    gs = GanjaScene()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                gs.add_object(grid[i,j,k])
    draw(gs, browser_window=True, scale=0.1)


def test_decimate_sphere_grid():
    grid_high = construct_sphere_grid(16,4)
    grid = SphereTree.decimate_grid(grid_high)
    gs = GanjaScene()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                gs.add_object(grid[i,j,k].sphere)
    draw(gs, browser_window=True, scale=0.1)

    grid = SphereTree.decimate_grid(grid)
    gs = GanjaScene()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                gs.add_object(grid[i,j,k].sphere)
    draw(gs, browser_window=True, scale=0.1)

def test_create_minimal_sphere_tree():
    sphere_list = construct_sphere_grid(2,4).flatten()
    st = SphereTree.from_list(sphere_list)
    draw([i for i in sphere_list], browser_window=True, scale=0.1)
    draw([st.sphere], browser_window=True, scale=0.1)


if __name__ == '__main__':
    #test_join_spheres() 
    #test_enclosing_spheres()
    #test_in_sphere()
    #test_construct_sphere_grid()
    test_decimate_sphere_grid()
    #test_create_minimal_sphere_tree()

