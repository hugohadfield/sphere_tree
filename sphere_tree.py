import numpy as np
import pickle
import time

from clifford.g3c import *
from clifford.tools.g3c import *
from pyganja import *

root2 = np.sqrt(2)

@numba.njit
def check_sphere_line_intersect(s,l):
    mv = meet_val(s, l)
    return imt_func(mv,mv)[0]

def sphere_beyond_plane(sphere, plane):
    snorm = unsign_sphere(sphere)
    return (snorm|plane)[0] < -get_radius_from_sphere(snorm)

def flatten(iterable):
    """ 
    Flatten an arbitrarily nested list 
    https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
    """
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


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
    def __init__(self, children=[], parent=None, sphere=None, isdatum=False):
        if len(children) == 0:
            self.children = children
        else:
            self.children = []
            self.add_children(children)
        self.parent = parent
        self._sphere = sphere
        self.isdatum = isdatum

    def apply_rotor(self, rotor):
        self._sphere = apply_rotor(self.sphere, rotor)
        for c in self.children:
            c.apply_rotor(rotor)

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
    def nchildren(self):
        return len(self.children)

    @property
    def isleaf(self):
        return (len(self.children) == 0)

    def add_children(self, streechildren):
        for s in streechildren:
            s.parent = self
            self.children.append(s)
        self._sphere = enclosing_sphere([c.sphere for c in self.children])

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
            print('Converting sphere grid to sphere tree grid')
            st_grid = np.empty((s_per_side,s_per_side,s_per_side),dtype=np.object)
            for i in range(s_per_side):
                for j in range(s_per_side):
                    for k in range(s_per_side):
                        st_grid[i,j,k] = SphereTree(sphere=sphere_grid[i,j,k],isdatum=True)
        else:
            st_grid = sphere_grid
        print('Decimating grid')
        # Start to bin them
        st_grid_2 = np.empty((s_per_side//2,s_per_side//2,s_per_side//2),dtype=np.object)
        for i in range(s_per_side//2):
            for j in range(s_per_side//2):
                for k in range(s_per_side//2):
                    sphere_tree_list = st_grid[2*i:2*i+2,2*j:2*j+2,2*k:2*k+2].flatten()
                    p_sphere = SphereTree(children=sphere_tree_list)
                    st_grid_2[i,j,k] = p_sphere
        return st_grid_2

    @staticmethod
    def from_grid(sphere_grid):
        sgrid = sphere_grid
        while sgrid.shape[0] > 1:
            sgrid = SphereTree.decimate_grid(sgrid)
        return sgrid[0,0,0]

    def drill(self, line):
        nchild = self.nchildren
        i = 0
        if check_sphere_line_intersect(self.sphere.value, line.value) > 0:
            while i < nchild:
                c = self.children[i]
                if c.isdatum:
                    if check_sphere_line_intersect(c.sphere.value, line.value) > 0:
                        del self.children[i]
                        nchild -= 1
                    else:
                        i += 1
                else:
                    c.drill(line)
                    if c.nchildren == 0:
                        del self.children[i]
                        nchild -= 1
                    else:
                        i += 1

    def slice(self, plane):
        if sphere_beyond_plane(self.sphere, plane):
            self.children = []
        else:
            i = 0
            nchild = len(self.children)
            while i < nchild:
                c = self.children[i]
                if not c.isdatum:
                    c.slice(plane)
                    if c.nchildren == 0:
                        del self.children[i]
                        nchild -= 1
                    else:
                        i += 1
                else:
                    if sphere_beyond_plane(c.sphere, plane):
                        del self.children[i]
                        nchild -= 1
                    else:
                        i += 1

    def intersect_with_line(self, line):
        if check_sphere_line_intersect(self.sphere.value, line.value) > 0:
            if self.isleaf:
                return [self]
            else:
                res = [c.intersect_with_line(line) for c in self.children]
                return res
        else:
            return []

    def add_to_scene(self, scene, datum_only=False):
        if self.isleaf:
            if datum_only:
                if self.isdatum:
                    scene.add_object(self.sphere)
            else:
                scene.add_object(self.sphere)
        else:
            for c in self.children:
                c.add_to_scene(scene, datum_only=datum_only)

    def save(self, filename="sphere_tree.pickle"):
        with open(filename,"wb") as pickle_out:
            pickle.dump(self, pickle_out)

    @staticmethod
    def load(filename="sphere_tree.pickle"):
        with open(filename,"rb") as pickle_in:
            st = pickle.load(pickle_in)
        return st


def intersect_sphere_tree_with_line(sphere_tree, line):
    result = list(flatten(sphere_tree.intersect_with_line(line)))
    return result


def construct_sphere_grid(s_per_side, side_length, ndims=3):
    print('Constructing sphere grid')
    sphere_grade = ndims + 1
    sphere_radius = root2*0.5*side_length/(s_per_side-1)
    vspaces = np.linspace(-side_length/2, side_length/2, s_per_side)
    XX,YY,ZZ = np.meshgrid(*[vspaces for i in range(ndims)])
    centers = np.stack([XX,YY,ZZ],axis=-1)
    sphere_grid = np.empty((s_per_side,s_per_side,s_per_side),dtype=np.object)
    I_N = layout.pseudoScalar
    scount = 0
    for i in range(s_per_side):
        for j in range(s_per_side):
            print(100*scount/(s_per_side**3))
            for k in range(s_per_side):
                dual_sphere_val = np.zeros(32)
                dual_sphere_val[1:ndims+1] = centers[i,j,k]
                dual_sphere_val = val_up(dual_sphere_val)
                dual_sphere = layout.MultiVector(value=dual_sphere_val)
                dual_sphere = dual_sphere - 0.5*sphere_radius**2*layout.einf
                sphere = dual_sphere*I_N
                sphere_grid[i,j,k] = sphere
                scount += 1
    return sphere_grid


def time_intersect_sphere_line():
    S = random_sphere()
    L = random_line()
    ntests = 10000
    check_sphere_line_intersect(S.value,L.value)
    start_time = time.time()
    for i in range(ntests):
        check_sphere_line_intersect(S.value,L.value)
    tot_time = time.time() - start_time
    print('microseconds for one sphere line intersect ', 1E6*tot_time/ntests)
    print('# of sphere - line intersections per sec: ', ntests/tot_time)


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


def test_from_grid():
    sphere_grid = construct_sphere_grid(16,4)
    st = SphereTree.from_grid(sphere_grid)
    for i in range(100):
        if not st.isleaf:
            draw([s.sphere for s in st.children], browser_window=True, scale=0.1)
            st = st.children[0]


def test_intersect_with_line():
    line = ((up(e2)^up(-e1 + 2*e2+0.3*e3))^einf).normal()
    sphere_grid = construct_sphere_grid(16,4)
    st = SphereTree.from_grid(sphere_grid)
    print('Intersecting',flush=True)
    result = intersect_sphere_tree_with_line(st, line)
    spheres = [st_node.sphere for st_node in result]
    draw(spheres+[line], browser_window=True, scale=0.1)


def test_drill():
    line = ((up(e2)^up(-e1 + 2*e2+0.3*e3))^einf).normal()
    sphere_grid = construct_sphere_grid(8,4)
    st = SphereTree.from_grid(sphere_grid)
    print('Drilling',flush=True)
    st.drill(line)
    gs = GanjaScene()
    st.add_to_scene(gs)
    gs.add_object(line)
    draw(gs, browser_window=True, scale=0.1)


def test_slice():
    plane = ((up(e3)^up(e2)^up(-e1 + 2*e2+0.3*e3))^einf).normal()
    sphere_grid = construct_sphere_grid(8,4)
    st = SphereTree.from_grid(sphere_grid)
    print('Slicing',flush=True)
    st.slice(plane)
    gs = GanjaScene()
    st.add_to_scene(gs, datum_only=True)
    gs.add_object(plane)
    draw(gs, browser_window=True, scale=0.1)


def test_slice_sphere_grid():
    grid = construct_sphere_grid(8,4)
    plane = ((up(e3)^up(e2)^up(-e1 + 2*e2+0.3*e3))^einf).normal()
    gs = GanjaScene()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                s = grid[i,j,k]
                if sphere_beyond_plane(s, plane):
                    gs.add_object(s)
    gs.add_object(plane)
    draw(gs, browser_window=True, scale=0.1)


def test_save_load():
    sphere_grid = construct_sphere_grid(8,1)
    st = SphereTree.from_grid(sphere_grid)
    st.save()
    loaded_tree = SphereTree.load()
    gs = GanjaScene()
    loaded_tree.add_to_scene(gs)
    draw(gs,browser_window=True,scale=0.1)


def test_drill_timing():
    filename = 'standard_grid16.pickle'
    try:
        st = SphereTree.load(filename)
    except:
        sphere_grid = construct_sphere_grid(16,16)
        st = SphereTree.from_grid(sphere_grid)
        st.save(filename=filename)
    print('Drilling',flush=True)
    line_array = [random_line_at_origin() for i in range(100)]

    start_time = time.time()
    for line in line_array:
        st.drill(line)
    print('Run time: ', time.time() - start_time)

    gs = GanjaScene()
    st.add_to_scene(gs)
    gs.add_objects(line_array, static=True)
    draw(gs, browser_window=True, scale=0.1)


def test_apply_rotor():
    filename = 'standard_grid4.pickle'
    try:
        st = SphereTree.load(filename)
    except:
        sphere_grid = construct_sphere_grid(4,4)
        st = SphereTree.from_grid(sphere_grid)
        st.save(filename=filename)
    gs = GanjaScene()
    st.add_to_scene(gs)
    RT = random_rotation_translation_rotor()
    st.apply_rotor(RT)
    st.add_to_scene(gs)
    draw(gs,browser_window=True,scale=0.1)


if __name__ == '__main__':
    time_intersect_sphere_line()
    #test_join_spheres() 
    #test_enclosing_spheres()
    #test_in_sphere()
    #test_construct_sphere_grid()
    #test_decimate_sphere_grid()
    #test_create_minimal_sphere_tree()
    #test_from_grid()
    #test_intersect_with_line()
    #test_drill()
    #test_slice()
    #test_slice_sphere_grid()
    #test_save_load()
    #test_drill_timing()
    #test_apply_rotor()

