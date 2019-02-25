# sphere_tree
Builds sphere trees on grids

# What?
This builds the equivalent of an [Octree](https://en.wikipedia.org/wiki/Octree) but made of spheres. 

This is useful for spatial querying and for intersections with lines and other objects etc.

**HIGHLY EXPERIMENTAL** No promises are made about speed or memory efficiency. 
Currently requires the [clifford](https://www.github.com/pygae/clifford) python library to calculate bounding spheres etc.

Most likely this will in the future be merged into the clifford library
