import bpy
import numpy as np

# Define a function to find a 3D view
def find_3d_view():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {'window': bpy.context.window, 'screen': screen, 'area': area, 'region': region}
                        return override
    return None

# Find a 3D view
view3d = find_3d_view()

# Call the operator with the overridden context
if view3d is not None:
    bpy.ops.view3d.view_selected(view3d)
else:
    print("No 3D view found")



# Assume we have some 3D points stored in a numpy array
points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    # ... etc.
])

# Create a new curve and a new object that uses this curve
curve_data = bpy.data.curves.new(name='my_curve', type='CURVE')
curve_data.dimensions = '3D'
curve_object = bpy.data.objects.new('my_curve', curve_data)

# Link the curve object to the collection
bpy.context.collection.objects.link(curve_object)

# Create a new spline in the curve
spline = curve_data.splines.new(type='POLY')

# Add points to the spline
spline.points.add(len(points) - 1)  # The spline starts with one point by default
for i, point in enumerate(points):
    x, y, z = point
    spline.points[i].co = (x, y, z, 1)  # Points are 4D vectors, with the 4th dimension usually set to 1