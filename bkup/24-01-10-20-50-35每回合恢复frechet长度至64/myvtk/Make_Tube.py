import numpy as np
import vtk
import glob
from scipy import interpolate
from myvtk.GetMakeVtk import measure_length

import numpy as np
import vtk

def write_tube_to_vtk(curve, r, filename, num_segments_300, NumberOfSides=60):
    # Create a vtkPoints object and store the points in it
    points = vtk.vtkPoints()
    for point in curve:
        points.InsertNextPoint(point)
    
    # Create a vtkParametricSpline object and set the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)
    
    # Generate the curve using vtkParametricFunctionSource
    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(num_segments)
    functionSource.Update()
    
    # Create a vtkDoubleArray to store radius information
    radiusData = vtk.vtkDoubleArray()
    radiusData.SetName('TubeRadius')
    for radius in r:
        radiusData.InsertNextValue(radius)

    # Add radius data to the points in polyData
    functionSource.GetOutput().GetPointData().AddArray(radiusData)
    functionSource.GetOutput().GetPointData().SetActiveScalars('TubeRadius')

    # Create tube filter
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(functionSource.GetOutput())
    tubeFilter.SetVaryRadiusToVaryRadiusByScalar()
    tubeFilter.SetNumberOfSides(NumberOfSides)
    tubeFilter.Update()

    # Write the file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(tubeFilter.GetOutput())
    writer.Write()



# Call the function to write the tube to .vtk file
# num_segments = 300
# write_tube_to_vtk(curve2, r2, "tube.vtk", num_segments=num_segments)
