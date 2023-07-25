import numpy as np
import vtk


def GetMyVtk(filepath, frenet=0):
    reader = vtk.vtkPolyDataReader()
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()

    ######################################################
    #             複数のCellがあるときに使う              #
    #                                                    #
    # for i in range(polydata.GetNumberOfCells()):       #
    #    pts = polydata.GetCell(i).GetPoints()           # 
    #    np_pts = np.array([pts.GetPoint(i) for i in     # 
    #             range(pts.GetNumberOfPoints())])       #
    #                                                    #
    ######################################################

    pts = polydata.GetPoints()    
    np_pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    # np_pts = np_pts - np_pts[0] # distalの始点を(0, 0, 0)まで移動

    Curv = np.array(polydata.GetPointData().GetArray("Curvature"))
    Tors = np.array(polydata.GetPointData().GetArray("Torsion"))
    Radius = np.array(polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))
    Abscissas = np.array(polydata.GetPointData().GetArray("Abscissas"))
    ParallelTransportNormals = np.array(polydata.GetPointData().GetArray("ParallelTransportNormals"))
    if frenet == 1:
        FrenetTangent = np.array(polydata.GetPointData().GetArray("FrenetTangent"))
        FrenetNormal = np.array(polydata.GetPointData().GetArray("FrenetNormal"))
        FrenetBinormal = np.array(polydata.GetPointData().GetArray("FrenetBinormal"))
        return np_pts, Curv, Tors, Radius, Abscissas, ParallelTransportNormals, FrenetTangent, FrenetNormal, FrenetBinormal
    else:
         return np_pts, Curv, Tors, Radius, Abscissas, ParallelTransportNormals
        

def measure_length(v):
    l = 0
    for node_idx in range(1, len(v)):
        dl = np.linalg.norm(v[node_idx][0:3]-v[node_idx-1][0:3])
        l = l+dl
    return l

def measure_length_f_n2n(line, n1, n2):
     v = line[n1:n2]
     return measure_length(v)
     
# coords(df):
# [[x0,y0,z0], [x1,y1,z1],...]

# scalarAttribute(list):
# [['MaximumInscribedSphereRadius', 'float', pandas.Series]]

# fieldAttribute(list):
#  ['Curvature', 'float',pandas.Series],
#  ['Torsion', 'float',pandas.Series]]


def makeVtkFile(savePath, coords, scalarAttributes, fieldAttributes):
    v = open(savePath, "w+")
    v.write("# vtk DataFile Version 2.0\nVessel Segment\nASCII\nDATASET POLYDATA\nPOINTS {} float\n".format(len(coords)))
    for i in range(len(coords)):
        v.write("{} {} {}\n".format(coords[i,0], coords[i,1], coords[i,2]))

    v.write("LINES {} {}\n".format(1, len(coords)+1))
    v.write("{}".format(len(coords)))
    for i in range(len(coords)):
        v.write(" {}".format(i))
    v.write("\n")
    
    #####################################
    #for i in new_LINES:                #
    #    v.write("{} ".format(len(i)))  #
    #    for j in range(len(i)):        #
    #        v.write("{} ".format(i[j]))#
    #    v.write("\n")                  #
    #####################################

    ####################################
    #        scalar Attributes         #
    ####################################

    if len(scalarAttributes) > 0:
        v.write("POINT_DATA {}\n".format(len(coords)))
    else: 
        return

    for i in range(len(scalarAttributes)):
        v.write("SCALARS {} {}\n".format(scalarAttributes[i][0], scalarAttributes[i][1]))
        v.write("LOOKUP_TABLE default\n")
        for j in range(len(coords)):
                v.write("{}\n".format(scalarAttributes[i][2][j]))

    ####################################
    #         field Attributes         #
    ####################################

    if len(fieldAttributes) > 0:
        v.write("FIELD FieldData {}\n".format(len(fieldAttributes)))
    else: 
        return

    for i in range(len(fieldAttributes)):
        v.write("{} 1 {} {}\n".format(fieldAttributes[i][0], len(coords), fieldAttributes[i][1]))
        # v.write("LOOKUP_TABLE default\n")
        for j in range(len(coords)):
                v.write("{}\n".format(fieldAttributes[i][2][j]))

    v.close()
