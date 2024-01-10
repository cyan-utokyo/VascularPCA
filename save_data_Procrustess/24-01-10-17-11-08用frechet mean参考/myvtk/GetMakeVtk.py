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
def write_vtk_line(file_name, lines, quad_params, single_component_values, single_curvatures, single_torsions, curvatures, torsions):
    with open(file_name, 'w') as vtk_file:
        # Write VTK header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Count the total number of points
        total_points = sum([len(line) for line in lines])
        total_lines = len(lines)

        # Write points
        vtk_file.write(f"POINTS {total_points} float\n")
        for line in lines:
            for point in line:
                vtk_file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # Write lines
        vtk_file.write(f"LINES {total_lines} {total_lines + total_points}\n")
        point_index = 0
        for line in lines:
            vtk_file.write(f"{len(line)} ")
            vtk_file.write(" ".join(str(i) for i in range(point_index, point_index + len(line))))
            vtk_file.write("\n")
            point_index += len(line)

        # Write scalar data for quad_params (as integer)
        vtk_file.write(f"CELL_DATA {total_lines}\n")  # This specifies that the data applies to lines not points
        vtk_file.write("SCALARS quad_param_group int 1\n")  # Each line has one integer value associated with it
        vtk_file.write("LOOKUP_TABLE default\n")
        for param in quad_params:
            vtk_file.write(f"{param}\n")

        # Write scalar data for single_component_values (as float)
        vtk_file.write("SCALARS single_component_feature float 1\n")  # Each line has one float value associated with it
        vtk_file.write("LOOKUP_TABLE default\n")
        for value in single_component_values:
            vtk_file.write(f"{value}\n")
        # Write scalar data for curvatures and torsions as point data
        vtk_file.write(f"POINT_DATA {total_points}\n")

        # Write scalar data for curvatures (as float)
        vtk_file.write("SCALARS single_curvature float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for single_curvature in single_curvatures:
            vtk_file.write(f"{single_curvature}\n")

        # Write scalar data for torsions (as float)
        vtk_file.write("SCALARS single_torsion float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for single_torsion in single_torsions:
            vtk_file.write(f"{single_torsion}\n")

        # Write scalar data for curvatures (as float)
        vtk_file.write("SCALARS curvature float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for curvature in curvatures:
            vtk_file.write(f"{curvature}\n")

        # Write scalar data for torsions (as float)
        vtk_file.write("SCALARS torsion float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for torsion in torsions:
            vtk_file.write(f"{torsion}\n")



def save_vtk_file(vtk_points, vtk_groups, vtk_variance_curvature, vtk_variance_torsion, file_name):
    # 创建 vtkPoints 对象
    points = vtk.vtkPoints()

    # 创建 vtkFloatArray 对象来存储额外数据
    groups_array = vtk.vtkFloatArray()
    groups_array.SetName("Groups")
    
    variance_curvature_array = vtk.vtkFloatArray()
    variance_curvature_array.SetName("Variance_Curvature")

    variance_torsion_array = vtk.vtkFloatArray()
    variance_torsion_array.SetName("Variance_Torsion")

    # 创建 vtkCellArray 对象来存储polylines
    lines = vtk.vtkCellArray()

    point_id = 0
    for i in range(vtk_points.shape[0]):
        # 每条曲线的点数
        num_points = vtk_points.shape[1]

        # 插入新的polyline
        lines.InsertNextCell(num_points)
        for j in range(num_points):
            points.InsertNextPoint(vtk_points[i, j])
            lines.InsertCellPoint(point_id)

            # 添加额外数据
            groups_array.InsertNextValue(vtk_groups[i, j])
            variance_curvature_array.InsertNextValue(vtk_variance_curvature[i, j])
            variance_torsion_array.InsertNextValue(vtk_variance_torsion[i, j])

            point_id += 1

    # 创建 vtkPolyData 对象
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # 将额外的数据数组添加到 polydata
    polydata.GetPointData().AddArray(groups_array)
    polydata.GetPointData().AddArray(variance_curvature_array)
    polydata.GetPointData().AddArray(variance_torsion_array)

    # 创建 writer 对象
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(polydata)

    # 写入文件
    writer.Write()

# 示例使用
# 假设 vtk_points, vtk_groups, vtk_variance_curvature, vtk_variance_torsion 已定义
# file_name = "output.vtk"
# save_vtk_file(vtk_points, vtk_groups, vtk_variance_curvature, vtk_variance_torsion, file_name)


def save_transfer_vtk_file(transfer_vec, transfer_idx, file_name):
    # 创建 vtkPoints 对象
    points = vtk.vtkPoints()

    # 创建 vtkIntArray 对象来存储索引数据
    index_array = vtk.vtkIntArray()
    index_array.SetName("Index")

    # 填充 points 和 index_array
    for i in range(len(transfer_vec)):
        # 插入点
        points.InsertNextPoint(transfer_vec[i])

        # 添加索引数据
        index_array.InsertNextValue(transfer_idx[i])

    # 创建 vtkPolyData 对象
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # 将索引数据数组添加到 polydata
    polydata.GetPointData().AddArray(index_array)

    # 创建 writer 对象
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(polydata)

    # 写入文件
    writer.Write()


