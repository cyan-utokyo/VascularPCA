import numpy as np
import vtk

def numpy_to_vtk_polydata(numpy_array):
    # 这个函数暂时没法用
    assert numpy_array.shape[1] == 3, "Expected Nx3 numpy array"

    # 创建点
    points = vtk.vtkPoints()
    for point in numpy_array:
        points.InsertNextPoint(point)

    # 创建polyline
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(numpy_array.shape[0])
    for i in range(numpy_array.shape[0]):
        polyline.GetPointIds().SetId(i, i)

    # 创建cells
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # 创建polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    return polydata

def load_vtk_file(filename, renderer):
    # 读取vtk文件
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)

    # 使用tube filter加粗线条
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(reader.GetOutputPort())
    tube_filter.SetRadius(0.5)  
    tube_filter.SetNumberOfSides(16)
    tube_filter.CappingOn()

    # 映射数据
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube_filter.GetOutputPort())

    # 创建actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 添加到渲染器
    renderer.AddActor(actor)

def create_vtk_viewer(files):
    # 创建渲染器
    renderer = vtk.vtkRenderer()

    # 对每个文件执行加载操作
    for file in files:
        load_vtk_file(file, renderer)

    # 创建VTK渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # 创建VTK渲染窗口交互器
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # 启动渲染循环
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    # 你可以指定多个文件路径
    files = ["./scaling/BG0002_Left.vtk", "./scaling/BG0002_Right.vtk"]
    create_vtk_viewer(files)
