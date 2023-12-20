import sys
from PyQt5 import QtWidgets, QtCore
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkPolyData
# from vtkmodules.vtkFiltersModeling import vtkTubeFilter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # 创建中央窗口部件
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建水平布局
        self.horizontal_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # 存储所有滑块值的列表
        self.slider_values = [5] * 8  # 假设初始值为5

        # 创建左侧滑块
        self.left_panel = QtWidgets.QVBoxLayout()
        self.sliders = []
        for i in range(8):
            slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(5)
            slider.valueChanged.connect(lambda value, idx=i: self.update_sliders_and_plots(idx, value))
            self.sliders.append(slider)
            self.left_panel.addWidget(slider)

        # 创建右侧图表面板
        self.right_panel = QtWidgets.QVBoxLayout()
        self.figures = []
        self.axes = []
        for _ in range(2):
            fig = Figure(figsize=(5, 3))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(self.slider_values, 'o-')  # 使用滑块值初始化图表
            self.figures.append(fig)
            self.axes.append(ax)
            self.right_panel.addWidget(canvas)
        
        
        # 创建VTK渲染窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        # 将面板和VTK窗口添加到布局中
        self.horizontal_layout.addLayout(self.left_panel)
        self.horizontal_layout.addWidget(self.vtk_widget)
        self.horizontal_layout.addLayout(self.right_panel)

        # 配置滑块
        self.configure_sliders()
        
        # Add menu bar with 'Open' action
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('&File')
        open_action = file_menu.addAction('&Open')
        open_action.triggered.connect(self.open_file)

        # Initialize VTK scene without any objects
        self.initialize_vtk_scene()

    def configure_sliders(self):
        # 配置所有滑块
        for slider in self.sliders:
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(5)
            slider.valueChanged.connect(self.on_slider_value_changed)


    def initialize_vtk_scene(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()


    def open_file(self):
        # Open file dialog and select .vtk file
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "VTK Files (*.vtk)",
            options=options
        )
        if file_name:
            self.display_polyline(file_name)

    def on_slider_value_changed(self, value):
        # 滑块值变化时的事件处理函数
        print("Slider value changed:", value)
        # 在这里根据滑块的值更新VTK场景


    def display_polyline(self, file_name):
        # Read the .vtk file containing the polyline
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()

        # Create a mapper and actor for the polyline
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Add the actor to the scene and render
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()  # Adjust the camera to see the object
        self.vtk_widget.GetRenderWindow().Render()

    def update_sliders_and_plots(self, idx, value):
        # 更新滑块值
        self.slider_values[idx] = value

        # 更新图表
        for ax in self.axes:
            ax.clear()
            ax.plot(self.slider_values, 'o-')  # 使用更新后的滑块值绘制图表
            ax.figure.canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())