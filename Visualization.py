import vtk as vtk
import pandas as pd

def vtkfile():

    csvfile = './sample.csv'

    colors = vtk.vtkNamedColors()

    points_reader = vtk.vtkDelimitedTextReader()
    points_reader.SetFileName(csvfile)
    points_reader.DetectNumericColumnsOn()
    points_reader.SetFieldDelimiterCharacters(',')
    points_reader.SetHaveHeaders(True)

    table_points = vtk.vtkTableToPolyData()
    table_points.SetInputConnection(points_reader.GetOutputPort())
    table_points.SetXColumn('CentroidX')
    table_points.SetYColumn('CentroidY')
    table_points.SetZColumn('CentroidZ')
    table_points.Update()

    points = table_points.GetOutput()
    points.GetPointData().SetActiveScalars('VelocityMagnitude')
    # range = points.GetPointData().GetScalars().GetRange()

    box = vtk.vtkImageData()
    box.SetDimensions([101,101,101])

    gaussian_kernel = vtk.vtkGaussianKernel()
    gaussian_kernel.SetSharpness(2)
    gaussian_kernel.SetRadius(12)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(box)
    interpolator.SetSourceData(points)
    interpolator.SetKernel(gaussian_kernel)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(interpolator.GetOutputPort())
    # mapper.SetScalarRange(range)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # point_mapper = vtk.vtkPointGaussianMapper()
    # point_mapper.SetInputData(points)
    # # point_mapper.SetScalarRange(range)
    # point_mapper.SetScaleFactor(0.6)
    # point_mapper.EmissiveOff()
    # point_mapper.SetSplatShaderCode(
    #     "//VTK::Color::Impl\n"
    #     "float dist = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);\n"
    #     "if (dist > 1.0) {\n"
    #     "  discard;\n"
    #     "} else {\n"
    #     "  float scale = (1.0 - dist);\n"
    #     "  ambientColor *= scale;\n"
    #     "  diffuseColor *= scale;\n"
    #     "}\n"
    # )
    #
    # point_actor = vtk.vtkActor()
    # point_actor.SetMapper(point_mapper)

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renderer.AddActor(actor)
    # renderer.AddActor(point_actor)
    renderer.SetBackground(colors.GetColor3d("SlateGray"))

    renWin.SetSize(640, 480)
    renWin.SetWindowName('PointInterpolator')

    renderer.ResetCamera()
    renderer.GetActiveCamera().Elevation(-45)

    iren.Initialize()

    renWin.Render()
    iren.Start()




vtkfile()