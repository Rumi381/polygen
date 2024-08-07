import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from voronoi_meshing.utils import computeVoronoi
from voronoi_meshing.geometry import*

def main(boundary, N_points, points_seed=42, N_iter=100000, edgeLength_threshold=None, 
               boundary_optimize=True, visualize=True, figureLabels=False, figureTitle=True, saveData=False):
    
    # Check if boundary is a path or a geometric object
    if isinstance(boundary, str):
        boundary = os.path.abspath(boundary)

    computeVoronoi(
        boundary=boundary,
        N_points=N_points,
        points_seed=points_seed,
        N_iter=N_iter,
        edgeLength_threshold=edgeLength_threshold,
        boundary_optimize=boundary_optimize,
        visualize=visualize,
        figureLabels=figureLabels,
        figureTitle=figureTitle,
        saveData=saveData
    )


if __name__ == "__main__":

    # Examples for using standard geometries from the Geometry class

    # Example Use Case for Circle
    circle1 = Geometry.circle((0, 0), radius=5, name='CircleByRadius')
    circle2 = Geometry.circle((0, 0), point=(3, 4), name='CircleByPoint')

    # Example Use Case for Annular Region
    # annular1 = Geometry.annular_region((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, name='AnnularByRadius')
    annular2 = Geometry.annular_region((0, 0), point_outer=(7, 7), center_inner=(0, -1), point_inner=(3, 4), name='AnnularByPoint')

    # Example usage for annular region arc with straight ends
    # annular_arc_straight1 = Geometry.annular_region_arc_WithStraightEnds((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, theta=240, axis='y', name='ArcByRadiusY')
    # annular_arc_straight2 = Geometry.annular_region_arc_WithStraightEnds((0, 0), point_outer=(7, 7), center_inner=(0, 0), point_inner=(3, 4), theta=240, axis='x', name='ArcByPointX')
    annular_arc_straight3 = Geometry.annular_region_arc_WithStraightEnds((0, 0), radius_outer=10, center_inner=(0, -1), radius_inner=5, theta=240, alpha=135, name='ArcByRadiusAlpha')

    # Example usage for annular region arc with rounded ends
    # annular_arc_rounded1 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), radius_outer=10, center_inner=(0, 0), radius_inner=5, theta=240, axis='y', name='RoundedArcByRadiusY')
    # annular_arc_rounded2 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), point_outer=(7, 7), center_inner=(0, 0), point_inner=(3, 4), theta=240, axis='x', name='RoundedArcByPointX')
    annular_arc_rounded3 = Geometry.annular_region_arc_WithRoundedEnds((0, 0), radius_outer=10, center_inner=(0, -1), radius_inner=5, theta=180, alpha=-45, name='RoundedArcByRadiusAlpha')

    # Example Use Case for Ellipse
    ellipse = Geometry.ellipse(center=(0, 0), semi_major_axis=7, semi_minor_axis=4, rotation=30, name='Ellipse')
    # Example Use Case for Annular Ellipse
    annular_ellipse = Geometry.annular_ellipse(center=(0, 0), semi_major_axis_outer=10, semi_minor_axis_outer=6, semi_major_axis_inner=7, semi_minor_axis_inner=4, rotation=30, name='AnnularEllipse')

    # Example Use Case for Rectangle
    rectangle = Geometry.rectangle(width=10, height=5, name='Rectangle')

    # Example Use Case for Rectangle by Corners
    rectangle_corners = Geometry.rectangle_corners(point1=(0, 0), point2=(10, 5), name='RectangleCorners')

    # Example Use Case for Square
    square = Geometry.square(center=(0, 0), side_length=6, name='Square')

    # Example Use Case for Regular Polygon (Pentagon)
    pentagon = Geometry.regular_polygon(center=(0, 0), radius=5, num_sides=5, name='Pentagon')
    # pentagon = Geometry.pentagon(center=(0, 0), radius=5, name='Pentagon')


    # Example Use Case for Regular Polygon (Hexagon)
    hexagon = Geometry.hexagon(center=(0, 0), radius=5, name='Hexagon')

    # Example Use Case for Regular Polygon (Octagon)
    octagon = Geometry.octagon(center=(0, 0), radius=5, name='Octagon')

    # Example Use Case for Star
    star = Geometry.star(center=(0, 0), outer_radius=7, inner_radius=3, num_points=5, name='Star')

    # Example Use Case for Trapezoid
    trapezoid = Geometry.trapezoid(base1=8, base2=5, height=4, name='Trapezoid')

    # Example Use Case for Parallelogram
    parallelogram = Geometry.parallelogram(base=10, side_length=6, height=4, name='Parallelogram')

    # Usage: Either by boundary with obeject from Geometry class or the path of a .OBJ file
    boundary = './examples/Random2.obj'
    # boundary = annular2

    # Define necessary parameters
    N_points = 100
    points_seed = 42
    N_iter = 100000
    edgeLength_threshold = 0.04
    boundary_optimize = True
    visualize = False
    figureLabels = False
    figureTitle = True
    saveData = True

    main(
        boundary=boundary,
        N_points=N_points,
        points_seed=points_seed,
        N_iter=N_iter,
        edgeLength_threshold=edgeLength_threshold,
        boundary_optimize=boundary_optimize,
        visualize=visualize,
        figureLabels=figureLabels,
        figureTitle=figureTitle,
        saveData=saveData
    )


    
