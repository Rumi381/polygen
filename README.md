# voronoi_meshing: Voronoi Meshing for FEM Integration with Abaqus CAE

This project aims to generalize Voronoi meshing for Finite Element Method (FEM) and integrate the mesh with Abaqus CAE. The provided Python package allows users to generate Voronoi diagrams within an arbitrarily shaped 2D region.

## Features

- Generate Voronoi diagrams within arbitrary 2D boundaries.
- Relax seed points using Lloyd's algorithm.
- Collapse short edges in the Voronoi diagram.
- Optimize boundaries by collapsing short boundary edges.
- Visualize the process and save the resulting data structures.
- Save Voronoi cell and boundary data for further use in Abaqus.

## Installation

To install the package, use `pip`:

```sh
pip install voronoi_meshing
```
## Get Help
To get a detailed information about any function or method, you can use any of the following methods:

### Accessing Documentation

To access the documentation for any function, class, or method in the project, you can use the `help()` function or the `__doc__` attribute. But the project has the **print_docstring(obj)** function which will print the avaiable docstring for any passed object in the function

### Example Usage

1. **Using `help()` Function**:

    ```python
    from voronoi_meshing.geometry import Geometry

    # Display the docstring of the Geometry class
    help(Geometry)

    # Display the docstring of the circle method of the Geometry class
    help(Geometry.circle)
    ```

2. **Using `__doc__` Attribute**:

    ```python
    from voronoi_meshing.geometry import Geometry

    # Print the docstring of the Geometry class
    print(Geometry.__doc__)

    # Print the docstring of the circle method of the Geometry class
    print(Geometry.circle.__doc__)
    ```

3. **Using Utility Function**:
You can edit the **get_help.py** file in the **voronoi_meshing** module and access the docstring of any desired object in the project in an organized way. Following is an example edit of the **get_help.py** file to access the information about the **Geometry.circle** method.

    ```python
    import sys
    import os

    # Add the project root directory to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from voronoi_meshing.utils import print_docstring

    # Add your desired imports here
    from voronoi_meshing.geometry import*
    

    print_docstring(Geometry.circle)
    ```
This will output the following:

    ==================================================
    DOCSTRING
    ==================================================
    Create a circular region.

    --------------------------------------------------
    PARAMETERS
    --------------------------------------------------
            - center: Tuple (x, y) representing the center of the circle.
            - radius: Radius of the circle (optional if point is provided).
            - point: Tuple (x, y) representing a point on the circle (optional if radius is provided).
            - name: Name of the geometry.


    --------------------------------------------------
    RETURNS
    --------------------------------------------------
            - A dictionary with 'polygon' as the key for the Shapely polygon object and 'name' as the key for the name of the geometry.


    --------------------------------------------------
    EXAMPLE USAGE
    --------------------------------------------------
            - circle1 = Geometry.circle((0, 0), radius=5, name='CircleByRadius') # With center and radius

            - circle2 = Geometry.circle((0, 0), point=(3, 4), name='CircleByPoint') # With center and a point on the perimeter


This ensures that you can easily access the documentation and understand how to use the various functions and methods provided in your project.

## Usage
The general instruction is that you should be in the root directory of the project and try to execute any script from the root directory. Then you can run any script from the command line with general rule of script_directory.script. For example: Assuming you in the project root directory, to run the main_ComputeVoronoi.py script from the 'voronoi_meshing' directory, run the following command in the terminal:

```sh
python voro_meshing.main_ComputeVoronoi
```

If you create any script inside any directory any want to run that script either by 'Code Runner' extension in VSCode or from the terminal, please add the following lines at top of your script to avoid any error regarding 'path' issue.

```python

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
```


## Input File (`input.in`)

The `input.in` file should contain the necessary inputs for running the `computeVoronoi` function. The file should follow a simple key-value format, with each parameter on a new line. The parameters can be provided in any order. 

### Parameters

- **boundary**: Path to the OBJ file defining

 the polygonal boundary or a geometric object from the `Geometry` class.
- **N_points**: Number of initial seed points for the Voronoi diagram.
- **points_seed**: Random seed for generating initial Poisson disk sampling points.
- **N_iter**: Number of iterations for Lloyd's algorithm to relax the points.
- **edgeLength_threshold**: Length threshold for collapsing short edges in the Voronoi diagram.
- **boundary_optimize**: Boolean indicating whether to optimize the boundary by collapsing short edges.
- **visualize**: Boolean indicating whether to generate and save visualizations of the process.
- **figureLabels**: Boolean indicating whether to show labels in the generated plots.
- **figureTitle**: Boolean indicating whether to add titles to the generated plots.
- **saveData**: Boolean indicating whether to save the resulting data structures to .py files.

Here is an example of how to organize the `input.in` file:

```ini
# This is a comment
# Boundary is defined by the methods from the Geometry class
boundary = Geometry.trapezoid(base1=8, base2=5, height=4, name='Trapezoid')
# Boundary is defined by the .obj file
# boundary = ./examples/Random2.obj
N_points = 100
points_seed = 42
N_iter = 100000
edgeLength_threshold = 0.04
boundary_optimize = True
visualize = False
figureLabels = False
figureTitle = True
saveData = True
```

### Running the Script with `main.py`

Please provide accurate path for the main.py file and the input.in file. Assuming you are in the same directory as the main.py file, to run the script using the input file, use the following command:

```sh
python voro_meshing.main ./examples/input.in
```

This command will execute the `computeVoronoi` function with the parameters specified in the `input.in` file.

### Example Commands

Here are some example commands to demonstrate different use cases:

1. **Basic Voronoi Diagram Generation**:
    ```sh
    python voro_meshing.main ./examples/input_basic.in
    ```
    `input_basic.in`:
    ```ini
    boundary = ./examples/Random2.obj
    N_points = 100
    visualize = True
    ```

2. **Voronoi Diagram with Edge Collapsing**:
    ```sh
    python voro_meshing.main ./examples/input_edge_collapse.in
    ```
    `input_edge_collapse.in`:
    ```ini
    boundary = ./examples/Random2.obj
    N_points = 100
    edgeLength_threshold = 0.1
    visualize = True
    ```

3. **Optimized Boundary Handling**:
    ```sh
    python voro_meshing.main ./examples/input_optimized_boundary.in
    ```
    `input_optimized_boundary.in`:
    ```ini
    boundary = ./examples/Random2.obj
    N_points = 100
    edgeLength_threshold = 0.1
    boundary_optimize = True
    visualize = True
    ```

4. **Data Saving**:
    ```sh
    python voro_meshing.main ./examples/input_save_data.in
    ```
    `input_save_data.in`:
    ```ini
    boundary = ./examples/Random2.obj
    N_points = 100
    saveData = True
    ```

## Using `main_ComputeVoronoi.py`

The `main_ComputeVoronoi.py` script allows for the flexibility in testing the `Geometry` class along with the .obj file to define the boundary. You can directly put parameters in the code without needing the input file.

### Running the `main_ComputeVoronoi.py` Script

Use can directly run the file using 'Code Runner' extension if you are using VSCode or you can run the following command in the terminal:

```sh
python voro_meshing.main_ComputeVoronoi
```

### Example Usage

To run the `main_ComputeVoronoi.py` script, you can specify the parameters directly within the script without using the input file to fine tune parameters interectively. Here is an example of how to set the parameters directly:

1. **Edit the `main_ComputeVoronoi.py` Script**:

    ```python

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

    N_points = 10

    # Usage: Either by boundary with object from Geometry class or the path of a .OBJ file
    boundary = './examples/Random2.obj'
    # boundary = annular2


    computeVoronoi(boundary, N_points, points_seed=42, N_iter=100000, edgeLength_threshold=0.04, 
                   boundary_optimize=True, visualize=False, figureLabels=True, figureTitle=True, saveData=False)
    
    # Or using the .obj file path
    computeVoronoi(path, N_points, points_seed=42, N_iter=100000, edgeLength_threshold=0.04, 
               boundary_optimize=True, visualize=False, figureLabels=True, figureTitle=True, saveData=False)
    ```

2. **Run the Script**:

    ```sh
    python voro_meshing_main_ComputeVoronoi
    ```

This approach provides the flexibility to test different configurations directly within the script, without needing to modify an external input file.


## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgements

Special thanks to all contributors and the open-source community for their valuable work and support.

---

Feel free to modify and expand upon this draft to better suit your project's needs and details. This structure should give users a clear understanding of the project and how to use the various scripts.

---

This `README.md` provides comprehensive instructions for using both `main.py` and `main_ComputeVoronoi.py`, ensuring that users can easily understand how to configure and run the scripts with the desired parameters and geometric configurations.