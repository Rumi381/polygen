import os
import re
import time
import trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from .geometry import Geometry
from .voroGen import generate_poisson_points, lloyds_algorithm_polygon, generate_voronoi_cells
from .optimize import collapse_short_boundaryEdges, collapse_short_voronoiEdges, get_voronoi_stats, print_short_boundaryEdges, print_short_voronoiEdges
from .plotting import plot_boundary_with_points, plot_boundary_with_short_edges, plot_voronoi_cells, plot_voronoi_cells_with_short_edges, plot_voronoi_cells_withBoundaryFiltering
from .savingData import save_polygon_boundaries_to_py, save_voronoi_cells_withBoundaryFiltering_to_py, save_voronoi_cells_withoutFiltering_to_py


# Function to read the input file
def parse_input_file(input_file_path):
    parameters = {
        "boundary": None,
        "N_points": 100,
        "points_seed": 42,
        "N_iter": 100000,
        "edgeLength_threshold": None,
        "boundary_optimize": True,
        "visualize": False,
        "figureLabels": False,
        "figureTitle": False,
        "saveData": True
    }

    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # Skip empty lines or lines that don't contain an '=' or are commented out
            if '=' not in line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove any surrounding quotes

            if key in parameters:
                if value.lower() == 'true':
                    parameters[key] = True
                elif value.lower() == 'false':
                    parameters[key] = False
                elif value.isdigit():
                    parameters[key] = int(value)
                else:
                    try:
                        parameters[key] = float(value)
                    except ValueError:
                        parameters[key] = value

    # Check if boundary is a method of Geometry
    boundary_value = parameters["boundary"]
    if boundary_value is not None:
        try:
            # Evaluate the boundary value if it's a Geometry method
            parameters["boundary"] = eval(boundary_value, {"Geometry": Geometry, "Point": Point})
        except (NameError, SyntaxError):
            # If it's not a Geometry method, treat it as a file path
            parameters["boundary"] = os.path.abspath(boundary_value)

    return parameters

# Functions to read the .obj file or load a Geometry Object and generate the random region as shapely polygon

def load_polygon_from_obj(input_obj):
    """
    Load a polygon either from an .obj file or a predefined geometric object.
    
    Parameters:
    - input_obj: Path to the .obj file or a predefined geometric object.
    
    Returns:
    - Unified polygon.
    """
    if isinstance(input_obj, str) and input_obj.endswith('.obj'):
        # Load the mesh from the OBJ file.
        try:
            mesh = trimesh.load(input_obj, force='mesh')
        except Exception as e:
            raise ValueError(f"Error loading OBJ file: {e}")

        # Assuming the mesh represents a 2D plane or a planar section,
        # we will only consider the x and y coordinates of the vertices.
        vertices_2d = mesh.vertices[:, :2]

        # Create a shapely Polygon for each face in the mesh.
        # Note: This example assumes a simple mesh structure.
        # For complex meshes, you might need to handle multiple polygons.
        try:
            polygons = [Polygon(vertices_2d[face]) for face in mesh.faces]
        except Exception as e:
            raise ValueError(f"Error creating polygons from OBJ mesh faces: {e}")

        # For a single, contiguous shape, unify the polygons.
        unified_polygon = unary_union(polygons)
        
    elif isinstance(input_obj, dict) and 'polygon' in input_obj:
        # Handle predefined geometric objects
        unified_polygon = input_obj['polygon']
        
    else:
        raise ValueError("Input must be a path to a .obj file or a predefined geometric object.")
    
    return unified_polygon


# Function to generate variables' name dynamically based on the .obj file name

def generate_variable_names(input_obj):
    """
    Generate variable names dynamically based on the given geometry object or .obj file.
    
    Parameters:
    - input_obj: Path to the .obj file or a predefined geometric object from the Geometry class.
    
    Returns:
    - A dictionary with dynamically generated variable names based on the input object's name or file name.
    """
    if isinstance(input_obj, str) and input_obj.endswith('.obj'):
        base_name = os.path.splitext(os.path.basename(input_obj))[0]
    elif isinstance(input_obj, dict) and 'name' in input_obj:
        base_name = input_obj['name']
    else:
        raise ValueError("Input must be a path to a .obj file or a predefined geometric object with a 'name' attribute.")

    variable_names = {
        "base_name": base_name,
        "polygon_region": f"polygon_region_{base_name}",
        "seed_points": f"seed_points_{base_name}",
        "relaxed_points": f"relaxed_points_{base_name}",
        "voronoi_cells": f"voronoi_cells_{base_name}",
        "collapsed_voronoi_cells": f"collapsed_voronoi_cells_{base_name}",
        "collapsed_polygonRegion": f"collapsed_polygonRegion_{base_name}",
        "figure_boundary_with_initial_seed_points": f"{base_name}_Polygon_boundary_with_initial_seed_points.png",
        "figure_boundary_with_lloyds_seed_points": f"{base_name}_Polygon_boundary_with_Lloyds_seed_points.png",
        "figure_boundary_with_short_edges": f"{base_name}_Polygon_boundary_with_short_edges.png",
        "figure_voronoi_with_short_edges": f"{base_name}_Voronoi_with_short_edges.png",
        "figure_original_voronoi_cells": f"{base_name}_original_Voronoi_cells.png",
        "figure_boundary_filtered_voronoi_cells": f"{base_name}_boundaryFiltered_Voronoi_cells.png",
        "figure_collapsed_voronoi_cells": f"{base_name}_collapsed_Voronoi_cells.png",
        "figure_collapsed_Boundaries": f"{base_name}_collapsed_Boundaries.png",
        "figure_nonOptimizedBoundary_filtered_collapsedVoronoi_Cells": f"{base_name}_nonOptimizedBoundary_filtered_collapsedVoronoi_Cells.png",
        "figure_OptimizedBoundary_filtered_collapsedVoronoi_Cells": f"{base_name}_nonOptimizedBoundary_filtered_collapsedVoronoi_Cells.png",
        "polygon_data_file": f"polygon_dataFile_{base_name}.py",
        "polygon_data": f"polygon_boundariesData_{base_name}",
        "voronoi_original_data_file": f"originalVoronoi_dataFile_{base_name}.py",
        "voronoi_original_data": f"originalVoronoi_data_{base_name}",
        "voronoi_with_boundary_filtering_data_file": f"voronoi_WithBoundaryFiltering_dataFile_{base_name}.py",
        "voronoi_with_boundary_filtering_data": f"voronoi_WithBoundaryFiltering_data_{base_name}"
    }

    return variable_names


def computeVoronoi2d(boundary, N_points, points_seed=42, N_iter=100000, edgeLength_threshold=None, boundary_optimize=True, visualize=False, figureLabels=False, figureTitle=False, saveData=True):
    """
    Compute and visualize Voronoi diagrams within an arbitrarily shaped 2D region, 
    with options for edge collapsing and boundary optimization. This function 
    integrates various stages of Voronoi diagram generation and provides visualization
    and data-saving capabilities.

    Parameters:
    - boundary: Path to the OBJ file defining the polygonal boundary.
    - N_points: Number of initial seed points for the Voronoi diagram.
    - points_seed: Random seed for generating initial Poisson disk sampling points.
    - N_iter: Number of iterations for Lloyd's algorithm to relax the points.
    - edgeLength_threshold: Length threshold for collapsing short edges in the Voronoi diagram.
    - boundary_optimize: Boolean indicating whether to optimize the boundary by collapsing short edges.
    - visualize: Boolean indicating whether to generate and save visualizations of the process.
    - figureLabels: Boolean indicating whether to show labels in the generated plots.
    - figureTitle: Boolean indicating whether to add titles to the generated plots.
    - saveData: Boolean indicating whether to save the resulting data structures to .py files.

    Use Cases:
    1. **Basic Voronoi Diagram Generation**:
       - Generate and visualize a Voronoi diagram within a given boundary with a specified number of seed points.
       - Example: `computeVoroni('boundary.obj', 100, visualize=True)`
    
    2. **Voronoi Diagram with Edge Collapsing**:
       - Generate a Voronoi diagram and collapse edges shorter than a specified length threshold.
       - Example: `computeVoroni('boundary.obj', 100, edgeLength_threshold=0.1, visualize=True)`
    
    3. **Optimized Boundary Handling**:
       - Generate a Voronoi diagram, collapse short edges, and optimize the boundary by collapsing short boundary edges.
       - Example: `computeVoroni('boundary.obj', 100, edgeLength_threshold=0.1, boundary_optimize=True, visualize=True)`
    
    4. **Data Saving**:
       - Save the resulting Voronoi cell and boundary data to .py files for further use in Abaqus.
       - Example: `computeVoroni('boundary.obj', 100, saveData=True)`

    Workflow:
    1. Load the polygonal boundary from the specified OBJ file.
    2. Generate initial Poisson disk sampling points within the boundary.
    3. Relax the points using Lloyd's algorithm for a specified number of iterations.
    4. Generate Voronoi cells based on the relaxed points.
    5. Optionally collapse short edges in the Voronoi cells and optimize the boundary.
    6. Visualize the results and save the plots, if requested.
    7. Save the Voronoi cell and boundary data to .py files, if requested.

    Example usage:
    -   computeVoroni(
            boundary='boundary.obj',
            N_points=100,
            points_seed=42,
            N_iter=100000,
            edgeLength_threshold=0.1,
            boundary_optimize=True,
            visualize=True,
            figureLabels=True,
            figureTitle=True,
            saveData=True
    )

    This example will generate a Voronoi diagram within the specified boundary, 
    collapse short edges below a threshold, optimize the boundary, visualize the 
    process with labels and titles, and save the resulting data structures to .py files.
    """
    start_time = time.time()  # Start timing

    names = generate_variable_names(boundary)

    # Initialize a dictionary to hold the dynamically named variables
    variables = {}
    
    # Use the generated variable names
    variables[names["polygon_region"]] = load_polygon_from_obj(boundary)
    variables[names["seed_points"]] = generate_poisson_points(variables[names["polygon_region"]], N_points, points_seed)
    variables[names["relaxed_points"]] = lloyds_algorithm_polygon(variables[names["polygon_region"]], variables[names["seed_points"]], N_iter)
    variables[names["voronoi_cells"]] = generate_voronoi_cells(variables[names["polygon_region"]], variables[names["relaxed_points"]])
    
    print('')
    print_short_voronoiEdges(variables[names["voronoi_cells"]], N=10)
    get_voronoi_stats(variables[names["voronoi_cells"]], N=10, file_name='Report_OriginalVoronoi.txt')
    print('')
    print('==========================================================')
    print('')
    print_short_boundaryEdges(variables[names["polygon_region"]], N=10)
    print('==========================================================')
    print('==========================================================')
    print('')
    
    if edgeLength_threshold is not None:
        variables[names["collapsed_voronoi_cells"]] = collapse_short_voronoiEdges(variables[names["voronoi_cells"]], threshold=edgeLength_threshold)
        get_voronoi_stats(variables[names["collapsed_voronoi_cells"]], N=10, file_name='Report_OptimizedVoronoi.txt')
        
        if boundary_optimize:
            variables[names["collapsed_polygonRegion"]] = collapse_short_boundaryEdges(variables[names["polygon_region"]], threshold=edgeLength_threshold)

    voronoiCompute_time = time.time()  # End timing
    elapsed_time = voronoiCompute_time - start_time

    # Print the execution time in an excellent format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Voronoi Computation Time: {int(hours):02} Hours: {int(minutes):02} Minutes: {seconds:05.2f} Seconds")
    print('')
    print('==========================================================')
    print('')

    if visualize:
        # Plotting functions
        plot_boundary_with_points(names["figure_boundary_with_initial_seed_points"], variables[names["polygon_region"]], points=variables[names["seed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle)
        plot_boundary_with_points(names["figure_boundary_with_lloyds_seed_points"], variables[names["polygon_region"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle, title='Boundaries with CVT Seed Points')
        plot_voronoi_cells(names["figure_original_voronoi_cells"], variables[names["voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle)
        plot_voronoi_cells_withBoundaryFiltering(names["figure_boundary_filtered_voronoi_cells"], variables[names["polygon_region"]], variables[names["voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle)
        # plot_voronoi_cells_withBoundaryFiltering(names["figure_boundary_filtered_voronoi_cells"], variables[names["polygon_region"]], variables[names["collapsed_voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1)
        
        if edgeLength_threshold is not None:
            plot_voronoi_cells_with_short_edges(names["figure_boundary_with_short_edges"], variables[names["voronoi_cells"]], threshold=edgeLength_threshold, marker_size=0.1, show_labels=figureLabels, show_title=figureTitle)
            plot_voronoi_cells(names["figure_collapsed_voronoi_cells"], variables[names["collapsed_voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle, title='Voronoi after Collapsing Short Edges')
            plot_voronoi_cells_withBoundaryFiltering(names["figure_nonOptimizedBoundary_filtered_collapsedVoronoi_Cells"], variables[names["polygon_region"]], variables[names["collapsed_voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle, title='Collapsed Voronoi with Non-Optimized Boundary Filtering')
            
            if boundary_optimize:
                plot_boundary_with_short_edges(names["figure_boundary_with_short_edges"], variables[names["polygon_region"]], threshold=edgeLength_threshold, marker_size=0.1, show_labels=figureLabels, show_title=figureTitle)
                plot_boundary_with_points(names["figure_collapsed_Boundaries"], variables[names["collapsed_polygonRegion"]], show_labels=figureLabels, show_title=figureTitle, title='Boundaries after Collapsing Short Edges')
                plot_voronoi_cells_withBoundaryFiltering(names["figure_OptimizedBoundary_filtered_collapsedVoronoi_Cells"], variables[names["collapsed_polygonRegion"]], variables[names["collapsed_voronoi_cells"]], points=variables[names["relaxed_points"]], marker_size=0.1, show_labels=figureLabels, show_title=figureTitle, title='Collapsed Voronoi with Optimized Boundary Filtering')

    if saveData:
        if edgeLength_threshold is not None:
            variables[names["voronoi_cells"]] = variables[names["collapsed_voronoi_cells"]]
            
            if boundary_optimize:
                variables[names["polygon_region"]] = variables[names["collapsed_polygonRegion"]]
        
        # Saving functions
        save_polygon_boundaries_to_py(variables[names["polygon_region"]], filename=names["polygon_data_file"], data_name=names["polygon_data"])
        save_voronoi_cells_withoutFiltering_to_py(variables[names["voronoi_cells"]], filename=names["voronoi_original_data_file"], data_name=names["voronoi_original_data"])
        save_voronoi_cells_withBoundaryFiltering_to_py(variables[names["polygon_region"]], variables[names["voronoi_cells"]], filename=names["voronoi_with_boundary_filtering_data_file"], data_name=names["voronoi_with_boundary_filtering_data"])

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time

    # Print the execution time in an excellent format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Execution Time: {int(hours):02} Hours: {int(minutes):02} Minutes: {seconds:05.2f} Seconds")
    print('')
    print('==========================================================')


# Function to compute 2D Voronoi based on the given input file
def computeVoronoi2d_fromInputFile(input_file_path):
    """
    Compute the 2D Voronoi mesh from parameters specified in an input file.

    This function reads the parameters from an input file and then calls the 
    `computeVoronoi2d` function to generate the Voronoi diagram based on those parameters.

    Parameters:
    - input_file_path (str): The path to the input file containing the parameters.

    The input file should have the following format:
    ```
    boundary = /path/to/boundary.obj
    N_points = 100
    points_seed = 42
    N_iter = 100000
    edgeLength_threshold = 0.1
    boundary_optimize = True
    visualize = False
    figureLabels = False
    figureTitle = False
    saveData = True
    ```

    Example usage:
    ```
    # Assuming the input file is located at './examples/input.in'
    computeVoronoi2d_fromInputFile('./examples/input.in')
    ```

    The function extracts the following parameters from the input file:
    - boundary: The path to the OBJ file defining the boundary.
    - N_points: Number of initial seed points for the Voronoi diagram.
    - points_seed: Random seed for generating initial Poisson disk sampling points.
    - N_iter: Number of iterations for Lloyd's algorithm to relax the points.
    - edgeLength_threshold: Length threshold for collapsing short edges in the Voronoi diagram.
    - boundary_optimize: Boolean indicating whether to optimize the boundary by collapsing short edges.
    - visualize: Boolean indicating whether to generate and save visualizations of the process.
    - figureLabels: Boolean indicating whether to show labels in the generated plots.
    - figureTitle: Boolean indicating whether to add titles to the generated plots.
    - saveData: Boolean indicating whether to save the resulting data structures to .py files.
    """
    parameters = parse_input_file(input_file_path)

    # Extract parameters from the dictionary
    boundary = parameters["boundary"]
    N_points = parameters["N_points"]
    points_seed = parameters["points_seed"]
    N_iter = parameters["N_iter"]
    edgeLength_threshold = parameters["edgeLength_threshold"]
    boundary_optimize = parameters["boundary_optimize"]
    visualize = parameters["visualize"]
    figureLabels = parameters["figureLabels"]
    figureTitle = parameters["figureTitle"]
    saveData = parameters["saveData"]

    if boundary is None:
        print('Please at least provide an accurate path of the OBJ file as the boundary in the input file')
        return

    # Call the computeVoronoi2d function with the extracted parameters
    computeVoronoi2d(
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


# Function to print the docstring of any object in the project. This function will help the user to see a detailed description of the object in a more structured way.
def print_docstring(obj):
    """
    Print the docstring of a given object with enhanced formatting.

    Parameters:
    - obj: The object whose docstring is to be printed.
    """
    docstring = obj.__doc__
    if not docstring:
        print("No docstring available for this object.")
        return
    
    # Split the docstring into lines
    lines = docstring.strip().split('\n')
    
    # Extract the summary, parameters, returns, example usage, and other sections
    summary = []
    params = []
    returns = []
    example_usage = []
    others = []
    current_section = summary

    param_pattern = re.compile(r'\s*- \w+:')

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("Parameters:"):
            current_section = params
            continue
        if stripped_line.startswith("Returns:"):
            current_section = returns
            continue
        if stripped_line.startswith("Example usage:"):
            current_section = example_usage
            continue
        current_section.append(line)
    
    # Define a function to format a section
    def format_section(section, title):
        if section:
            return '\n'.join([
                "-" * 50,
                title,
                "-" * 50,
                '\n'.join(section),
                ""
            ])
        return ""

    # Print the formatted docstring
    formatted_docstring = "\n".join([
        "=" * 50,
        "DOCSTRING",
        "=" * 50,
        '\n'.join(summary),
        format_section(params, "PARAMETERS"),
        format_section(returns, "RETURNS"),
        format_section(example_usage, "EXAMPLE USAGE"),
        format_section(others, "OTHER SECTIONS")
    ])
    
    print(formatted_docstring)