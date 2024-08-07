import os
from shapely.geometry import Polygon, MultiPolygon, Point, LineString

# Function to save original Voronoi cells data structure without the boundary filtering
def save_voronoi_cells_withoutFiltering_to_py(voronoi_cells, filename='voronoi_data.py', data_name='voronoi_lines'):
    """
    Save the structured information of the original Voronoi cells to a .py file for use in Abaqus, without any boundary filtering.
    
    Parameters:
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - filename: The name of the file to save the Voronoi cell information.
    - data_name: The name of the data variable to be saved in the .py file.
    """
    lines_to_save = []

    def process_polygon(polygon):
        exterior_coords = list(polygon.exterior.coords)
        for i in range(len(exterior_coords) - 1):
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            lines_to_save.append(((line.coords[0][0], line.coords[0][1]), (line.coords[1][0], line.coords[1][1])))

    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            process_polygon(cell)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                process_polygon(poly)

    # Ensure the 'voronoiDataFiles' directory exists
    if not os.path.exists('voronoiDataFiles'):
        os.makedirs('voronoiDataFiles')

    # Write to file inside 'voronoiDataFiles' directory
    file_path = os.path.join('voronoiDataFiles', filename)
    with open(file_path, 'w') as file:
        file.write(f"{data_name} = {repr(lines_to_save)}\n")


# Function to save Voronoi cells data structure with the boundary filtering
def save_voronoi_cells_withBoundaryFiltering_to_py(polygon, voronoi_cells, filename='voronoi_data.py', data_name='voronoi_lines'):
    """
    Save the structured information of the Voronoi cells to a .py file for use in Abaqus with the boundary filtering.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon defining the boundary.
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - filename: The name of the file to save the Voronoi cell information.
    - data_name: The name of the data variable to be saved in the .py file.
    """
    lines_to_save = []

    def is_on_boundary(line, polygon):
        """
        Check if a line is on the boundary of the polygon.
        
        Parameters:
        - line: A Shapely LineString object.
        - polygon: A Shapely Polygon or MultiPolygon object.
        
        Returns:
        - Boolean indicating whether the line is on the boundary.
        """
        return (
            polygon.exterior.distance(Point(line.coords[0])) < 1e-9 and
            polygon.exterior.distance(Point(line.coords[1])) < 1e-9
        ) or any(
            interior.distance(Point(line.coords[0])) < 1e-9 and
            interior.distance(Point(line.coords[1])) < 1e-9
            for interior in polygon.interiors
        )

    def process_polygon(cell):
        """
        Process a single polygon to save its lines if they are not on the boundary.
        
        Parameters:
        - cell: A Shapely Polygon object.
        """
        exterior_coords = list(cell.exterior.coords)
        for i in range(len(exterior_coords) - 1):
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            if not is_on_boundary(line, polygon):
                lines_to_save.append(((line.coords[0][0], line.coords[0][1]), (line.coords[1][0], line.coords[1][1])))

    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            process_polygon(cell)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                process_polygon(poly)

    # Ensure the 'voronoiDataFiles' directory exists
    if not os.path.exists('voronoiDataFiles'):
        os.makedirs('voronoiDataFiles')

    # Write to file inside 'voronoiDataFiles' directory
    file_path = os.path.join('voronoiDataFiles', filename)
    with open(file_path, 'w') as file:
        file.write(f"{data_name} = {repr(lines_to_save)}\n")
        
        
# Function to save Polygon boundary data structure     
def save_polygon_boundaries_to_py(polygon, filename='polygon_data.py', data_name='polygon_boundaries'):
    """
    Save the boundary lines of the polygon to a .py file for use in Abaqus.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon defining the boundary.
    - filename: The name of the file to save the boundary information.
    - data_name: The name of the data variable to be saved in the .py file.
    """
    lines_to_save = []
    exterior_lines = []
    interior_lines = []

    def process_boundary(coords, lines_to_save, lines_list):
        """
        Process the boundary lines from coordinates and append to the provided lists.
        
        Parameters:
        - coords: List of coordinates representing the boundary.
        - lines_to_save: List to store all lines to save.
        - lines_list: Specific list to store either exterior or interior lines.
        """
        for i in range(len(coords) - 1):
            line = LineString([coords[i], coords[i + 1]])
            line_tuple = ((line.coords[0][0], line.coords[0][1]), (line.coords[1][0], line.coords[1][1]))
            lines_to_save.append(line_tuple)
            lines_list.append(line_tuple)

    # Extract exterior and interior boundaries
    if isinstance(polygon, Polygon):
        process_boundary(list(polygon.exterior.coords), lines_to_save, exterior_lines)
        for interior in polygon.interiors:
            process_boundary(list(interior.coords), lines_to_save, interior_lines)
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            process_boundary(list(poly.exterior.coords), lines_to_save, exterior_lines)
            for interior in poly.interiors:
                process_boundary(list(interior.coords), lines_to_save, interior_lines)

    # Write to file
    exterior_data_name = f"{data_name}_exterior"
    interior_data_name = f"{data_name}_interior"

    # Ensure the 'voronoiDataFiles' directory exists
    if not os.path.exists('voronoiDataFiles'):
        os.makedirs('voronoiDataFiles')

    # Write to file inside 'voronoiDataFiles' directory
    file_path = os.path.join('voronoiDataFiles', filename)
    with open(file_path, 'w') as file:
        file.write(f"{data_name} = {repr(lines_to_save)}\n")
        file.write(f"{exterior_data_name} = {repr(exterior_lines)}\n")
        file.write(f"{interior_data_name} = {repr(interior_lines)}\n")