import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

# ===============================================================================
# ===============================================================================
# Fuctions to optimize generated Voronoi cells by collapsing targeted short edges

# Function to analyze the short edges of the generated Voronoi cells
def print_short_voronoiEdges(voronoi_cells, N=None, threshold=None):
    """
    Print the lengths of the edges in the Voronoi cells.

    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.
    - N: Optional, the number of shortest edges to print. If None, print all edges.
    - threshold: Optional, the length threshold to print edges shorter than this value. If None, print edges based on N.

    Prints the lengths of the edges in the format:
    Index: <index>, Length: <length>
    
    - Example usage:
        - print_short_voronoiEdges(voronoi_cells, N=10)
        - print_short_voronoiEdges(voronoi_cells, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    # Collect all edges with their lengths
    edges = []
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                edges.append((line.length, line))
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    edges.append((line.length, line))

    # Sort edges by length
    edges.sort(key=lambda x: x[0])

    # Determine the edges to print based on N or threshold
    if N is not None:
        edges_to_print = edges[:N]
    else:
        edges_to_print = [edge for edge in edges if edge[0] < threshold]

    if not edges_to_print:
        # If there are no edges to print, print a message
        print(f'There are no edges shorter than the threshold of {threshold}.')
        return

    # Print the edges
    print('10 shortest edge lengths in the original Vononoi')
    for idx, (length, _) in enumerate(edges_to_print):
        print(f"Index: {idx}, Length: {length}")


# This is the mesh optimization function to collapse the short edges of the generated Voronoi cells.
def collapse_short_voronoiEdges(voronoi_cells, N=None, threshold=None):
    """
    Collapse the N shortest edges of the Voronoi cells by merging their vertices or collapse edges 
    shorter than a specified threshold.
    
    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.
    - N: Number of shortest edges to collapse.
    - threshold: Length threshold to collapse edges shorter than this value.
    
    Returns:
    - List of modified Shapely Polygon objects representing the Voronoi cells with collapsed edges.
    
    - Example usage:
        - modified_voronoiCells = collapse_short_voronoiEdges(voronoi_cells, N=10)
        - modified_voronoiCells = collapse_short_voronoiEdges(voronoi_cells, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    # Collect all edges with their lengths and corresponding cell and index information
    shortest_edges = []
    for cell_index, cell in enumerate(voronoi_cells):
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                shortest_edges.append((line.length, line, cell_index, i))
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    shortest_edges.append((line.length, line, cell_index, i))

    # Sort edges by length
    shortest_edges.sort(key=lambda x: x[0])

    # Select the edges to collapse based on N or threshold
    if N is not None:
        edges_to_collapse = shortest_edges[:N]
    else:
        edges_to_collapse = [edge for edge in shortest_edges if edge[0] < threshold]
        
        
    if not edges_to_collapse:
        # If there are no edges to collapse, return the original voronoi_cells
        print('There are no edges to collapse, returned the original voronoi_cells')
        return voronoi_cells

    
    # Step 1: Create a merge list for vertices
    merge_list = {}
    for _, line, _, _ in edges_to_collapse:
        start, end = line.coords[0], line.coords[1]
        merge_list[end] = start  # Merge end vertex into start vertex

    # Step 2: Create a point map for merging vertices and check for cycles
    point_map = {}
    def find_root(vertex):
        # Find the root of the vertex
        root = vertex
        while root in point_map:
            root = point_map[root]
        # Path compression
        while vertex != root:
            parent = point_map[vertex]
            point_map[vertex] = root
            vertex = parent
        return root

    for old_point, new_point in merge_list.items():
        root_old = find_root(old_point)
        root_new = find_root(new_point)
        if root_old != root_new:
#             point_map[root_new] = root_old  # This is original implementation
            point_map[root_old] = root_new  # We can test which one works better

    # Step 3: Update vertices in each Voronoi cell using the point map
    def update_vertex(vertex):
        return find_root(vertex)

    modified_cells = []
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            new_coords = [update_vertex(coord) for coord in cell.exterior.coords[:-1]]
            modified_cells.append(Polygon(new_coords))
        elif isinstance(cell, MultiPolygon):
            new_polys = []
            for poly in cell.geoms:
                new_coords = [update_vertex(coord) for coord in poly.exterior.coords[:-1]]
                new_polys.append(Polygon(new_coords))
            modified_cells.append(MultiPolygon(new_polys))

    return modified_cells


# ====================================================================================
# ====================================================================================
# Fuctions to optimize generated Polygon boundaries by collapsing targeted short edges

# Function to print the short edges of the polygon boundaries
def print_short_boundaryEdges(polygon, N=None, threshold=None):
    """
    Print the N shortest edges of the polygon or edges shorter than a specified threshold.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon whose edges are to be printed.
    - N: Number of shortest edges to print.
    - threshold: Length threshold to print edges shorter than this value.
    
    - Example usage:
        - print_short_boundaryEdges(polygon, N=10)
        - print_short_boundaryEdges(polygon, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    def collect_edges_from_polygon(polygon):
        edges = []
        exterior_coords = list(polygon.exterior.coords)
        for i in range(len(exterior_coords) - 1):
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            edges.append((line.length, line))
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            for i in range(len(interior_coords) - 1):
                line = LineString([interior_coords[i], interior_coords[i + 1]])
                edges.append((line.length, line))
        return edges

    # Collect all edges with their lengths
    edges = collect_edges_from_polygon(polygon)
    if isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            edges.extend(collect_edges_from_polygon(poly))
    
    # Sort edges by length
    edges.sort(key=lambda x: x[0])

    # Select the edges to print based on N or threshold
    if N is not None:
        edges_to_print = edges[:N]
    else:
        edges_to_print = [edge for edge in edges if edge[0] < threshold]

    if not edges_to_print:
        print("No edges to print.")
        return

    # Print the edges
    print('10 shortest edge lengths in the original Boundaries')
    for index, (length, line) in enumerate(edges_to_print):
        print(f"Index: {index}, Length: {length}")


# Function to collapse short edges of the polygon boundaries
def collapse_short_boundaryEdges(polygon, N=None, threshold=None):
    """
    Collapse the N shortest edges of the polygon by merging their vertices or collapse edges 
    shorter than a specified threshold.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon defining the boundary.
    - N: Number of shortest edges to collapse.
    - threshold: Length threshold to collapse edges shorter than this value.
    
    Returns:
    - Modified Shapely Polygon or MultiPolygon with collapsed edges.
    
    - Example usage:
        - modified_polygon = collapse_short_boundaryEdges(polygon, N=10)
        - modified_polygon = collapse_short_boundaryEdges(polygon, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    def collect_edges_from_polygon(polygon):
        edges = []
        exterior_coords = list(polygon.exterior.coords)
        for i in range(len(exterior_coords) - 1):
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            edges.append((line.length, line, 'exterior', i))
        for interior_index, interior in enumerate(polygon.interiors):
            interior_coords = list(interior.coords)
            for i in range(len(interior_coords) - 1):
                line = LineString([interior_coords[i], interior_coords[i + 1]])
                edges.append((line.length, line, ('interior', interior_index), i))
        return edges

    # Collect all edges with their lengths and corresponding cell and index information
    shortest_edges = collect_edges_from_polygon(polygon)
    if isinstance(polygon, MultiPolygon):
        for poly_index, poly in enumerate(polygon.geoms):
            shortest_edges.extend(collect_edges_from_polygon(poly))
    
    # Sort edges by length
    shortest_edges.sort(key=lambda x: x[0])

    # Select the edges to collapse based on N or threshold
    if N is not None:
        edges_to_collapse = shortest_edges[:N]
    else:
        edges_to_collapse = [edge for edge in shortest_edges if edge[0] < threshold]
    
    if not edges_to_collapse:
        # If there are no edges to collapse, return the original polygon
        print('There are no edges to collapse, returned the original polygon')
        return polygon

    # Step 1: Create a merge list for vertices
    merge_list = {}
    for _, line, _, _ in edges_to_collapse:
        start, end = line.coords[0], line.coords[1]
        merge_list[end] = start  # Merge end vertex into start vertex

    # Step 2: Create a point map for merging vertices and check for cycles
    point_map = {}
    def find_root(vertex):
        # Find the root of the vertex
        root = vertex
        while root in point_map:
            root = point_map[root]
        # Path compression
        while vertex != root:
            parent = point_map[vertex]
            point_map[vertex] = root
            vertex = parent
        return root

    for old_point, new_point in merge_list.items():
        root_old = find_root(old_point)
        root_new = find_root(new_point)
        if root_old != root_new:
            point_map[root_old] = root_new

    # Step 3: Update vertices in the polygon using the point map
    def update_vertex(vertex):
        return find_root(vertex)
    
    def update_polygon(polygon):
        new_exterior_coords = [update_vertex(coord) for coord in polygon.exterior.coords[:-1]]
        new_interiors = []
        for interior in polygon.interiors:
            new_interior_coords = [update_vertex(coord) for coord in interior.coords[:-1]]
            new_interiors.append(new_interior_coords)
        return Polygon(new_exterior_coords, new_interiors)

    if isinstance(polygon, Polygon):
        modified_polygon = update_polygon(polygon)
    elif isinstance(polygon, MultiPolygon):
        modified_polys = [update_polygon(poly) for poly in polygon.geoms]
        modified_polygon = unary_union(modified_polys)

    return modified_polygon


# Function to analyze the statistics of the Voronoi Cells
def analyze_voronoi_cells(voronoi_cells):
    """
    Analyze the Voronoi cells to calculate useful statistics such as average grain size,
    percentage of different grain sizes, and the lengths of the shortest edges.

    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.

    Returns:
    - A dictionary containing various statistics about the Voronoi cells.
    """
    edge_lengths = []
    cell_areas = []

    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            cell_areas.append(cell.area)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                edge_lengths.append(line.length)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                cell_areas.append(poly.area)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    edge_lengths.append(line.length)

    edge_lengths = np.array(edge_lengths)
    cell_areas = np.array(cell_areas)

    stats = {
        "average_edge_length": np.mean(edge_lengths),
        "min_edge_length": np.min(edge_lengths),
        "max_edge_length": np.max(edge_lengths),
        "average_cell_area": np.mean(cell_areas),
        "min_cell_area": np.min(cell_areas),
        "max_cell_area": np.max(cell_areas),
        "total_cells": len(cell_areas),
        "total_edges": len(edge_lengths)
    }

    grain_size_bins = np.histogram(cell_areas, bins='auto')
    stats["grain_size_distribution"] = grain_size_bins

    return stats

# Function to create the report of the statistics
def get_voronoi_stats(voronoi_cells, N=None, threshold=None, file_name='Report_OriginalVoronoi.txt'):
    """
    Print useful statistics about the Voronoi cells, including the lengths of the shortest edges.
    Save the print statements in the Report.txt file inside the 'Report' directory.

    Parameters:
    - voronoi_cells: List of Shapely Polygon or MultiPolygon objects representing the Voronoi cells.
    - N: Optional, the number of shortest edges to print. If None, print all edges.
    - threshold: Optional, the length threshold to print edges shorter than this value. If None, print edges based on N.

    Prints the lengths of the edges in the format:
    Index: <index>, Length: <length>
    
    Example usage:
        - print_voronoi_statistics(voronoi_cells, N=10)
        - print_voronoi_statistics(voronoi_cells, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    stats = analyze_voronoi_cells(voronoi_cells)

    # Ensure the 'Report' directory exists
    if not os.path.exists('Reports'):
        os.makedirs('Reports')

    # Write to file inside 'Report' directory
    file_path = os.path.join('Reports', file_name)
    
    with open(file_path, 'w') as file:
        # Print general statistics
        file.write("Voronoi Cell Statistics:\n")
        file.write(f"Average Edge Length: {stats['average_edge_length']:.5f}\n")
        file.write(f"Minimum Edge Length: {stats['min_edge_length']:.5f}\n")
        file.write(f"Maximum Edge Length: {stats['max_edge_length']:.5f}\n")
        file.write(f"Average Cell Area: {stats['average_cell_area']:.5f}\n")
        file.write(f"Minimum Cell Area: {stats['min_cell_area']:.5f}\n")
        file.write(f"Maximum Cell Area: {stats['max_cell_area']:.5f}\n")
        file.write(f"Total Number of Cells: {stats['total_cells']}\n")
        file.write(f"Total Number of Edges: {stats['total_edges']}\n")

        # Print grain size distribution
        file.write("\nGrain Size Distribution:\n")
        for i in range(len(stats["grain_size_distribution"][0])):
            file.write(f"Grain size range {stats['grain_size_distribution'][1][i]:.5f} to {stats['grain_size_distribution'][1][i + 1]:.5f}: {stats['grain_size_distribution'][0][i]} cells\n")

        # Collect all edges with their lengths for the shortest edges analysis
        edges = []
        for cell in voronoi_cells:
            if isinstance(cell, Polygon):
                exterior_coords = list(cell.exterior.coords)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    edges.append((line.length, line))
            elif isinstance(cell, MultiPolygon):
                for poly in cell.geoms:
                    exterior_coords = list(poly.exterior.coords)
                    for i in range(len(exterior_coords) - 1):
                        line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                        edges.append((line.length, line))

        # Sort edges by length
        edges.sort(key=lambda x: x[0])

        # Determine the edges to print based on N or threshold
        if N is not None:
            edges_to_print = edges[:N]
        else:
            edges_to_print = [edge for edge in edges if edge[0] < threshold]

        if not edges_to_print:
            # If there are no edges to print, print a message
            file.write(f'\nThere are no edges shorter than the threshold of {threshold}.\n')
            return

        # Print the shortest edges
        file.write('\n10 Shortest Edge Lengths:\n')
        for idx, (length, _) in enumerate(edges_to_print):
            file.write(f"Index: {idx}, Length: {length:.5f}\n")