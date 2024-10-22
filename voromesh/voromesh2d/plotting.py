import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from collections import defaultdict

# Plotting the original Voronoi cells without any boudary filtering
def plot_voronoi_cells(figure_name, voronoi_cells, points=None, marker_size=10, fill_color=False, show_figure=True, show_labels=False, show_title=False, title='Original Voronoi Cells'):
    """
    Plot the original Voronoi cells within a polygon and overlay seed points.
    
    Parameters:
    - figure_name: The name of the file to save the plot.
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - points: Optional 2D numpy array of points (x, y) to overlay on the plot.
    - marker_size: Size of the markers for the points.
    - fill_color: Boolean indicating whether to fill the Voronoi cells with color.
    - show_labels: Boolean indicating whether to show labels in the legend.
    - show_title: Boolean indicating whether to show the title on the plot.
    - title: The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    
    # Plot the Voronoi cells
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            x, y = cell.exterior.xy
            if fill_color:
                ax.fill(x, y, 'skyblue', edgecolor='blue', linestyle='-', linewidth=1)
            else:
                ax.plot(x, y, 'blue', linestyle='-', linewidth=1, label='Voronoi Edges' if show_labels else None)
            ax.scatter(x, y, color='green', marker='o', s=marker_size)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                x, y = poly.exterior.xy
                if fill_color:
                    ax.fill(x, y, 'skyblue', edgecolor='blue', linestyle='-', linewidth=1)
                else:
                    ax.plot(x, y, 'blue', linestyle='-', linewidth=1, label='Voronoi Edges' if show_labels else None)
                ax.scatter(x, y, color='green', marker='o', s=marker_size)

    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', edgecolors='black', s=marker_size*5, label='Seed Points' if show_labels else None)
    
    # Aesthetic improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if show_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')

    if show_title:
        ax.set_title(title, fontsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)

    if show_figure:
        plt.show()
    

# Plotting the Voronoi cells with reconstructed boundaries and internal edges
def plot_voronoi_edges(clipped_cells, figure_name, show_figure=True):
    """
    Plot the whole region, boundary edges, and internal cell edges in separate plots.
    
    Parameters:
    - clipped_cells: List of Shapely Polygon objects representing the clipped Voronoi cells.
    - figure_name: The name of the file to save the figure.
    """
    # Function to extract the boundaries and the inter edges from the Voronoi cells
    def extract_voronoi_edges(clipped_cells):
        """
        Extract and categorize Voronoi edges into boundary edges and internal cell edges.
        
        Boundary edges include all kinds of boundary edges (external and internal).
        Internal cell edges are edges shared by two or more Voronoi cells.
        
        Parameters:
        - clipped_cells: List of Shapely Polygon objects representing the clipped Voronoi cells.
        
        Returns:
        - boundary_edges: List of LineString objects representing all boundary edges (external and internal).
        - internal_cell_edges: List of LineString objects representing the internal Voronoi cell edges.
        
        Example usage:
        - boundary_edges, internal_cell_edges = extract_voronoi_edges(clipped_cells)
        """
        # Dictionary to store edges with their counts
        edge_count = defaultdict(int)

        for cell in clipped_cells:
            if isinstance(cell, Polygon):
                exterior_coords = list(cell.exterior.coords)
                for i in range(len(exterior_coords) - 1):
                    line = tuple(sorted((exterior_coords[i], exterior_coords[i + 1])))
                    edge_count[line] += 1

                for interior in cell.interiors:
                    interior_coords = list(interior.coords)
                    for i in range(len(interior_coords) - 1):
                        line = tuple(sorted((interior_coords[i], interior_coords[i + 1])))
                        edge_count[line] += 1

            elif isinstance(cell, MultiPolygon):
                for poly in cell.geoms:
                    exterior_coords = list(poly.exterior.coords)
                    for i in range(len(exterior_coords) - 1):
                        line = tuple(sorted((exterior_coords[i], exterior_coords[i + 1])))
                        edge_count[line] += 1

                    for interior in poly.interiors:
                        interior_coords = list(interior.coords)
                        for i in range(len(interior_coords) - 1):
                            line = tuple(sorted((interior_coords[i], interior_coords[i + 1])))
                            edge_count[line] += 1

        boundary_edges = []
        internal_cell_edges = []

        for edge, count in edge_count.items():
            line = LineString(edge)
            if count == 1:
                # Boundary edges are those that are shared by only one polygon
                boundary_edges.append(line)
            else:
                # Internal cell edges are those that are shared by two or more polygons
                internal_cell_edges.append(line)

        return boundary_edges, internal_cell_edges
    
    # Extract boundary and internal cell edges
    boundary_edges, internal_cell_edges = extract_voronoi_edges(clipped_cells)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    def plot_cells(ax, cells, color='black', line_width=0.5):
        for cell in cells:
            if isinstance(cell, Polygon):
                exterior = patches.Polygon(list(cell.exterior.coords), closed=True, edgecolor=color, fill=None, linewidth=line_width)
                ax.add_patch(exterior)
                for interior in cell.interiors:
                    hole = patches.Polygon(list(interior.coords), closed=True, edgecolor=color, fill=None, linewidth=line_width)
                    ax.add_patch(hole)
            elif isinstance(cell, MultiPolygon):
                for poly in cell.geoms:
                    exterior = patches.Polygon(list(poly.exterior.coords), closed=True, edgecolor=color, fill=None, linewidth=line_width)
                    ax.add_patch(exterior)
                    for interior in poly.interiors:
                        hole = patches.Polygon(list(interior.coords), closed=True, edgecolor=color, fill=None, linewidth=line_width)
                        ax.add_patch(hole)

    # Plot the whole region
    plot_cells(axs[0], clipped_cells)
    axs[0].set_title('Whole Region with Voronoi Meshing', fontsize=14)
    axs[0].set_aspect('equal')
    axs[0].axis('off')

    # Calculate plot limits for the whole region
    all_coords = [coord for cell in clipped_cells for coord in cell.exterior.coords]
    min_x, min_y = min([coord[0] for coord in all_coords]), min([coord[1] for coord in all_coords])
    max_x, max_y = max([coord[0] for coord in all_coords]), max([coord[1] for coord in all_coords])
    axs[0].set_xlim(min_x, max_x)
    axs[0].set_ylim(min_y, max_y)

    # Plot boundary edges
    for edge in boundary_edges:
        axs[1].plot(*edge.xy, color='red', linewidth=0.5)
    axs[1].set_title('Reconstructed Boundaries', fontsize=14)
    axs[1].set_aspect('equal')
    axs[1].axis('off')

    # Plot internal cell edges
    for edge in internal_cell_edges:
        axs[2].plot(*edge.xy, color='blue', linewidth=0.5)
    axs[2].set_title('Internal Cell Edges', fontsize=14)
    axs[2].set_aspect('equal')
    axs[2].axis('off')

    plt.tight_layout(pad=2.0)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    if show_figure:
        plt.show()

# Plotting the original Voronoi cells with boudary filtering    
def plot_voronoi_cells_withBoundaryFiltering(figure_name, polygon, voronoi_cells, points=None, marker_size=10, fill_color=False, show_figure=True, show_labels=False, show_title=False, title='Original Voronoi Cells with Boundary Filtering'):
    """
    Plot Voronoi cells within a given polygon and overlay seed points and cell vertices,
    excluding the original polygon boundaries from the Voronoi cell boundaries.
    
    Parameters:
    - figure_name: The name of the file to save the plot.
    - polygon: Shapely Polygon or MultiPolygon defining the boundary.
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - points: Optional 2D numpy array of points (x, y) to overlay on the plot.
    - marker_size: Size of the markers for the points.
    - fill_color: Boolean indicating whether to fill the Voronoi cells with color.
    - show_labels: Boolean indicating whether to show labels in the legend.
    - show_title: Boolean indicating whether to show the title on the plot.
    - title: The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    
    # Plot the boundary of the polygon
    def plot_polygon(polygon, color_ex, color_in, linestyle, linewidth, label=None):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color_ex, linestyle=linestyle, linewidth=linewidth, label=label)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color_in, linestyle=linestyle, linewidth=linewidth)
    
    # Add the polygon boundary once
    label_added = False
    if isinstance(polygon, Polygon):
        plot_polygon(polygon, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
        label_added = True
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not label_added:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
                label_added = True
            else:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2)
    
    # Function to check if a line is on the polygon boundary
    def is_on_boundary(line, polygon):
        return all(polygon.exterior.distance(Point(coord)) < 1e-9 for coord in line.coords) or any(
            all(interior.distance(Point(coord)) < 1e-9 for coord in line.coords) for interior in polygon.interiors)

    # Plot the Voronoi cells excluding the original polygon boundaries
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            if fill_color:
                x, y = cell.exterior.xy
                ax.fill(x, y, 'skyblue', edgecolor='blue', linestyle='-', linewidth=1)
            for i in range(len(exterior_coords) - 1):
                line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                if not is_on_boundary(line, polygon):
                    x, y = line.xy
                    ax.plot(x, y, 'blue', linestyle='-', linewidth=1, label='Voronoi Edges' if show_labels and not label_added else None)
                    label_added = True
            ax.scatter(*zip(*exterior_coords), color='green', marker='o', s=marker_size)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                if fill_color:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, 'skyblue', edgecolor='blue', linestyle='-', linewidth=1)
                for i in range(len(exterior_coords) - 1):
                    line = LineString([exterior_coords[i], exterior_coords[i + 1]])
                    if not is_on_boundary(line, polygon):
                        x, y = line.xy
                        ax.plot(x, y, 'blue', linestyle='-', linewidth=1, label='Voronoi Edges' if show_labels and not label_added else None)
                        label_added = True
                ax.scatter(*zip(*exterior_coords), color='green', marker='o', s=marker_size)
    
    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', edgecolors='black', s=marker_size*5, label='Seed Points' if show_labels else None)
    
    # Aesthetic improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if show_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')

    if show_title:
        ax.set_title(title, fontsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    if show_figure:
        plt.show()
   

# Plotting the polygon boundaries from the given polygon region and also the seed points    
def plot_boundary_with_points(figure_name, polygon, points=None, marker_size=1, show_figure=True, show_labels=False, show_title=False, title='Boundaries with Initial Seed Points'):
    """
    Plots the boundary of a Shapely polygon or multipolygon, and overlays seed points.
    
    Parameters:
    - figure_name: The name of the file to save the plot.
    - polygon: Shapely Polygon or MultiPolygon whose boundary is to be plotted.
    - points: A 2D numpy array of points (x, y) to be plotted.
    - marker_size: Size of the markers for the points.
    - show_labels: Boolean indicating whether to show labels in the legend.
    - show_title: Boolean indicating whether to show the title on the plot.
    - title: The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Plot the boundary of the polygon
    def plot_polygon(polygon, color_ex, color_in, linestyle, linewidth, label=None):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color_ex, linestyle=linestyle, linewidth=linewidth, label=label)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color_in, linestyle=linestyle, linewidth=linewidth)
    
    # Add the polygon boundary once
    label_added = False
    if isinstance(polygon, Polygon):
        plot_polygon(polygon, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
        label_added = True
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not label_added:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
                label_added = True
            else:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2)
    
    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', edgecolors='black', s=marker_size*5, label='Seed Points' if show_labels else None)
    
    # Aesthetic improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if show_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

    if show_title:
        ax.set_title(title, fontsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    if show_figure:
        plt.show()
    
    

# Plotting the original Voronoi cells highlighting the targeted shortest edges with either N th or a threshold length
def plot_voronoi_cells_with_short_edges(figure_name, voronoi_cells, N=None, threshold=None, points=None, marker_size=10, show_figure=True, show_labels=False, show_title=False, title='Original Voronoi Cells Highlighting Short Edges'):
    """
    Plot the original Voronoi cells within a polygon and overlay seed points.
    Highlight the shortest edges that are less than or equal to the given 'N' th shortest edge or below a length threshold.
    
    Parameters:
    - figure_name: The name of the file to save the plot.
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - N: The number of shortest edges to highlight.
    - threshold: Length threshold to highlight edges shorter than this value.
    - points: Optional 2D numpy array of points (x, y) to overlay on the plot.
    - marker_size: Size of the markers for the points.
    - show_labels: Boolean indicating whether to show labels in the legend.
    - show_title: Boolean indicating whether to show the title on the plot.
    - title: The title of the plot.
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')

    # Collect all edge lengths
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

    # Select the edges to highlight based on N or threshold
    if N is not None:
        shortest_edges = [edge for _, edge in edges[:N]]
    else:
        shortest_edges = [edge for length, edge in edges if length < threshold]

    # Plot the Voronoi cells
    label_added = False
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            exterior_coords = list(cell.exterior.coords)
            x, y = cell.exterior.xy
            if not label_added and show_labels:
                ax.plot(x, y, 'blue', linestyle='-', linewidth=1, alpha=0.6, label='Voronoi Edges')  # Plot the Voronoi cell boundary
                label_added = True
            else:
                ax.plot(x, y, 'blue', linestyle='-', linewidth=1, alpha=0.6)  # Plot the Voronoi cell boundary
            ax.scatter(*zip(*exterior_coords), color='green', marker='o', s=marker_size)
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                exterior_coords = list(poly.exterior.coords)
                x, y = poly.exterior.xy
                if not label_added and show_labels:
                    ax.plot(x, y, 'blue', linestyle='-', linewidth=1, alpha=0.6, label='Voronoi Edges')  # Plot the Voronoi cell boundary
                    label_added = True
                else:
                    ax.plot(x, y, 'blue', linestyle='-', linewidth=1, alpha=0.6)  # Plot the Voronoi cell boundary
                ax.scatter(*zip(*exterior_coords), color='green', marker='o', s=marker_size)

    # Highlight the shortest edges if there are any
    if shortest_edges:
        label_added = False
        for line in shortest_edges:
            x, y = line.xy
            if not label_added and show_labels:
                ax.plot(x, y, 'orange', linestyle='-', linewidth=2.5, alpha=0.8, label='Shortest Edges')
                label_added = True
            else:
                ax.plot(x, y, 'orange', linestyle='-', linewidth=2.5, alpha=0.8)

    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', edgecolors='black', s=marker_size*5, label='Seed Points' if show_labels else None)
    
    # Aesthetic improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if show_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')

    if show_title:
        ax.set_title(title, fontsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    if show_figure:
        plt.show()
    
# Fuction to plot the Polygon boundaries highlighting the short edges
def plot_boundary_with_short_edges(figure_name, polygon, N=None, threshold=None, points=None, marker_size=1, show_figure=True, show_labels=False, show_title=False, title='Boundaries Highlighting Short Edges'):
    """
    Plots the boundary of a Shapely polygon or multipolygon, and overlays seed points.
    Highlights the shortest edges of the boundaries that are less than or equal to the given 'N' th shortest edge
    or below a length threshold.
    
    Parameters:
    - figure_name: The name of the file to save the plot.
    - polygon: Shapely Polygon or MultiPolygon whose boundary is to be plotted.
    - N: The number of shortest edges to highlight.
    - threshold: Length threshold to highlight edges shorter than this value.
    - points: A 2D numpy array of points (x, y) to be plotted.
    - marker_size: Size of the markers for the points.
    - show_labels: Boolean indicating whether to show labels in the legend.
    - show_title: Boolean indicating whether to show the title on the plot.
    - title: The title of the plot.
    - Example usage:
        - plot_boundary_with_short_edges('boundary_with_short_edges.png', polygon, N=10)
        - plot_boundary_with_short_edges('boundary_with_short_edges.png', polygon, threshold=0.1)
    """
    if (N is None and threshold is None) or (N is not None and threshold is not None):
        raise ValueError("Specify exactly one of 'N' or 'threshold'.")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    
    # Function to collect edges from a polygon
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
    
    # Collect all edge lengths
    edges = []
    if isinstance(polygon, Polygon):
        edges.extend(collect_edges_from_polygon(polygon))
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            edges.extend(collect_edges_from_polygon(poly))

    # Sort edges by length
    edges.sort(key=lambda x: x[0])

    # Select the edges to highlight based on N or threshold
    if N is not None:
        shortest_edges = [edge for _, edge in edges[:N]]
    else:
        shortest_edges = [edge for length, edge in edges if length < threshold]

    # Plot the boundary of the polygon
    def plot_polygon(polygon, color_ex, color_in, linestyle, linewidth, label=None):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color_ex, linestyle=linestyle, linewidth=linewidth, alpha=0.6, label=label)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color_in, linestyle=linestyle, linewidth=linewidth, alpha=0.6)
    
    # Add the polygon boundary once
    label_added = False
    if isinstance(polygon, Polygon):
        plot_polygon(polygon, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
        label_added = True
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not label_added:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary' if show_labels else None)
                label_added = True
            else:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2)
    
    # Highlight the shortest edges if there are any
    if shortest_edges:
        label_added = False
        for line in shortest_edges:
            x, y = line.xy
            if not label_added and show_labels:
                ax.plot(x, y, 'orange', linestyle='-', linewidth=2.5, alpha=0.8, label='Shortest Edges')
                label_added = True
            else:
                ax.plot(x, y, 'orange', linestyle='-', linewidth=2.5, alpha=0.8)

    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='o', edgecolors='black', s=marker_size*5, label='Seed Points' if show_labels else None)
    
    # Aesthetic improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if show_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best')

    if show_title:
        ax.set_title(title, fontsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    if show_figure:
        plt.show()

# Function to plot the energy and norm of energy gradient vs. Lloyd's iterations graph
def plot_energy_and_gradient(energy_list, grad_norm_list, figure_name="Energy_and _Norm_of_Energy_Gradient.png"):
    iterations = np.arange(1, len(energy_list) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Energy on the left y-axis
    ax1.set_xlabel('Iteration, $i$', fontsize=14)
    ax1.set_ylabel(r'Energy, $\mathcal{E}$', fontsize=14, color='black')

    # Plot the energy and store the label for the legend
    line1, = ax1.plot(iterations, energy_list, 'ko-', label=r'Energy, $\mathcal{E}(P_i)$', markersize=4)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Set axis limits to ensure the visibility of energy values
    ax1.set_ylim([min(energy_list) * 0.5, max(energy_list) * 1.5])

    # Add grid to the plot
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

    # Create a second y-axis for the norm of the gradient
    ax2 = ax1.twinx()  
    ax2.set_ylabel(r'Norm of energy gradient, $||\nabla \mathcal{E}||$', fontsize=14, color='green')

    # Plot the norm of the energy gradient and store the label for the legend
    line2, = ax2.plot(iterations, grad_norm_list, 'gs-', label=r'Norm of energy gradient, $||\nabla \mathcal{E}(P_i)||$', markersize=4)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='green', labelsize=12)

    # Set axis limits to ensure the visibility of total error values
    ax2.set_ylim([min(grad_norm_list) * 0.5, max(grad_norm_list) * 1.5])

    # Add legends for both plots (top right corner inside the grid)
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize=12)

    # Set titles and layout
    plt.title("Energy and  Norm of Energy Gradient vs Iteration", fontsize=16)
    fig.tight_layout()

    # Create 'Plots' folder if it doesn't exist
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    
    # Save the figure in the 'Plots' folder
    plt.savefig(os.path.join('Plots', figure_name), format='png', dpi=600)
    plt.show()