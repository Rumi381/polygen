import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point

# Plotting the original Voronoi cells without any boudary filtering
def plot_voronoi_cells(figure_name, voronoi_cells, points=None, marker_size=10, fill_color=False, show_labels=False, show_title=False, title='Original Voronoi Cells'):
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
    plt.show()
    
    

# Plotting the original Voronoi cells with boudary filtering    
def plot_voronoi_cells_withBoundaryFiltering(figure_name, polygon, voronoi_cells, points=None, marker_size=10, fill_color=False, show_labels=False, show_title=False, title='Original Voronoi Cells with Boundary Filtering'):
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
    plt.show()
   

# Plotting the polygon boundaries from the given polygon region and also the seed points    
def plot_boundary_with_points(figure_name, polygon, points=None, marker_size=1, show_labels=False, show_title=False, title='Boundaries with Initial Seed Points'):
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
    plt.show()
    
    

# Plotting the original Voronoi cells highlighting the targeted shortest edges with either N th or a threshold length
def plot_voronoi_cells_with_short_edges(figure_name, voronoi_cells, N=None, threshold=None, points=None, marker_size=10, show_labels=False, show_title=False, title='Original Voronoi Cells Highlighting Short Edges'):
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
    plt.show()
    
# Fuction to plot the Polygon boundaries highlighting the short edges
def plot_boundary_with_short_edges(figure_name, polygon, N=None, threshold=None, points=None, marker_size=1, show_labels=False, show_title=False, title='Boundaries Highlighting Short Edges'):
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
    plt.show()


# ==========================================================================================================
# ==========================================================================================================
# Plotting the polygon boundaries from the given polygon region and also the original Voronoi cells (Extra function, may need later)    
def plot_polygonBoundary_with_originalVoronoi_cells(figure_name, polygon, voronoi_cells, points=None, marker_size=10):
    """
    Plot Voronoi cells within a given polygon and overlay seed points.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon defining the boundary.
    - voronoi_cells: List of Shapely Polygon objects representing the Voronoi cells.
    - points: Optional 2D numpy array of points (x, y) to overlay on the plot.
    - marker_size: Size of the markers for the points.
    """
    fig, ax = plt.subplots()
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
        plot_polygon(polygon, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary')
        label_added = True
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            if not label_added:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2, label='Polygon Boundary')
                label_added = True
            else:
                plot_polygon(poly, color_ex='black', color_in='red', linestyle='-', linewidth=2)
    
    # Plot the Voronoi cells
    for cell in voronoi_cells:
        if isinstance(cell, Polygon):
            x, y = cell.exterior.xy
            ax.plot(x, y, 'blue')  # Plot the Voronoi cell boundary
        elif isinstance(cell, MultiPolygon):
            for poly in cell.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, 'blue')  # Plot the boundary of each polygon

    # Overlay the points, if provided
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', marker='.', s=marker_size, label='Seed Points')
    
    ax.set_aspect('equal')
    plt.legend()
    plt.savefig(figure_name, format='png', dpi=600)
    plt.show()