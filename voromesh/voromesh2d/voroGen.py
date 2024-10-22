import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point


# Function to implement Poisson's Disc Sampling to generate the seed points inside the random region
def generate_poisson_points(polygon, N, seed=None, k=30, tolerance=0.05):
    """
    Generate N points within the given polygon using Poisson's Disc Sampling.
    
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon within which to generate points.
    - N: The number of points to generate.
    - seed: Seed for the random number generator to ensure reproducibility.
    - k: Number of samples to choose before rejection in the algorithm (default is 30).
    - tolerance: The acceptable deviation from the desired number of points.
    
    Returns:
    - A list of points (x, y) that are within the polygon.
    """
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise ValueError("The input must be a Shapely Polygon or MultiPolygon.")
    
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    
    if seed is not None:
        np.random.seed(seed)
    
    area = polygon.area
    radius = np.sqrt(area / (N * np.pi))  # Initial estimated radius to fit approximately N points

    def poisson_disc_sampling(radius):
        min_x, min_y, max_x, max_y = polygon.bounds
        width, height = max_x - min_x, max_y - min_y

        # Cell side length
        cell_size = radius / np.sqrt(2)

        # Number of cells in the grid
        grid_width = int(np.ceil(width / cell_size))
        grid_height = int(np.ceil(height / cell_size))

        # Initialize grid and active list
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        active_list = []

        # Generate the initial point
        initial_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        while not polygon.contains(initial_point):
            initial_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))

        active_list.append(initial_point)
        grid_x, grid_y = int((initial_point.x - min_x) / cell_size), int((initial_point.y - min_y) / cell_size)
        grid[grid_x][grid_y] = initial_point

        points = [initial_point]

        while active_list:
            idx = np.random.randint(0, len(active_list))
            point = active_list[idx]
            found = False

            for _ in range(k):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(radius, 2 * radius)
                new_point = Point(point.x + r * np.cos(angle), point.y + r * np.sin(angle))

                if polygon.contains(new_point):
                    grid_x, grid_y = int((new_point.x - min_x) / cell_size), int((new_point.y - min_y) / cell_size)
                    if grid[grid_x][grid_y] is None:
                        # Check neighbors
                        too_close = False
                        for i in range(max(0, grid_x - 2), min(grid_width, grid_x + 3)):
                            for j in range(max(0, grid_y - 2), min(grid_height, grid_y + 3)):
                                neighbor = grid[i][j]
                                if neighbor and new_point.distance(neighbor) < radius:
                                    too_close = True
                                    break
                            if too_close:
                                break

                        if not too_close:
                            grid[grid_x][grid_y] = new_point
                            active_list.append(new_point)
                            points.append(new_point)
                            found = True

            if not found:
                active_list.pop(idx)

        return np.array([(p.x, p.y) for p in points])

    points = poisson_disc_sampling(radius)
    num_points = len(points)
    
    # Adjust radius iteratively to get close to N points
    iteration = 0
    while abs(num_points - N) > tolerance * N and iteration < 10:
        if num_points < N:
            radius *= 0.9  # Decrease radius to increase the number of points
        else:
            radius *= 1.1  # Increase radius to decrease the number of points

        points = poisson_disc_sampling(radius)
        num_points = len(points)
        iteration += 1

    return points


# Function to apply Lloyd's Algorithm to redistribute the initial seed points
def lloyds_algorithm_polygon(polygon, seed_points, iterations=100, tol=1e-5):
    """
    Apply Lloyd's algorithm to redistribute seed points for more uniform Voronoi cells within a polygonal region.

    Parameters:
    - polygon: Shapely Polygon or MultiPolygon within which to relax the points.
    - seed_points: Initial set of points (Nx2 array) within the polygon.
    - iterations: Number of iterations to perform.
    - tol: Tolerance for convergence, stop if points move less than this value.

    Returns:
    - A list of points (x, y) after applying Lloyd's Algorithm.
    """
    seed_points = np.array(seed_points)
    
    for _ in range(iterations):
        vor = Voronoi(seed_points)
        new_points = np.empty_like(seed_points)
        converged = True

        for point_idx, region_idx in enumerate(vor.point_region):
            vertices = vor.regions[region_idx]
            if all(v >= 0 for v in vertices):  # Check for bounded region
                region = [vor.vertices[i] for i in vertices]
                polygon_voronoi = Polygon(region)
                if not polygon_voronoi.is_valid:
                    polygon_voronoi = polygon_voronoi.buffer(0)  # Fix any geometry issues
                
                clipped_polygon = polygon_voronoi.intersection(polygon)
                if clipped_polygon.is_empty:
                    new_points[point_idx] = seed_points[point_idx]
                    continue
                
                centroid = np.array(clipped_polygon.centroid.coords[0])

                if polygon.contains(Point(centroid)):
                    new_points[point_idx] = centroid
                else:
                    new_points[point_idx] = seed_points[point_idx]
            else:
                new_points[point_idx] = seed_points[point_idx]

            if np.linalg.norm(new_points[point_idx] - seed_points[point_idx]) > tol:
                converged = False

        if converged:
            break

        seed_points = new_points

    return seed_points


# Helper function to convert initial Voronoi cells to shapely polygon
def voronoi_polygons(vor, bbox):
    """
    Generate Voronoi polygons clipped by a bounding box.
    
    Parameters:
    - vor: A scipy.spatial.Voronoi object.
    - bbox: A bounding box as a shapely Polygon object to clip the Voronoi cells.
    
    Returns:
    - A list of shapely.geometry.Polygon objects representing the Voronoi cells.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    # radius = vor.points.ptp().max()
    radius = np.ptp(vor.points, axis=0).max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            far_point = vor.vertices[v2] + np.sign(np.dot(midpoint - center, n)) * n * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    # Create Voronoi polygons and clip with bounding box
    voronoi_polygons = [Polygon([new_vertices[v] for v in region]).intersection(bbox) for region in new_regions]

    # Filter out invalid geometries
    voronoi_polygons = [poly for poly in voronoi_polygons if poly.is_valid and not poly.is_empty]

    return voronoi_polygons


# Function to generate Voronoi cells inside the random region
def generate_voronoi_cells(polygon, points):
    """
    Generate Voronoi cells within the given polygon for the provided points.
    
    Parameters
    ----------
    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A Shapely Polygon or MultiPolygon defining the region within which to generate Voronoi cells.
    points : array_like, shape (N, 2)
        Array of (x, y) coordinates representing the points for which to generate the Voronoi diagram.
    
    Returns
    -------
    list of shapely.geometry.Polygon
        List of Shapely Polygon objects representing the clipped Voronoi cells.

    Notes
    -----
    This function generates Voronoi cells for a given set of points and clips them to fit within the specified
    polygon. The Voronoi cells that fall outside the polygon are clipped, ensuring all cells lie within the polygonal
    region. Invalid cells (empty or invalid geometries) are filtered out.

    Examples
    --------
    Basic usage:
    
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> points = np.random.rand(10, 2)
    >>> voronoi_cells = generate_voronoi_cells(polygon, points)
    >>> len(voronoi_cells)  # Number of Voronoi cells inside the polygon
    10
    
    Advanced usage with a larger domain:
    
    >>> large_polygon = Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)])
    >>> points = np.random.rand(100, 2) * 10 - 5  # Random points within a larger domain
    >>> voronoi_cells = generate_voronoi_cells(large_polygon, points)
    >>> len(voronoi_cells)
    100

    Performance Tip
    ---------------
    - For a large number of points, consider generating Voronoi cells in parallel or applying a bounding box with 
      slightly extended edges to improve performance when clipping large Voronoi regions.
    - The quality and shape of the Voronoi cells depend on the initial distribution of points. Use more uniform
      distributions for smoother Voronoi cells.
    """
    vor = Voronoi(points)
    min_x, min_y, max_x, max_y = polygon.bounds
    bbox = Polygon([
        (min_x - 1, min_y - 1), 
        (min_x - 1, max_y + 1), 
        (max_x + 1, max_y + 1), 
        (max_x + 1, min_y - 1)
    ])
    voronoi_cells = voronoi_polygons(vor, bbox)
    
    # Clip Voronoi cells to fit within the specified polygon and filter out invalid cells
    clipped_cells = [cell.intersection(polygon) for cell in voronoi_cells if cell.is_valid and not cell.is_empty]
    
    return clipped_cells