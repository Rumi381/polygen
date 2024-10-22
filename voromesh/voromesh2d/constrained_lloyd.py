import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point
from scipy.spatial import Voronoi
import time

def lloyd(polygon, seed_points, max_iterations=100, tol=1e-5, use_decay=True, decay_start=2.0):
    """
    Apply Lloyd's algorithm to redistribute seed points for more uniform Voronoi cells within a polygonal region.
    
    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        A Shapely polygon or multipolygon that defines the boundary within which the points are to be redistributed.
    seed_points : array_like, shape (N, 2)
        Initial set of points within the polygon, representing the seed points to be relaxed.
    max_iterations : int, optional
        Maximum number of iterations to perform. Default is 100.
    tol : float, optional
        Tolerance for convergence. The algorithm stops when the norm of the energy gradient falls below this value.
        Default is 1e-5.
    use_decay : bool, optional
        If True, a decay factor is applied to the point movement, which gradually reduces the magnitude of point shifts
        as iterations progress. Default is True.
    decay_start : float, optional
        Initial decay factor applied to the point movement in the first iteration. This value reduces in subsequent
        iterations if `use_decay` is True. Default is 2.0.
    
    Returns
    -------
    seed_points : array_like, shape (N, 2)
        The updated seed points after applying Lloyd's algorithm.
    
    Raises
    ------
    ValueError
        If the provided `polygon` is not valid or if `seed_points` are outside the polygon.
    
    Notes
    -----
    Lloyd's algorithm is an iterative process to improve the uniformity of Voronoi cells by moving each seed point 
    to the centroid of its corresponding Voronoi cell. This version of the algorithm includes optional decay to 
    reduce the magnitude of point movement over time, which helps prevent large fluctuations after nearing convergence.
    
    The algorithm terminates either when the norm of the energy gradient falls below the tolerance `tol` or if the 
    energy gradient norm increases after reaching a minimum, which suggests the system has converged with some fluctuations.
    
    Time taken for convergence is printed along with the number of iterations required.

    Examples
    --------
    Basic usage with a polygon and seed points:
    
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> seed_points = np.random.rand(10, 2)
    >>> relaxed_points = lloyd(polygon, seed_points)
    
    Applying Lloyd's algorithm with a decay factor:
    
    >>> relaxed_points = lloyd(polygon, seed_points, use_decay=True, decay_start=2.0)
    
    Increasing maximum iterations for more precision:
    
    >>> relaxed_points = lloyd(polygon, seed_points, max_iterations=200, tol=1e-6)

    Performance Tip
    ---------------
    When working with large polygons and a large number of seed points, consider increasing the `tol` to speed up convergence.
    
    """
    seed_points = np.array(seed_points)
    min_grad_norm = float('inf')  # Track minimum norm of energy gradient

    if use_decay:
        decay_values = np.exp(-np.linspace(0, max_iterations, max_iterations) / max_iterations * np.log(0.1)) * (decay_start - 1) + 1

    # Start timer
    start_time = time.time()

    for iteration in range(max_iterations):
        vor = Voronoi(seed_points)
        new_points = np.empty_like(seed_points)
        total_grad_norm = 0

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

                # Calculate the centroid using the area-weighted centroid formula
                # centroid = calculate_polygon_centroid(clipped_polygon)
                centroid = np.array(clipped_polygon.centroid.coords[0])
                
                # Norm of gradient is the movement from current point to the centroid
                grad_norm = np.linalg.norm(centroid - seed_points[point_idx])
                total_grad_norm += grad_norm
                
                # Update the point position
                if polygon.contains(Point(centroid)):
                    if use_decay:
                        decay = decay_values[iteration]
                        new_points[point_idx] = seed_points[point_idx] + (centroid - seed_points[point_idx]) * decay
                    else:
                        new_points[point_idx] = centroid
                else:
                    new_points[point_idx] = seed_points[point_idx]
            else:
                new_points[point_idx] = seed_points[point_idx]

        # Check for convergence using the norm of energy gradient
        total_grad_norm /= len(seed_points)  # Average norm of energy gradient
        if total_grad_norm < tol:
            elapsed_time = time.time() - start_time
            print(f"Converged after {iteration+1} iterations with norm of energy gradient: {total_grad_norm:.5e}")
            print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")
            break
        
        # Track the minimum norm of energy gradient and stop if the norm of energy gradient increases after reaching the minimum
        if total_grad_norm < min_grad_norm:
            min_grad_norm = total_grad_norm  # Update minimum norm of energy gradient
        elif total_grad_norm > min_grad_norm:
            elapsed_time = time.time() - start_time
            print(f"Minimum value of energy gradient norm found. Stopped at iteration {iteration+1} with norm of energy gradient: {total_grad_norm:.5e}")
            print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")
            break

        seed_points = new_points

    else:
        elapsed_time = time.time() - start_time
        print(f"Reached the maximum number of iterations ({max_iterations}) with norm of energy gradient: {total_grad_norm:.5e}")
        print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")

    return seed_points


# Function to calculate and return the energy and norm of the energy gradient in each iteration (For analysis)
def lloyd_with_energyCalculation(polygon, seed_points, max_iterations=100, tol=1e-5, use_decay=True, decay_start=2.0):
    """
    Apply Lloyd's algorithm to redistribute seed points for more uniform Voronoi cells within a polygonal region,
    while tracking the energy and norm of the energy gradient at each iteration.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        A Shapely polygon or multipolygon that defines the boundary within which the points are to be redistributed.
    seed_points : array_like, shape (N, 2)
        Initial set of points within the polygon, representing the seed points to be relaxed.
    max_iterations : int, optional
        Maximum number of iterations to perform. Default is 100.
    tol : float, optional
        Tolerance for convergence. The algorithm stops when the norm of the energy gradient falls below this value.
        Default is 1e-5.
    use_decay : bool, optional
        If True, a decay factor is applied to the point movement, which gradually reduces the magnitude of point shifts
        as iterations progress. Default is True.
    decay_start : float, optional
        Initial decay factor applied to the point movement in the first iteration. This value reduces in subsequent
        iterations if `use_decay` is True. Default is 2.0.

    Returns
    -------
    seed_points : array_like, shape (N, 2)
        The updated seed points after applying Lloyd's algorithm.
    energy_list : list of float
        List of energy values at each iteration, where energy is defined as the sum of squared distances between each
        seed point and its corresponding Voronoi centroid.
    grad_norm_list : list of float
        List of norm values of the energy gradient at each iteration, representing the average movement of seed points
        toward their centroids.

    Raises
    ------
    ValueError
        If the provided `polygon` is not valid or if `seed_points` are outside the polygon.

    Notes
    -----
    Lloyd's algorithm iteratively adjusts the seed points by moving them toward the centroids of their corresponding 
    Voronoi cells, reducing the variation in Voronoi cell sizes. In this version of the algorithm, both the energy 
    (sum of squared distances between seed points and centroids) and the norm of the energy gradient are tracked at 
    each iteration for analysis purposes.
    
    The algorithm terminates either when the norm of the energy gradient falls below the tolerance `tol` or if the 
    energy gradient norm increases after reaching a minimum. This helps in detecting convergence with some fluctuations.

    Examples
    --------
    Basic usage with energy tracking:
    
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> seed_points = np.random.rand(10, 2)
    >>> relaxed_points, energy, grad_norm = lloyd_with_energyCalculation(polygon, seed_points)
    
    Tracking energy and norm of the energy gradient over iterations:
    
    >>> relaxed_points, energy, grad_norm = lloyd_with_energyCalculation(polygon, seed_points)
    >>> print("Energy at each iteration:", energy)
    >>> print("Norm of energy gradient at each iteration:", grad_norm)
    
    Applying Lloyd's algorithm with a decay factor:
    
    >>> relaxed_points, energy, grad_norm = lloyd_with_energyCalculation(polygon, seed_points, use_decay=True, decay_start=2.0)

    Performance Tip
    ---------------
    If tracking energy and gradient norm is not needed for analysis, consider using the `lloyd` function without energy
    calculation to save computational resources.
    
    """
    seed_points = np.array(seed_points)
    
    # Initialize lists to store energy and norm of energy gradient values
    energy_list = []
    grad_norm_list = []
    
    min_grad_norm = float('inf')  # Track minimum norm of energy gradient

    if use_decay:
        decay_values = np.exp(-np.linspace(0, max_iterations, max_iterations) / max_iterations * np.log(0.1)) * (decay_start - 1) + 1

    # Start timer
    start_time = time.time()

    for iteration in range(max_iterations):
        vor = Voronoi(seed_points)
        new_points = np.empty_like(seed_points)
        total_energy = 0
        total_grad_norm = 0

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

                # Calculate the centroid using the area-weighted centroid formula
                # centroid = calculate_polygon_centroid(clipped_polygon)
                centroid = np.array(clipped_polygon.centroid.coords[0])
                
                # Calculate the distance between the seed point and its centroid (this contributes to energy)
                dist = np.linalg.norm(centroid - seed_points[point_idx])
                total_energy += dist ** 2  # Energy as the sum of squared distances
                
                # Norm of gradient is the movement from current point to the centroid
                grad_norm = np.linalg.norm(centroid - seed_points[point_idx])
                total_grad_norm += grad_norm
                
                # Update the point position
                if polygon.contains(Point(centroid)):
                    if use_decay:
                        decay = decay_values[iteration]
                        new_points[point_idx] = seed_points[point_idx] + (centroid - seed_points[point_idx]) * decay
                    else:
                        new_points[point_idx] = centroid
                else:
                    new_points[point_idx] = seed_points[point_idx]
            else:
                new_points[point_idx] = seed_points[point_idx]

        # Append the total energy and norm of energy gradient for this iteration
        energy_list.append(total_energy / len(seed_points))  # Average energy
        grad_norm_list.append(total_grad_norm / len(seed_points))  # Average norm of energy gradient
        
        # Check for convergence using the norm of energy gradient
        if grad_norm_list[-1] < tol:
            elapsed_time = time.time() - start_time
            print(f"Converged after {iteration+1} iterations with error: {total_grad_norm:.5e}")
            print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")
            break
        
        # Track the minimum norm of energy gradient and stop if the norm of energy gradient increases after reaching the minimum
        if grad_norm_list[-1] < min_grad_norm:
            min_grad_norm = grad_norm_list[-1]  # Update minimum norm of energy gradient
        elif grad_norm_list[-1] > min_grad_norm:
            elapsed_time = time.time() - start_time
            print(f"Minimum value of energy gradient norm found. Stopped at iteration {iteration+1} with norm of energy gradient: {grad_norm_list[-1]:.5e}")
            print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")
            break

        seed_points = new_points

    else:
        elapsed_time = time.time() - start_time
        print(f"Reached the maximum number of iterations ({max_iterations}) with error: {grad_norm_list[-1]:.5e}")
        print(f"Total time taken for the Lloyd's iterations: {elapsed_time:.5f} seconds")

    return seed_points, energy_list, grad_norm_list


def _calculate_polygon_centroid(polygon):
    if polygon.is_empty:
        return np.array([0.0, 0.0])
    
    # Handle MultiPolygon case
    if isinstance(polygon, MultiPolygon):
        total_area = 0.0
        weighted_centroid = np.array([0.0, 0.0])
        
        for poly in polygon.geoms:
            area = poly.area
            centroid = poly.centroid
            weighted_centroid += np.array([centroid.x, centroid.y]) * area
            total_area += area
        
        # Avoid division by zero in case of zero total area
        if total_area == 0.0:
            return np.array([0.0, 0.0])
        
        return weighted_centroid / total_area
    
    # For simple Polygon
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y])