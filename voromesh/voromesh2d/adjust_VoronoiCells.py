from shapely.affinity import scale

def Interpolthickness(thickness_list, ratio_list, volume_fraction):
    """
    Interpolate thickness to adjust based on the target area ratio.

    Parameters
    ----------
    thickness_list : list of float
        List of thickness values tested in previous iterations.
    ratio_list : list of float
        Corresponding list of area ratios (volume fractions) achieved with each thickness.
    volume_fraction : float
        The target area ratio to achieve by interpolating a suitable thickness value.
    
    Returns
    -------
    float
        The interpolated thickness value that approximates the target area ratio.
    
    Notes
    -----
    - This function uses linear interpolation if two thickness-ratio points are provided and 
      quadratic interpolation if three or more points are given.
    - It is intended to help adjust thickness iteratively to achieve a target area ratio when
      exact thickness is not known.
    
    Examples
    --------
    Linear interpolation example:
    
    >>> thickness_list = [0.1, 0.2]
    >>> ratio_list = [0.8, 0.6]
    >>> volume_fraction = 0.7
    >>> Interpolthickness(thickness_list, ratio_list, volume_fraction)
    0.15  # Approximated thickness value

    Quadratic interpolation example:
    
    >>> thickness_list = [0.1, 0.2, 0.3]
    >>> ratio_list = [0.8, 0.6, 0.4]
    >>> volume_fraction = 0.5
    >>> Interpolthickness(thickness_list, ratio_list, volume_fraction)
    0.25  # Approximated thickness value based on quadratic interpolation
    
    Performance Tip
    ---------------
    For accurate results, ensure that thickness and ratio values cover a range close to the target
    volume fraction.
    """
    if len(thickness_list) < 2:
        # If there’s only one data point, return a linear estimate.
        return thickness_list[0] * volume_fraction / ratio_list[0]
    elif len(thickness_list) == 2:
        # Linear interpolation if we have two points
        x1, y1 = thickness_list[0], ratio_list[0]
        x2, y2 = thickness_list[1], ratio_list[1]
        return x1 + (volume_fraction - y1) * (x2 - x1) / (y2 - y1)
    else:
        # Quadratic interpolation if we have three or more points
        x1, y1 = thickness_list[-3], ratio_list[-3]
        x2, y2 = thickness_list[-2], ratio_list[-2]
        x3, y3 = thickness_list[-1], ratio_list[-1]
        
        # Using Lagrange's formula for quadratic interpolation
        L1 = ((volume_fraction - y2) * (volume_fraction - y3)) / ((y1 - y2) * (y1 - y3))
        L2 = ((volume_fraction - y1) * (volume_fraction - y3)) / ((y2 - y1) * (y2 - y3))
        L3 = ((volume_fraction - y1) * (volume_fraction - y2)) / ((y3 - y1) * (y3 - y2))
        
        return L1 * x1 + L2 * x2 + L3 * x3

def adjust_voronoi_cells(clipped_cells, thickness=None, volume_fraction=None, tolerance=0.005, max_iter=10):
    """
    Adjust Voronoi cells based on specified thickness or target area ratio (volume_fraction).

    Parameters
    ----------
    clipped_cells : list of shapely.geometry.Polygon
        List of Voronoi cells clipped to fit within a specific region.
    thickness : float, optional
        Thickness parameter to apply a uniform gap between adjacent cells. 
        If provided, volume_fraction is ignored.
    volume_fraction : float, optional
        Target area ratio to achieve by adjusting thickness iteratively.
        Used only if thickness is not provided.
    tolerance : float, optional
        Allowable tolerance for achieving the target volume_fraction. Default is 0.005.
    max_iter : int, optional
        Maximum number of iterations to reach the target volume_fraction. Default is 10.

    Returns
    -------
    list of shapely.geometry.Polygon
        List of adjusted Voronoi cells modified based on the specified thickness or area ratio.
    
    Raises
    ------
    ValueError
        If both thickness and volume_fraction are specified, or if neither is specified.
    
    Notes
    -----
    This function allows users to either apply a fixed thickness gap between Voronoi cells or
    iteratively adjust the thickness to reach a desired area coverage ratio within a specified
    tolerance. When adjusting to meet a volume_fraction, interpolation is used to estimate the
    thickness value in each iteration.
    
    Examples
    --------
    Adjusting with a fixed thickness:
    
    >>> from shapely.geometry import Polygon
    >>> from shapely.affinity import scale
    >>> clipped_cells = [Polygon([(0, 0), (1, 0), (0.5, 1)]), Polygon([(1, 0), (2, 0), (1.5, 1)])]
    >>> adjusted_cells = adjust_voronoi_cells(clipped_cells, thickness=0.1)
    >>> len(adjusted_cells)
    2

    Adjusting with a target volume_fraction:
    
    >>> adjusted_cells = adjust_voronoi_cells(clipped_cells, volume_fraction=0.75)
    >>> len(adjusted_cells)
    2
    
    Performance Tip
    ---------------
    - For precise control over area coverage, increase the max_iter or reduce tolerance.
    - Ensure that initial cells are well-defined polygons and not empty to avoid invalid adjustments.
    """
    if thickness is not None and volume_fraction is not None:
        raise ValueError("Specify either thickness or volume_fraction, not both.")
    
    adjusted_cells = []
    
    # If thickness is specified, adjust the cells directly and calculate the resulting volume_fraction
    if thickness is not None:
        initial_total_area = sum(cell.area for cell in clipped_cells)
        adjusted_total_area = 0

        for cell in clipped_cells:
            # Scale down the cell to create the gap defined by thickness
            adjusted_cell = scale(cell, xfact=(1 - thickness), yfact=(1 - thickness), origin='centroid')
            if adjusted_cell.is_valid and not adjusted_cell.is_empty:
                adjusted_cells.append(adjusted_cell)
                adjusted_total_area += adjusted_cell.area
        
        # Calculate the achieved volume_fraction
        achieved_volume_fraction = adjusted_total_area / initial_total_area
        print(f"Voronoi cells adjusted with a specified gap (thickness) of {thickness}.")
        print(f"Achieved area coverage (volume_fraction): {achieved_volume_fraction:.4f}")
        return adjusted_cells
    
    # If volume_fraction is specified, iteratively adjust thickness to achieve target area ratio
    elif volume_fraction is not None:
        thickness_list = []
        ratio_list = []
        current_thickness = 0.1  # Start with an initial small thickness
        initial_total_area = sum(cell.area for cell in clipped_cells)
        # target_area = initial_total_area * volume_fraction

        for _ in range(max_iter):
            modified_cells = []
            current_total_area = 0

            for cell in clipped_cells:
                # Scale the cell with the current thickness factor
                scaled_cell = scale(cell, xfact=(1 - current_thickness), yfact=(1 - current_thickness), origin='centroid')
                
                if scaled_cell.is_valid and not scaled_cell.is_empty:
                    modified_cells.append(scaled_cell)
                    current_total_area += scaled_cell.area

            # Calculate the current area ratio and store it for interpolation
            current_ratio = current_total_area / initial_total_area
            thickness_list.append(current_thickness)
            ratio_list.append(current_ratio)
            
            # Check if the current area ratio is within tolerance
            if abs(current_ratio - volume_fraction) <= tolerance:
                print(f"Achieved the target area ratio (volume_fraction) of {volume_fraction} with a thickness value of {current_thickness:.4f}.")
                return modified_cells

            # Update thickness using interpolation
            current_thickness = Interpolthickness(thickness_list, ratio_list, volume_fraction)

        # If maximum iterations reached, display the closest ratio achieved
        print(f"Warning: Maximum iterations reached. Closest area ratio achieved: {current_ratio:.4f} with thickness: {current_thickness:.4f}.")
        return modified_cells

    else:
        raise ValueError("Either thickness or volume_fraction must be specified.")