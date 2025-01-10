from typing import List, Tuple
from shapely.geometry import Polygon
from shapely.affinity import scale

class CohesiveZoneAdjuster:
    """
    A class for adjusting Voronoi cells to introduce finite-thickness cohesive zones.
    
    This class provides methods for modifying Voronoi cells by either applying a fixed
    thickness gap or iteratively adjusting to achieve a target area ratio.
    
    Parameters
    ----------
    tolerance : float, optional
        Convergence tolerance for area ratio, default=0.005
    max_iterations : int, optional
        Maximum number of adjustment iterations, default=10
    verbose : bool, optional
        Whether to print progress information, default=False
        
    Attributes
    ----------
    history : List[Tuple[float, float]]
        History of (thickness, ratio) pairs from iterations
    
    Notes
    -----
    The adjustment process follows two possible approaches:
    
    1. Fixed thickness (t):
       Each cell Vᵢ is scaled as V'ᵢ = S_{(1-t)}(Vᵢ)
       
    2. Target area ratio (r*):
       Iteratively adjusts thickness until:
       |rₖ - r*| ≤ ε or k > K
       
    where:
    - S_α is the scaling operator
    - rₖ is the area ratio at iteration k
    - ε is the tolerance
    - K is max_iterations
    """
    
    def __init__(
        self,
        tolerance: float = 0.005,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history: List[Tuple[float, float]] = []
        
    def _interpolate_thickness(
        self,
        target_ratio: float
    ) -> float:
        """
        Interpolate thickness using Lagrange polynomials.
        
        Parameters
        ----------
        target_ratio : float
            Target area ratio to achieve
            
        Returns
        -------
        float
            Interpolated thickness value
            
        Notes
        -----
        Uses linear or quadratic interpolation based on available history:
        - n=1: Linear scaling
        - n=2: Linear interpolation
        - n≥3: Quadratic interpolation using last 3 points
        """
        if len(self.history) < 2:
            # Linear scaling for single point
            thickness, ratio = self.history[0]
            return thickness * target_ratio / ratio
            
        elif len(self.history) == 2:
            # Linear interpolation
            (t1, r1), (t2, r2) = self.history[-2:]
            return t1 + (target_ratio - r1) * (t2 - t1) / (r2 - r1)
            
        else:
            # Quadratic interpolation using last 3 points
            points = self.history[-3:]
            thicknesses, ratios = zip(*points)
            
            # Lagrange basis polynomials
            L = []
            for i in range(3):
                product = 1.0
                for j in range(3):
                    if i != j:
                        product *= ((target_ratio - ratios[j]) / 
                                  (ratios[i] - ratios[j]))
                L.append(product)
                
            return sum(L[i] * thicknesses[i] for i in range(3))
        
    def _adjust_cells(
        self,
        cells: List[Polygon],
        thickness: float
    ) -> Tuple[List[Polygon], float]:
        """
        Adjust cells using given thickness factor.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        thickness : float
            Scaling factor to apply
            
        Returns
        -------
        Tuple[List[Polygon], float]
            Adjusted cells and achieved area ratio
        """
        initial_area = sum(cell.area for cell in cells)
        adjusted_cells = []
        current_area = 0.0
        
        for cell in cells:
            adjusted = scale(
                cell,
                xfact=(1 - thickness),
                yfact=(1 - thickness),
                origin='centroid'
            )
            
            if adjusted.is_valid and not adjusted.is_empty:
                adjusted_cells.append(adjusted)
                current_area += adjusted.area
                
        achieved_ratio = current_area / initial_area
        return adjusted_cells, achieved_ratio
        
    def adjust_fixed_thickness(
        self,
        cells: List[Polygon],
        thickness: float
    ) -> List[Polygon]:
        """
        Adjust cells using a fixed thickness value.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        thickness : float
            Fixed gap thickness to apply
            
        Returns
        -------
        List[Polygon]
            Adjusted Voronoi cells
            
        Examples
        --------
        >>> adjuster = CohesiveZoneAdjuster()
        >>> adjusted = adjuster.adjust_fixed_thickness(cells, thickness=0.1)
        """
        adjusted_cells, achieved_ratio = self._adjust_cells(cells, thickness)
        
        if self.verbose:
            print(f"Applied thickness: {thickness:.4f}")
            print(f"Achieved area ratio: {achieved_ratio:.4f}")
            
        return adjusted_cells
        
    def adjust_target_ratio(
        self,
        cells: List[Polygon],
        target_ratio: float,
        initial_thickness: float = 0.1
    ) -> List[Polygon]:
        """
        Iteratively adjust cells to achieve target area ratio.
        
        Parameters
        ----------
        cells : List[Polygon]
            Input Voronoi cells
        target_ratio : float
            Target area ratio to achieve
        initial_thickness : float, optional
            Starting thickness value, default=0.1
            
        Returns
        -------
        List[Polygon]
            Adjusted Voronoi cells
            
        Examples
        --------
        >>> adjuster = CohesiveZoneAdjuster(tolerance=0.001)
        >>> adjusted = adjuster.adjust_target_ratio(cells, target_ratio=0.75)
        """
        self.history.clear()
        current_thickness = initial_thickness
        
        for iteration in range(self.max_iterations):
            # Adjust cells with current thickness
            adjusted_cells, achieved_ratio = self._adjust_cells(
                cells,
                current_thickness
            )
            
            # Update history
            self.history.append((current_thickness, achieved_ratio))
            
            # Check convergence
            if abs(achieved_ratio - target_ratio) <= self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                    print(f"Final thickness: {current_thickness:.4f}")
                    print(f"Achieved ratio: {achieved_ratio:.4f}")
                return adjusted_cells
                
            # Update thickness through interpolation
            current_thickness = self._interpolate_thickness(target_ratio)
            
            if self.verbose:
                print(f"Iteration {iteration + 1}:")
                print(f"  Thickness: {current_thickness:.4f}")
                print(f"  Area ratio: {achieved_ratio:.4f}")
                
        # Maximum iterations reached
        if self.verbose:
            print("Warning: Maximum iterations reached")
            print(f"Best ratio: {achieved_ratio:.4f}")
            print(f"Final thickness: {current_thickness:.4f}")
            
        return adjusted_cells