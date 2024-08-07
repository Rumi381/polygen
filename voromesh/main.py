import os
import sys
import argparse

# Add the project root directory to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

from voromesh.voromesh2d import parse_input_file, computeVoronoi2d
# from voromesh.voromesh3d import computeVoronoi3d  # Uncomment when 3D functionality is available

def main_2d(input_file_path):
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

def main_3d(input_file_path):
    # Placeholder for 3D Voronoi computation
    # parameters = parse_input_file(input_file_path)
    # Call the computeVoronoi3d function with the extracted parameters
    pass

def main():
    parser = argparse.ArgumentParser(description='Voronoi Mesh Generation CLI')
    parser.add_argument('-d', '--dimension', choices=['2', '3'], help='Specify the dimension for Voronoi mesh generation')
    parser.add_argument('input_file', help='Path to the input file')

    args = parser.parse_args()

    input_file_path = os.path.abspath(args.input_file)

    if not os.path.exists(input_file_path):
        print(f"Input file '{input_file_path}' does not exist.")
        sys.exit(1)

    if args.dimension == '2':
        main_2d(input_file_path)
    elif args.dimension == '3':
        print("3D functionality not yet implemented.")
        main_3d(input_file_path)

if __name__ == "__main__":
    main()

