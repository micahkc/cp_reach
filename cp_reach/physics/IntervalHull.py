import numpy as np
import sys
from scipy.spatial import ConvexHull

def minBoundingRect(hull_points_2d):
    """
    Compute the minimum-area bounding rectangle for a 2D convex hull.

    Returns:
        angle: rotation angle in radians
        area: area of the bounding box
        width, height: dimensions of the box
        center_point: center of the box
        corner_points: 4 corner points of the box (in original coordinates)
    """
    edges = np.diff(hull_points_2d, axis=0)
    edge_angles = np.abs(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    edge_angles = np.unique(edge_angles)

    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0)  # angle, area, width, height, min_x, max_x, min_y, max_y

    for angle in edge_angles:
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        rot_points = R @ hull_points_2d.T
        min_x, max_x = np.min(rot_points[0]), np.max(rot_points[0])
        min_y, max_y = np.min(rot_points[1]), np.max(rot_points[1])
        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        if area < min_bbox[1]:
            min_bbox = (angle, area, width, height, min_x, max_x, min_y, max_y)

    angle = min_bbox[0]
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    min_x, max_x = min_bbox[4], min_bbox[5]
    min_y, max_y = min_bbox[6], min_bbox[7]

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_point = R.T @ np.array([center_x, center_y])

    corner_points = np.array([
        R.T @ [max_x, min_y],
        R.T @ [min_x, min_y],
        R.T @ [min_x, max_y],
        R.T @ [max_x, max_y]
    ])

    return angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points

def qhull2D(points):
    """
    Return the ordered convex hull boundary points from a set of 2D points.

    Parameters:
        points: ndarray of shape (N, 2)

    Returns:
        hull_points: ndarray of convex hull vertices in order
    """
    hull = ConvexHull(points)
    return points[hull.vertices]
