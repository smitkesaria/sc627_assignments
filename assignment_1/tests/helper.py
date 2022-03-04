#!/usr/bin/env python3

"""SC 627 utility functions."""

# standard imports
import time
from math import sqrt
from typing import Tuple, List
from math import sqrt, hypot, pi, cos, sin, atan
from typing import Tuple

# app imports
class Point:
    """Data Structure for Point."""

    def __init__(self, x: float, y: float):
        """Initialize."""
        self.x = x  # ordinate
        self.y = y  # abscissa

    def __eq__(self, p):
        """Override the default equality check implementation."""
        if isinstance(p, Point):
            return round(self.x, 10) == round(p.x, 10) and round(self.y, 10) == round(p.y, 10)
        return False

    def get_distance(self, q) -> float:
        """Get distance between 2 points."""
        return hypot((self.x - q.x), (self.y - q.y))

    def move_magnitude_towards(self, v, magnitude: float):
        """Move this point an order of magnitude in directon of vector v.

        Input: Vector v, magnitude m
        Output: new Point
        """
        if v.x == 0:
            new_x = self.x
            y_sign = -1 if v.y < 0 else 1
            new_y = self.y + magnitude * y_sign
        else:
            theta = abs(atan(v.y / v.x))
            x_sign = -1 if v.x < 0 else 1
            y_sign = -1 if v.y < 0 else 1
            new_x = self.x + magnitude * cos(theta) * x_sign
            new_y = self.y + magnitude * sin(theta) * y_sign
        return Point(new_x, new_y)

    def __str__(self):
        return f"Point({self.x}, {self.y})"
        return f"Point({self.x:.7f}, {self.y:.7f})"

    def __repr__(self):
        return self.__str__()


class Line:
    """Data Structure for Line."""

    def __init__(self, **kwargs):
        """Initialize.

        ax + by + c = 0 and normalized form requires a^2 + b^2 = 1

        Assume,

        a' = a/X
        b' = b/X

        where X is the normalization factor

        and given a'^2 + b'^2 = 1
        => a^2 + b^2 = X^2
        => X = sqrt(a^2 + b^2)

        also if any 1 of a or b is 0, make the other positive by dividing by -X instead of X
        also try to make x coordinate positive for standardization

        Getting slope and y-intercept from normalized form,
        ax + by + c = 0
        => by = -ax - c
        => y = (-a/b)x + (-c/b)

        therefore slope = m = -a/b
        and y-intercept c = -c/b

        Getting normalized form from slope-intercept form,
        y = mx + c
        => mx - y + c = 0
        => a=m, b=-1, c=c
        """
        self.a = None
        self.b = None
        self.c = None

        self.slope = None
        self.y_intercept = None

        if "normalized-form" in kwargs:
            a, b, c = kwargs.get("normalized-form")

            normalization_factor = sqrt(a**2 + b**2)

            if (a == 0 and b < 0) or (b == 0 and a < 0) or (b > 0 and a < 0):
                normalization_factor = -normalization_factor

            a = a / normalization_factor
            b = b / normalization_factor
            c = c / normalization_factor

            self.a = a
            self.b = b
            self.c = c

            if self.b == 0:
                self.slope = float("inf")
            else:
                self.slope = -self.a / self.b
            if self.b == 0:
                self.y_intercept = float("inf")
            else:
                self.y_intercept = -self.c / self.b

        elif "slope-intercept-form" in kwargs:
            self.slope, self.y_intercept = kwargs.get("slope-intercept-form")
            a = self.slope
            b = -1
            c = self.y_intercept

            normalization_factor = sqrt(a**2 + b**2)

            if (a == 0 and b < 0) or (b == 0 and a < 0) or (b > 0 and a < 0):
                normalization_factor = -normalization_factor

            a = a / normalization_factor
            b = b / normalization_factor
            c = c / normalization_factor

            self.a = a
            self.b = b
            self.c = c

    def __str__(self):
        return f"Line[({self.a})x + ({self.b})y + ({self.c}) = 0]"

    def get_normalized_form(self) -> Tuple[float, float, float]:
        """Get normalized paramas a, b, c for line in form ax+by+c=0, where a^2 + b^2 = 1."""
        return self.a, self.b, self.c

    def get_not_so_random_point_on_line(self) -> Point:
        if self.a == 0:  # by + c = 0 => y = -c/b
            return Point(0, 0.0 - self.c / self.b)
        elif self.b == 0:  # ax + c = 0 => x = -c/a
            return Point(0.0 - self.c / self.a, 0)
        else:
            return Point(0, 0.0 - self.c / self.b)

    def get_two_points_on_either_side_of_given_point_on_line(self, p: Point) -> Tuple[Point, Point]:
        """
        ax + by + c = 0
        y = - (c + ax) / b
        """
        if self.a == 0:
            return Point(p.x + 1, p.y), Point(p.x - 1, p.y)
        if self.b == 0:
            return Point(p.x, p.y + 1), Point(p.x, p.y - 1)
        p1_x = p.x + 1
        p1_y = - (self.c + self.a * p1_x) / self.b

        p2_x = p.x - 1
        p2_y = - (self.c + self.a * p2_x) / self.b

        return Point(p1_x, p1_y), Point(p2_x, p2_y)

    def get_intersection_point(self, l) -> Point:
        """
        L1 -> a1x + b1y + c1 = 0
        L2 -> a2x + b2y + c2 = 0

        From L1 -> x = -(c1 + b1y) / a1

        Use x from L1 in L2
        => a2 (-(c1 + b1y) / a1) + b2y + c2 = 0
        => a2 (-(c1 + b1y)) + b2.y.a1 + c2.a1 = 0
        =>

        """
        # print(f"Get intersection of {self} and {l}")
        intersection_y = (l.a * self.c - self.a * l.c) / (self.a * l.b - l.a * self.b)
        if self.a == 0:
            intersection_x = - l.c / l.a
        else:
            intersection_x = -(self.b * intersection_y + self.c) / self.a
        return Point(intersection_x, intersection_y)

    def get_perpendicular_line_through_point(self, q: Point):  # returns Line
        # take negetive inverse of that slope, this'll be the slope of the perpendicular from the point onto the line
        if self.slope == 0:  # y = c
            a = 1.0
            b = 0.0
            c = -q.x
        else:
            slope_of_perpendicular = -1 / self.slope
            a = slope_of_perpendicular
            b = -1.0
            c = q.y - slope_of_perpendicular * q.x
        # calculate equation of line from point slope form
        return Line(**{"normalized-form": (a, b, c)})

    def get_parallel_line_through_point(self, q: Point):  # returns Line
        """Return a Line, parallel to self, passing through the given point.

        Equation of line in point-slope form:
        => y - y1 = m(x - x1)
        => y - y1 = mx - m.x1
        => mx - y + y1 - m.x1= 0
        => a=m; b=-1; c=y1-m.x1
        """
        if self.slope == float("inf"):  # x = c
            a = 1.0
            b = 0.0
            c = q.x
        else:
            slope_of_parallel = self.slope
            a = slope_of_parallel
            b = -1.0
            c = q.y - slope_of_parallel * q.x
        # calculate equation of line from point slope form
        return Line(**{"normalized-form": (a, b, c)})

    def is_point_on_line(self, p: Point) -> bool:
        return self.a * p.x + self.b * p.y + self.c == 0

    def are_points_on_same_side_of_line(self, p1: Point, p2: Point) -> bool:
        p1_line_val = self.a * p1.x + self.b * p1.y + self.c
        p2_line_val = self.a * p2.x + self.b * p2.y + self.c
        if (p1_line_val > 0 and p2_line_val > 0) or (
            p1_line_val < 0 and p2_line_val < 0
        ):
            return True
        return False

    def are_points_on_opposite_side_of_line(self, p1: Point, p2: Point) -> bool:
        return not self.are_points_on_same_side_of_line(p1, p2)


class Polygon:
    """Data Structure for Polygon.

    A polygon with n vertices is represented as an array with n rows and 2 columns.
    """

    def __init__(self, vertices: Tuple[Point, ...]):
        self.vertices = vertices

    def __str__(
        self,
    ):
        ret_val = "Polygon["
        for p in self.vertices:
            ret_val += f"({p.x}, {p.y}), "
        ret_val.strip(", ")
        ret_val += "]"
        return ret_val

    def __repr__(self):
        return self.__str__()

    def get_next_vertex_on_polygon(
        self, segment: Tuple[Point, Point] = None, vertex: Point = None
    ) -> Point:
        """Get next vertex on polygon segment.

        Implicit assumption of polygon vertices always stored in counter clockwise direction
        """
        # print(f"Getting next vertex on Polygon for segment: {segment} or vertex: {vertex}")
        if segment:
            p1_index = self.vertices.index(segment[0])
            p2_index = self.vertices.index(segment[1])
            # print(f"Segment indices: {p1_index}, {p2_index}")
            req_index = max(p1_index, p2_index)
            if (p1_index == 0 and p2_index == len(self.vertices) - 1) or (p2_index == 0 and p1_index == len(self.vertices) - 1):
                req_index = 0
        elif vertex:
            index = self.vertices.index(vertex)
            req_index = index + 1
            if req_index == len(self.vertices):
                req_index = 0
        return self.vertices[req_index]


class Vector:
    """Data Structure for Vector."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, v):
        """Override the default equality check implementation."""
        if isinstance(v, Vector):
            return round(self.x, 10) == round(v.x, 10) and round(self.y, 10) == round(v.y, 10)
        return False

    def __str__(self):
        return f"Vector({self.x}, {self.y})"
        return f"Vector({self.x:.7f})x + ({self.y:.7f})y"

    def __repr__(self):
        return self.__str__()


def computeLineThroughTwoPoints(p1: Point, p2: Point) -> Tuple[float, float, float]:
    """Compute equation parameters of line through 2 points.

    Input: two distinct points p1 = (x1, y1) and p2 = (x2, y2) on the plane.
    Output: parameters (a, b, c) defining the line {(x, y) | ax + by + c = 0} that passes
    through both p1 and p2. Normalize the parameters so that a^2 + b^2 = 1.

    Notes:
    Line in 2 point form equation: (y − y2) = ((y2 − y1) / (x2 − x1)) * (x − x2)

    => (y − y2) * (x2 − x1) = (y2 − y1) * (x − x2)
    => y.x2 - y.x1 - y2.x2 + y2.x1 = y2.x - y2.x2 - y1.x + y1.x2
    => y2.x - y2.x2 - y1.x + y1.x2 - y.x2 + y.x1 + y2.x2 - y2.x1 = 0
    => (y2 - y1)x + (x1 - x2)y + (y1.x2 - y2.x2 + y2.x2 - y2.x1) = 0

    Compare with ax + by + c = 0
    We have,
    a = y2 - y1
    b = x1 - x2
    c = y1.x2 - y2.x2 + y2.x2 - y2.x1
    """
    a = p2.y - p1.y
    b = p1.x - p2.x
    c = p1.y * p2.x - p2.y * p2.x + p2.y * p2.x - p2.y * p1.x

    line = Line(**{"normalized-form": (a, b, c)})

    return line.get_normalized_form()


def computeDistancePointToLine(q: Point, p1: Point, p2: Point) -> float:
    """Compute distance of a point from a line.

    Input: a point q and two distinct points p1 = (x1, y1) and p2 = (x2, y2) defining a line.
    Output: the distance from q to the line defined by p1 and p2.

    Notes:
    - get slope of existing line
    - take negetive inverse of that slope, this'll be the slope of the perpendicular from the point onto the line
    - calculate equation of line from point slope form
    - calculate the point of intersection of line 1 and perpendicular line
    - calculate distance between the two points

    Line in point slope form: (y − y2) = m * (x − x2)
    => y - y2 = mx - m.x2
    => mx - y + (y2 - m.x2)

    Point of intersection of 2 lines
    a1.x + b1.y + c1 = 0
    a2.x + b2.y + c2 = 0

    From eq1 => x = - (b1.y + c1) / a1
    Using in eq2 => - a2.(b1.y + c1) / a1 + b2.y + c2 = 0
    => - a2.(b1.y + c1) + a1.b2.y + a1.c2 = 0
    => - a2.b1.y - a2.c1 + a1.b2.y + a1.c2 = 0
    => y (a1.b2 - a2.b1) = a2.c1 - a1.c2
    => y = (a2.c1 - a1.c2) / (a1.b2 - a2.b1)

    Using in eq1 => x = - (b1.(a2.c1 - a1.c2) / (a1.b2 - a2.b1) + c1) / a1
    => x = - ((b1.(a2.c1 - a1.c2) + c1.(a1.b2 - a2.b1)) / a1.(a1.b2 - a2.b1)
    => x = ((b1.(a1.c2 - a2.c1) + c1.(a2.b1 - a1.b2)) / a1.(a1.b2 - a2.b1)

    Distance between 2 points
    => sqrt((x2-x1)^2 + (y2-y1)^2)

    x = 10
    => ax + by + c = 0 => a=1;b=0;c=-10
    => y = mx + c = 0 => mx - y + c = 0

    y = 10
    => ax + by + c = 0 => a=0;b=1;c=-10
    """
    # get slope of existing line
    line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(p1, p2)
    line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})
    # print(line)

    # get perpendicular line
    perpendicular_line = line.get_perpendicular_line_through_point(q)
    # print(perpendicular_line)

    # calculate the point of intersection of line 1 and perpendicular line
    intersection_point = line.get_intersection_point(perpendicular_line)
    # print(intersection_point)

    # calculate distance between the two points
    return q.get_distance(intersection_point)


def computeDistancePointToSegment(q: Point, p1: Point, p2: Point) -> Tuple[float, int]:
    """Compute distance from point to line segment.

    Input: a point q and a segment defined by two distinct points (p1, p2).
    Output: (d, w), where d is the distance from q to the segment with extreme points
        (p1, p2) and w ∈ {0, 1, 2} has the following meaning: w = 0 if the segment point
        closest to q is strictly inside the segment, w = 1 if the closest point is p1, and w = 2 if
        the closest point is p2.
    """
    line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(p1, p2)
    line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})

    perpendicular_at_p1 = line.get_perpendicular_line_through_point(p1)
    perpendicular_at_p2 = line.get_perpendicular_line_through_point(p2)

    if perpendicular_at_p1.is_point_on_line(
        q
    ) or perpendicular_at_p1.are_points_on_opposite_side_of_line(q, p2):
        return q.get_distance(p1), 1
    elif perpendicular_at_p2.is_point_on_line(
        q
    ) or perpendicular_at_p2.are_points_on_opposite_side_of_line(q, p1):
        return q.get_distance(p2), 2
    else:
        return computeDistancePointToLine(q, p1, p2), 0


def computeDistancePointToPolygon(P: Polygon, q: Point) -> float:
    """Compute distance of a point from a polygon.

    Input: a polygon P and a point q.
    Output: the distance from q to the closest point in P, called the distance from q to the polygon.
    """
    least_distance = float("inf")
    v1 = P.vertices[-1]
    for v2 in P.vertices:
        distance, _ = computeDistancePointToSegment(q, v1, v2)
        if distance < least_distance:
            least_distance = distance
        v1 = v2
    return least_distance


def computeClosestPointOnPolygon(P: Polygon, q: Point) -> Point:
    least_distance = float("inf")
    segment = []
    closest_point = None
    v1 = P.vertices[-1]
    for v2 in P.vertices:
        distance, closest_to = computeDistancePointToSegment(q, v1, v2)
        if distance < least_distance:
            least_distance = distance
            segment = [v1, v2]
            closest_point = closest_to
        v1 = v2

    if closest_point == 0:
        line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(
            segment[0], segment[1]
        )
        line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})
        perpendicular_line = line.get_perpendicular_line_through_point(q)
        intersection_point = perpendicular_line.get_intersection_point(line)
        return intersection_point
    return segment[closest_point-1]


def computeTangentVectorToPolygon(P: Polygon, q: Point) -> Vector:
    """Compute tangent vector to polygon.

    Input: a polygon P and a point q.
    Output: the unit-length vector u tangent at point q to the polygon P in the following sense:
        (i) if q is closest to a segment of the polygon, then u should be parallel to the segment,
        (ii) if q is closest to a vertex, then u should be tangent to a circle centered at the
            vertex that passes through q, and
        (iii) the tangent should lie in the counter-clockwise direction.

    y = mx
    x^2 + y^2 = 1
    => x^2 + m^2.x^2 = 1
    => x^2 (1 - m^2) = 1
    => x = sqrt(1 / (1 - m^2))
    and
    => y = m.x
    """
    least_distance = float("inf")
    segment = []
    closest_point = None
    directional_point = None
    v1 = P.vertices[-1]
    for v2 in P.vertices:
        distance, closest_to = computeDistancePointToSegment(q, v1, v2)
        if distance < least_distance:
            least_distance = distance
            segment = [v1, v2]
            closest_point = closest_to
        v1 = v2

    if closest_point == 0:
        # print(f"Closest to segment: [{segment[0]}, {segment[1]}]")
        line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(
            segment[0], segment[1]
        )
        line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})
        required_line = line.get_parallel_line_through_point(q)
        closest_point_on_polygon = line.get_intersection_point(
            required_line.get_perpendicular_line_through_point(q)
        )
        directional_point = P.get_next_vertex_on_polygon(
            segment=[segment[0], segment[1]]
        )
    elif closest_point == 1:
        # print(f"Closest to vertex: [{segment[0]}]")
        if segment[0] == q:
            raise ValueError(f"The point lies on the vertex {segment[0]}.")
        line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(
            segment[0], q
        )
        line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})
        required_line = line.get_perpendicular_line_through_point(q)
        closest_point_on_polygon = segment[0]
        directional_point = P.get_next_vertex_on_polygon(vertex=segment[0])
    elif closest_point == 2:
        # print(f"Closest to vertex: [{segment[1]}]")
        if segment[1] == q:
            raise ValueError(f"The point lies on the vertex {segment[0]}.")
        line_param_a, line_param_b, line_param_c = computeLineThroughTwoPoints(
            segment[1], q
        )
        line = Line(**{"normalized-form": (line_param_a, line_param_b, line_param_c)})
        required_line = line.get_perpendicular_line_through_point(q)
        closest_point_on_polygon = segment[1]
        directional_point = P.get_next_vertex_on_polygon(vertex=segment[1])

    # print(f"Required Line: {required_line}")
    p1, p2 = required_line.get_two_points_on_either_side_of_given_point_on_line(q)
    # print(f"Directinal Point: {directional_point}")
    p1_vector = Vector(
        p1.x - closest_point_on_polygon.x, p1.y - closest_point_on_polygon.y
    )
    p2_vector = Vector(
        p2.x - closest_point_on_polygon.x, p2.y - closest_point_on_polygon.y
    )
    q_vector = Vector(
        q.x - closest_point_on_polygon.x, q.y - closest_point_on_polygon.y
    )

    q_cross_p1 = p1_vector.x * q_vector.y - p1_vector.y * q_vector.x
    q_cross_p2 = p2_vector.x * q_vector.y - p2_vector.y * q_vector.x
    # print(f"p1_vector: {p1_vector}")
    # print(f"p2_vector: {p2_vector}")
    # print(f"q_vector: {q_vector}")
    # print(f"q.x: {q.x}")
    # print(f"q.y: {q.y}")
    # print(f"closest_point_on_polygon: {closest_point_on_polygon}")
    # print(f"q_cross_p1 = {p1_vector.x} * {q_vector.y} - {p1_vector.y} * {q.x} = {q_cross_p1}")
    # print(f"q_cross_p2 = {p2_vector.x} * {q_vector.y} - {p2_vector.y} * {q.x} = {q_cross_p2}")
    # print()

    if q_cross_p1 > 0:
        required_point_on_line = p2
    else:
        required_point_on_line = p1
    # print(f"Req Point on line: {required_point_on_line}")

    x_sign = -1 if required_point_on_line.x - q.x < 0 else 1
    y_sign = -1 if required_point_on_line.y - q.y < 0 else 1
    # print(f"x_sign, y_sign: {x_sign}, {y_sign} ")

    vector_line = required_line.get_parallel_line_through_point(Point(0, 0))
    # print(f"Vector line: {vector_line}")
    # print()

    return Vector(x_sign * abs(vector_line.b), y_sign * abs(vector_line.a))


def BugBase(
    start: Point, goal: Point, obstaclesList: Tuple[Polygon, ...], step_size: float
) -> Tuple[str, List[Point]]:
    """BugBase Algorithm.

    Input: Two locations start and goal in Wfree, a list of polygonal obstacles
        obstaclesList, and a length step-size

    Output: A sequence, denoted path, of points from start to the first obstacle
        between start and goal (or from start to goal if no obstacle lies between them).
        Successive points are separated by no more than step-size.

    Getting new current position based on step size
    Ref: https://math.stackexchange.com/a/1630886
    """
    current_position = start
    path = [start]
    while current_position.get_distance(goal) > step_size:
        closest_polygon_distance = float("inf")
        for obstacle in obstaclesList:
            d = computeDistancePointToPolygon(obstacle, current_position)
            if d < closest_polygon_distance:
                closest_polygon_distance = d
        if closest_polygon_distance < step_size:
            return (
                "Failure: There is an obstacle lying between the start and goal",
                path,
            )

        t = step_size / current_position.get_distance(goal)
        current_position = Point(
            ((1 - t) * current_position.x + t * goal.x),
            ((1 - t) * current_position.y + t * goal.y),
        )

        path.append(current_position)
    path.append(goal)
    return "Success", path


def computeBug1(
    start: Point, goal: Point, obstaclesList: Tuple[Polygon, ...], step_size: float
) -> Tuple[str, List[Point]]:
    """Bugone Algorithm.

    Input: Two locations start and goal in Wfree, a list of polygonal obstacles
        obstaclesList, and a length step-size

    Output: A sequence, denoted path, of points from start to goal or returns an
        error message if such a path does not exists. Successive points are separated by
        no more than step-size and are computed according to the Bug 1 algorithm.
    """
    current_position = start
    path = []
    while current_position.get_distance(goal) > step_size:
        success, subpath = BugBase(current_position, goal, obstaclesList, step_size)
        path.extend(subpath)
        current_position = subpath[-1]

        starting_position_for_circumventing_obstacle = current_position
        if "Failure" in success:
            closest_polygon = None
            closest_polygon_distance = float("inf")

            for obstacle in obstaclesList:
                distance = computeDistancePointToPolygon(obstacle, current_position)
                if distance < closest_polygon_distance:
                    closest_polygon_distance = distance
                    closest_polygon = obstacle

            obstacle_circumvention_path = []
            circumvention_path_nearest_point_to_goal = None
            circumvention_path_nearest_point_to_goal_distance = float("inf")
            starting_closest_point_on_polygon = computeClosestPointOnPolygon(closest_polygon, current_position)

            while True:
                direction_of_motion = computeTangentVectorToPolygon(
                    closest_polygon, current_position
                )
                # print(f"{current_position},")  # ; Dir: {direction_of_motion}")
                current_position = current_position.move_magnitude_towards(
                    direction_of_motion, step_size
                )
                current_closest_point_on_polygon = computeClosestPointOnPolygon(closest_polygon, current_position)
                obstacle_circumvention_path.append(current_position)
                distance = current_position.get_distance(goal)
                if distance < circumvention_path_nearest_point_to_goal_distance:
                    circumvention_path_nearest_point_to_goal_distance = distance
                    circumvention_path_nearest_point_to_goal = current_position

                if abs(current_closest_point_on_polygon.x - starting_closest_point_on_polygon.x) < step_size / 2 and abs(current_closest_point_on_polygon.y - starting_closest_point_on_polygon.y) < step_size / 2:
                    break

            path.extend(obstacle_circumvention_path)
            index = obstacle_circumvention_path.index(
                circumvention_path_nearest_point_to_goal
            )
            path.extend(obstacle_circumvention_path[: index + 1])
            current_position = path[-1]

    return path


"""Custom Classes for Kinematics."""

# standard imports




