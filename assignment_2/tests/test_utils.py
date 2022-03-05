#!/usr/bin/env python3

"Test Driven Development."

# standard imports
from math import sqrt, hypot

# third-party imports
import pytest

# app imports
from helper import Point, Polygon, Vector
from helper import (
    computeLineThroughTwoPoints,
    computeDistancePointToLine,
    computeDistancePointToSegment,
    computeDistancePointToPolygon,
    computeTangentVectorToPolygon,
    BugBase,
    computeBug1,
)


def test_computeLineThroughTwoPoints():
    # test for x-axis
    p1 = Point(5, 0)
    p2 = Point(-5, 0)
    assert computeLineThroughTwoPoints(p1, p2) == (0, 1, 0)

    # test for y-axis
    p1 = Point(0, 5)
    p2 = Point(0, -5)
    assert computeLineThroughTwoPoints(p1, p2) == (1, 0, 0)

    # test for x=y
    p1 = Point(5, 5)
    p2 = Point(-5, -5)
    assert computeLineThroughTwoPoints(p1, p2) == (1 / sqrt(2), -1 / sqrt(2), 0)


def test_computeDistancePointToLine():
    q = Point(0, 10)
    p1 = Point(5, 0)
    p2 = Point(-5, 0)
    assert computeDistancePointToLine(q, p1, p2) == 10

    q = Point(-10, 0)
    p1 = Point(0, 5)
    p2 = Point(0, -5)
    assert computeDistancePointToLine(q, p1, p2) == 10

    q = Point(-5, 5)
    p1 = Point(5, 5)
    p2 = Point(-5, -5)
    assert computeDistancePointToLine(q, p1, p2) == hypot(5, 5)

    q = Point(5, -5)
    p1 = Point(5, 5)
    p2 = Point(-5, -5)
    assert computeDistancePointToLine(q, p1, p2) == hypot(5, 5)

    q = Point(1.1, -0.1)
    p1 = Point(3, 0)
    p2 = Point(1, 0)
    assert computeDistancePointToLine(q, p1, p2) == 0.1

def test_computeDistancePointToSegment():
    p1 = Point(5, 0)
    p2 = Point(-5, 0)

    q = Point(0, 10)
    assert computeDistancePointToSegment(q, p1, p2) == (10, 0)
    q = Point(5, 10)
    assert computeDistancePointToSegment(q, p1, p2) == (10, 1)
    q = Point(10, 12)
    assert computeDistancePointToSegment(q, p1, p2) == (13, 1)
    q = Point(-10, -12)
    assert computeDistancePointToSegment(q, p1, p2) == (13, 2)
    q = Point(0, 0)
    assert computeDistancePointToSegment(q, p1, p2) == (0, 0)
    q = Point(10, 0)
    assert computeDistancePointToSegment(q, p1, p2) == (5, 1)
    q = Point(-10, 0)
    assert computeDistancePointToSegment(q, p1, p2) == (5, 2)

    p1 = Point(0, 5)
    p2 = Point(0, -5)

    q = Point(10, 0)
    assert computeDistancePointToSegment(q, p1, p2) == (10, 0)
    q = Point(10, 5)
    assert computeDistancePointToSegment(q, p1, p2) == (10, 1)
    q = Point(12, 10)
    assert computeDistancePointToSegment(q, p1, p2) == (13, 1)
    q = Point(-12, -10)
    assert computeDistancePointToSegment(q, p1, p2) == (13, 2)
    q = Point(0, 0)
    assert computeDistancePointToSegment(q, p1, p2) == (0, 0)
    q = Point(0, 10)
    assert computeDistancePointToSegment(q, p1, p2) == (5, 1)
    q = Point(0, -10)
    assert computeDistancePointToSegment(q, p1, p2) == (5, 2)

    p1 = Point(3, 0)
    p2 = Point(1, 0)

    q = Point(1.1, -0.1)
    assert computeDistancePointToSegment(q, p1, p2) == (0.1, 0)

def test_computeDistancePointToPolygon():
    p1 = Point(5, 5)
    p2 = Point(-5, 5)
    p3 = Point(-5, -5)
    p4 = Point(5, -5)
    P = Polygon((p1, p2, p3, p4))

    q = Point(10, 17)
    assert computeDistancePointToPolygon(P, q) == 13
    q = Point(5, 0)
    assert computeDistancePointToPolygon(P, q) == 0
    q = Point(10, 0)
    assert computeDistancePointToPolygon(P, q) == 5


def test_compute_move_magnitude_towards():
    p = Point(0, 0)
    magnitude = 1
    one_by_root_two = 1 / sqrt(2)

    v = Vector(1, 1)
    assert p.move_magnitude_towards(v, magnitude) == Point(one_by_root_two, one_by_root_two)
    v = Vector(-1, 1)
    assert p.move_magnitude_towards(v, magnitude) == Point(-one_by_root_two, one_by_root_two)
    v = Vector(-1, -1)
    assert p.move_magnitude_towards(v, magnitude) == Point(-one_by_root_two, -one_by_root_two)
    v = Vector(1, -1)
    assert p.move_magnitude_towards(v, magnitude) == Point(one_by_root_two, -one_by_root_two)
    v = Vector(-3/sqrt(10), 1/sqrt(10))
    magnitude = sqrt(10)
    assert p.move_magnitude_towards(v, magnitude) == Point(-3, 1)


def test_computeTangentVectorToPolygon():
    p1 = Point(5, 5)
    p2 = Point(-5, 5)
    p3 = Point(-5, -5)
    p4 = Point(5, -5)
    P = Polygon((p1, p2, p3, p4))

    q = Point(10, 0)
    assert computeTangentVectorToPolygon(P, q) == Vector(0, 1)
    q = Point(-10, 0)
    assert computeTangentVectorToPolygon(P, q) == Vector(0, -1)
    q = Point(-10, -10)
    assert computeTangentVectorToPolygon(P, q) == Vector(1 / sqrt(2), -1 / sqrt(2))
    q = Point(10, 10)
    assert computeTangentVectorToPolygon(P, q) == Vector(-1 / sqrt(2), 1 / sqrt(2))
    q = Point(5, 0)
    assert computeTangentVectorToPolygon(P, q) == Vector(0, 1)
    with pytest.raises(Exception):
        q = Point(5, 5)
        assert computeTangentVectorToPolygon(P, q) == Vector(-1 / sqrt(2), 1 / sqrt(2))

    p1 = Point(3, 0)
    p2 = Point(1, 2)
    p3 = Point(1, 0)
    P = Polygon((p1, p2, p3))

    q = Point(1.1, -0.1)
    assert computeTangentVectorToPolygon(P, q) == Vector(1, 0)

    q = Point(3.09, -0.12)
    v = computeTangentVectorToPolygon(P, q)
    assert v.x > 0
    assert v.y > 0

def test_BugBase():
    start = Point(0, 0)
    goal = Point(5, 3)
    obstaclesList = [
        Polygon((Point(1, 2), Point(1, 0), Point(3, 0))),
        Polygon((Point(2, 3), Point(4, 1), Point(5, 2))),
    ]
    step_size = 0.1
    assert BugBase(start, goal, obstaclesList, step_size) == (
        "Failure: There is an obstacle lying between the start and goal",
        [
            Point(0, 0),
            Point(0.08574929257125442, 0.05144957554275266),
            Point(0.17149858514250882, 0.10289915108550532),
            Point(0.25724787771376323, 0.15434872662825797),
            Point(0.34299717028501764, 0.20579830217101064),
            Point(0.42874646285627205, 0.2572478777137633),
            Point(0.5144957554275265, 0.30869745325651593),
            Point(0.6002450479987809, 0.3601470287992686),
            Point(0.6859943405700353, 0.41159660434202117),
            Point(0.7717436331412897, 0.4630461798847738),
            Point(0.8574929257125441, 0.5144957554275265),
            Point(0.9432422182837985, 0.5659453309702791),
        ],
    )

    start = Point(0, 0)
    goal = Point(5, 12)
    obstaclesList = []
    step_size = 1
    assert BugBase(start, goal, obstaclesList, step_size) == (
        "Success",
        [
            Point(0, 0),
            Point(0.38461538461538464, 0.9230769230769231),
            Point(0.7692307692307694, 1.8461538461538463),
            Point(1.153846153846154, 2.769230769230769),
            Point(1.5384615384615388, 3.6923076923076925),
            Point(1.9230769230769234, 4.615384615384615),
            Point(2.307692307692308, 5.538461538461538),
            Point(2.6923076923076925, 6.461538461538462),
            Point(3.076923076923077, 7.384615384615384),
            Point(3.4615384615384617, 8.307692307692308),
            Point(3.8461538461538467, 9.230769230769234),
            Point(4.230769230769232, 10.153846153846157),
            Point(4.615384615384617, 11.07692307692308),
            Point(5, 12),
        ],
    )


# def test_computeBug1():
#     start = Point(0, 0)
#     goal = Point(5, 3)
#     obstaclesList = [
#         Polygon((Point(1, 2), Point(1, 0), Point(3, 0))),
#         Polygon((Point(2, 3), Point(4, 1), Point(5, 2))),
#     ]
#     step_size = 0.1
#     print(computeBug1(start, goal, obstaclesList, step_size))
