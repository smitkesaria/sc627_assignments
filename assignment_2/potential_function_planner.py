#!/usr/bin/env python3

import os
import rospy
from geometry_msgs.msg import Twist
from sc627_helper.msg import MoveXYActionGoal, MoveXYActionResult

from helper import Polygon, Point, computePotentialPlanner

class RosPotentialPlanner:
    def __init__(self):
        rospy.init_node("bug_one")
        self.cwd = os.path.dirname(__file__)
        with open(f"{self.cwd}/input.txt") as file:
            obtained_start = False
            obtained_goal = False
            obtained_stepsize = False

            obstaclesList = []
            obstacle_acc = []
            for line in file.readlines():
                if not obtained_start:
                    x, y = line.split(",")
                    start = Point(int(x), int(y))
                    obtained_start = True
                    continue
                if not obtained_goal:
                    x, y = line.split(",")
                    goal = Point(int(x), int(y))
                    obtained_goal = True
                    continue
                if not obtained_stepsize:
                    step_size = float(line)
                    obtained_stepsize = True
                    continue
                if line == "\n":
                    if obstacle_acc:
                        obstaclesList.append(Polygon(obstacle_acc))
                    obstacle_acc = []
                    continue
                x, y = line.split(",")
                obstacle_acc.append(Point(int(x), int(y)))


            self.path = computePotentialPlanner(start, goal, obstaclesList, step_size)

        self.mxyar = MoveXYActionResult()
        self.cmd_pub = rospy.Publisher('/move_xy/goal', MoveXYActionGoal, queue_size=1)
        self.cmd_sub = rospy.Subscriber('/move_xy/result', MoveXYActionResult, self.feedback_cb)

    def feedback_cb(self, msg):
        self.mxyar = msg

    def move_turtlebot(self):
        pub_msg = MoveXYActionGoal()
        filename = "output.txt"
        output_file = open(f"{self.cwd}/{filename}", "w")
        for point in self.path:
            pub_msg.goal.pose_dest.x = point.x
            pub_msg.goal.pose_dest.y = point.y
            while not rospy.is_shutdown() and self.mxyar.status.status != 3:
                self.cmd_pub.publish(pub_msg)
                rospy.sleep(1)
            self.mxyar = MoveXYActionResult()
            output_file.write(f"{point.x},{point.y}\n")
            print(f"Reached {point}")
        output_file.close()
        print("PotentialPlanner Completed")


if __name__ == "__main__":
    server = RosPotentialPlanner()
    server.move_turtlebot()