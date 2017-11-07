#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
# All Rights Reserved.

# Author: Brandon Kinman

import rospy
import dynamic_reconfigure.client
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Wrench
from quad_controller.twiddle import Twiddle
from quad_controller.srv import SetPose
from quad_controller.srv import SetInt


class HoverTestRun:
    def __init__(self):
        self.is_started_ = False
        self.test_data_ = []
        self.oscillation_count_ = 0
        self.max_duration_ = 10
        self.start_time_ = 0
        self.duration_ = 0
    
    def addPose(self ,pose_stamped):
        if self.is_started_ is False:
            return
        if self.start_time_ is 0:
            self.start_time_ = pose_stamped.header.stamp 
            return
        self.test_data_.append((pose_stamped.header.stamp, pose_stamped.pose.position.z))

    def computeTotalError(self, set_point):
        total_error = 0
        for item in self.test_data_:
            total_error += abs(set_point - item[1])
        return total_error


    def isFinished(self):
        if self.is_started_ is False or len(self.test_data_) is 0:
            return False

        # Test is finished if we've reached max duration
        duration = self.test_data_[-1][0] - self.start_time_
        if duration.to_sec() > self.max_duration_:
            self.is_started_ = False
            return True
        else:
            return False

        # Test if finished it the number of oscillations are 
    
    def reset(self):
        self.__init__()
    
    def start(self):
        self.is_started_ = True


def initial_setup():
    rospy.wait_for_service('/quad_rotor/reset_orientation')
    rospy.wait_for_service('/quad_rotor/set_pose')
    rospy.wait_for_service('/quad_rotor/x_force_constrained')
    rospy.wait_for_service('/quad_rotor/y_force_constrained')
    rospy.wait_for_service('/quad_rotor/x_torque_constrained')
    rospy.wait_for_service('/quad_rotor/y_torque_constrained')
    rospy.wait_for_service('/quad_rotor/z_torque_constrained')
    rospy.wait_for_service('/quad_rotor/camera_pose_type')

    try:
        # Reset the forces and velocities on the quad
        reset_force_vel = rospy.ServiceProxy('/quad_rotor/reset_orientation', SetBool)
        reset_force_vel(True)
        
        # Call service to set position
        set_position = rospy.ServiceProxy('/quad_rotor/set_pose', SetPose)
        initial_pose = Pose()
        initial_pose.position.z = 10
        response = set_position(initial_pose)

        # Call service to constrain translations. We only want to be able to translate in Z
        x_force_constrained = rospy.ServiceProxy('/quad_rotor/x_force_constrained', SetBool)
        y_force_constrained = rospy.ServiceProxy('/quad_rotor/y_force_constrained', SetBool)
        x_force_constrained(True)
        y_force_constrained(True)

        # Call service to constrian rotations, we don't want to rotate at All
        x_torque_constrained = rospy.ServiceProxy('/quad_rotor/x_torque_constrained', SetBool)
        y_torque_constrained = rospy.ServiceProxy('/quad_rotor/y_torque_constrained', SetBool)
        z_torque_constrained = rospy.ServiceProxy('/quad_rotor/z_torque_constrained', SetBool)
        x_torque_constrained(True) 
        y_torque_constrained(True) 
        z_torque_constrained(True) 

        #Set camera pose to be aligned to X-axis (1)
        camera_pose_type = rospy.ServiceProxy('/quad_rotor/camera_pose_type', SetInt)
        camera_pose_type(1)
    except rospy.ServiceException, e:
        rospy.logerr('Service call failed: {}'.format(e))

def setup_test():
        # Reset the forces and velocities on the quad
        reset_force_vel = rospy.ServiceProxy('/quad_rotor/reset_orientation', SetBool)
        reset_force_vel(True)
        
        # Call service to set position
        set_position = rospy.ServiceProxy('/quad_rotor/set_pose', SetPose)
        initial_pose = Pose()
        initial_pose.position.z = 10
        response = set_position(initial_pose)

def algorithm(pid_params):
    setup_test()
    # Make the movement happen, stopping when some criteria has been met
    dyn_reconf_client.update_configuration({'target': 10.0,
                                            'kp': pid_params[0],
                                            'ki': pid_params[1],
                                            'kd': pid_params[2]})
    curr_test_run.reset()
    curr_test_run.start()

    while not curr_test_run.isFinished():
        rate.sleep()
    
    error = curr_test_run.computeTotalError(10.0)

    # return the error, here lower error is better.
    return error

def pose_callback(pose):
    curr_test_run.addPose(pose)

def dyn_reconf_callback(config):
    # rospy.loginfo("""Target:{target}, kp:{kp}, ki:{ki}, kd:{kd}""".format(**config))
    pass

def main():
    initial_params = [0.0, 0.0, 0.0]
    twiddle_hover = Twiddle(algorithm, initial_params)

    print("Initial Params: kp:{}, ki:{}, kd:{}".format(initial_params[0], initial_params[1], initial_params[2]))

    #while forever, just run the twiddle and print out the results
    while not rospy.is_shutdown():
        (iterations, params, best_run) = twiddle_hover.run()
        print('iterations: {} params: {} best_run: {}'.format(iterations, params, best_run))

if __name__ == '__main__':
    curr_test_run = HoverTestRun()
    curr_test_run.start()
    rospy.init_node('hover_twiddle_tuner')
    rate = rospy.Rate(10)

    pose_sub_ = rospy.Subscriber("/quad_rotor/pose", PoseStamped, pose_callback)
    cmd_force_pub_ = rospy.Publisher("/quad_rotor/cmd_force", Wrench, queue_size=10)
    dyn_reconf_client = dynamic_reconfigure.client.Client('/hover_controller', config_callback=dyn_reconf_callback)

    initial_setup()

    try:
        main()
    except rospy.ROSInterruptException: 
        pass
