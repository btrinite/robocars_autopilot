/**
 * @file offb_raw_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo SITL
 source $src_path/Tools/setup_gazebo.bash ${src_path} ${build_path}

 gzserver --verbose ${src_path}/Tools/sitl_gazebo/worlds/${model}.world &
 */
#include <tinyfsm.hpp>
#include <ros/ros.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#include <boost/format.hpp>

#include <date.h>
#include <json.hpp>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_transport/image_transport.h>


#include <robocars_msgs/robocars_brain_state.h>
#include <robocars_msgs/robocars_tof.h>
#include <robocars_msgs/robocars_autopilot_output.h>

#include <robocars_autopilot.hpp>

RosInterface * ri;

static int loop_hz;
static std::string model_filename;

class onRunningMode;
class onIdle;
class onManualDriving;
class onAutonomousDriving;

class onRunningMode
: public RobocarsStateMachine
{
    public:
        onRunningMode() : RobocarsStateMachine("onRunningMode"),__tick_count(0) {};
        onRunningMode(const char * subStateName) : RobocarsStateMachine(subStateName),__tick_count(0) {};


    protected:

        uint32_t __tick_count;
        
        void entry(void) override {
            RobocarsStateMachine::entry();
        };

        void react(ManualDrivingEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react( AutonomousDrivingEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react( PredictEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react( IdleStatusEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react(TickEvent const & e) override {
        };

};

class onIdle
: public onRunningMode
{
    public:
        onIdle() : onRunningMode("onArm") {};

    private:

        void entry(void) override {
            onRunningMode::entry();
        };
  
        void react(ManualDrivingEvent const & e) override { 
            onRunningMode::react(e);
            transit<onManualDriving>();
        };


        void react(TickEvent const & e) override {
            onRunningMode::react(e);
        };

};

class onManualDriving
: public onRunningMode
{
    public:
        onManualDriving() : onRunningMode("onManualDriving") {};

    private:

        void entry(void) override {
            onRunningMode::entry();
        };

        void react (AutonomousDrivingEvent const & e) override {
            onRunningMode::react(e);
            transit<onAutonomousDriving>();
        }

        void react(IdleStatusEvent const & e) override { 
            onRunningMode::react(e);
            transit<onIdle>();
        };

        void react (TickEvent const & e) override {
            onRunningMode::react(e);
        };

};

class onAutonomousDriving
: public onRunningMode
{
    public:
        onAutonomousDriving() : onRunningMode("onAutonomousDriving") {};

    private:

        virtual void entry(void) { 
            onRunningMode::entry();
        };  

        virtual void react(IdleStatusEvent                 const & e) override { 
            onRunningMode::react(e);
            transit<onIdle>();
        };

        virtual void react(ManualDrivingEvent              const & e) override { 
            onRunningMode::react(e);
            transit<onManualDriving>();
        };

        void react( PredictEvent const & e) override { 
            ri->publishPredict(e.steering_value, e.throttling_value);
            onRunningMode::react(e);
        };

        virtual void react(TickEvent                      const & e) override { 
            onRunningMode::react(e);
        };
};

FSM_INITIAL_STATE(RobocarsStateMachine, onIdle)

uint32_t mapRange(uint32_t in1,uint32_t in2,uint32_t out1,uint32_t out2,uint32_t value)
{
  if (value<in1) {value=in1;}
  if (value>in2) {value=in2;}
  return out1 + ((value-in1)*(out2-out1))/(in2-in1);
}

void RosInterface::predict() {
}

void RosInterface::publishPredict(_Float32 steering, _Float32 throttling) {
    robocars_msgs::robocars_autopilot_output steeringMsg;
    robocars_msgs::robocars_autopilot_output throttlingMsg;

    steeringMsg.header.stamp = ros::Time::now();
    steeringMsg.header.seq=1;
    steeringMsg.header.frame_id = "pilotSteering";
    steeringMsg.norm = steering;

    autopilot_steering_pub.publish(steeringMsg);

    throttlingMsg.header.stamp = ros::Time::now();
    throttlingMsg.header.seq=1;
    throttlingMsg.header.frame_id = "pilotSteering";
    throttlingMsg.norm = steering;

    autopilot_throttling_pub.publish(throttlingMsg);
}

void RosInterface::initParam() {
    if (!node_.hasParam("loop_hz")) {
        node_.setParam ("loop_hz", 60);       
    }
    if (!node_.hasParam("model_filename")) {
        node_.setParam("model_filename",  std::string("modelcat"));
    }
}
void RosInterface::updateParam() {
    node_.getParam("loop_hz", loop_hz);
    node_.getParam("model_filename", model_filename);
}


void RosInterface::initSub () {
    sub_image_and_camera = it->subscribeCamera("/front_video_resize/image", 1, &RosInterface::callbackWithCameraInfo, this);
    tof1_sub = node_.subscribe<robocars_msgs::robocars_tof>("/sensors/tof1", 1, &RosInterface::tof1_msg_cb, this);
    tof2_sub = node_.subscribe<robocars_msgs::robocars_tof>("/sensors/tof2", 1, &RosInterface::tof2_msg_cb, this);
    state_sub = node_.subscribe<robocars_msgs::robocars_brain_state>("/robocars_brain_state", 1, &RosInterface::state_msg_cb, this);
}
void RosInterface::initPub() {
        autopilot_steering_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/steering", 10);
        autopilot_throttling_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/throttling", 10);
}

static uint32_t lastTof1Value;
static uint32_t lastTof2Value;

void RosInterface::callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info) {
    static _Float32 fake_steering_value = -1.0;
    send_event(PredictEvent(fake_steering_value,0.0));
    fake_steering_value = fake_steering_value + 0.1;
    if (fake_steering_value>1.1) {fake_steering_value = -1.0;}
}

void RosInterface::tof1_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg){
    lastTof1Value = msg->distance;
}

void RosInterface::tof2_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg){
    lastTof2Value = msg->distance;
}

void RosInterface::state_msg_cb(const robocars_msgs::robocars_brain_state::ConstPtr& msg) {    
    static u_int32_t last_state = -1;
    if (msg->state != last_state) {
        switch (msg->state) {
            case robocars_msgs::robocars_brain_state::BRAIN_STATE_IDLE:
                send_event(IdleStatusEvent());        
            break;
            case robocars_msgs::robocars_brain_state::BRAIN_STATE_MANUAL_DRIVING:
                send_event(ManualDrivingEvent());        
            break;
            case robocars_msgs::robocars_brain_state::BRAIN_STATE_AUTONOMOUS_DRIVING:
                send_event(AutonomousDrivingEvent());        
            break;
        }
        last_state=msg->state;
    }    
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "robocars_autopilot");
    
    ri = new RosInterface;

    ri->initPub();
    fsm_list::start();
    ri->initSub();

    ROS_INFO("Autopilot: Starting");

    // wait for FCU connection
    ros::Rate rate(loop_hz);
    while(ros::ok()){
        ros::spinOnce();
        send_event (TickEvent());
        rate.sleep();
    }
}

