/**
 * @file robocars_autopilot.hpp
 * @brief When model loaded, perform inference on image received, and publish topic with predcited steering. Speed is a stupid constant for now
 * 
 * Copyright (c) 2020 Benoit TRINITE
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * 
 **/

#include <tinyfsm.hpp>
#include <ros/ros.h>
#include <stdio.h>


struct BaseEvent : tinyfsm::Event
{
    public:
        BaseEvent(const char * evtName) : _evtName(evtName) {};
        const char * getEvtName() const { return _evtName; };
    private:
        const char *  _evtName;
};

struct TickEvent                    : BaseEvent { public: TickEvent() : BaseEvent("TickEvent") {}; };
struct IdleStatusEvent              : BaseEvent { public: IdleStatusEvent() : BaseEvent("IdleStatusEvent") {}; };
struct ManualDrivingEvent           : BaseEvent { public: ManualDrivingEvent() : BaseEvent("ManualDrivingEvent") {}; };
struct AutonomousDrivingEvent       : BaseEvent { public: AutonomousDrivingEvent() : BaseEvent("AutonomousDrivingEvent") {}; };
struct PredictEvent                 : BaseEvent { public: 
    PredictEvent(const _Float32 steeringValue, const _Float32 throttlingValue, const _Float32 brakingValue, const __uint32_t seqNum, const __uint32_t carId) : steering_value(steeringValue), throttling_value(throttlingValue), braking_value(brakingValue), seq_num(seqNum), carId(carId),  BaseEvent("PredictEvent") {};
    _Float32 steering_value; 
    _Float32 throttling_value; 
    _Float32 braking_value; 
    __uint32_t seq_num;
    __uint32_t carId;
    };

class RobocarsStateMachine
: public tinyfsm::Fsm<RobocarsStateMachine>
{
    public:
        RobocarsStateMachine(const char * stateName) : _stateName(stateName), tinyfsm::Fsm<RobocarsStateMachine>::Fsm() { 
            ROS_INFO("Autopilot StateMachine: State created: %s", _stateName);      
        };
        const char * getStateName() const { return _stateName; };

    public:
        /* default reaction for unhandled events */
        void react(BaseEvent const & ev) { 
            ROS_INFO("state %s: unexpected event %s reveived", getStateName(), ev.getEvtName());      
        };

        virtual void react(TickEvent                      const & e) { /*logEvent(e);*/ };
        virtual void react(IdleStatusEvent                const & e) { logEvent(e); };
        virtual void react(ManualDrivingEvent             const & e) { logEvent(e); };
        virtual void react(AutonomousDrivingEvent         const & e) { logEvent(e); };
        virtual void react(PredictEvent                   const & e) {  };

        virtual void entry(void) { 
            ROS_INFO("State %s: entering", getStateName()); 
        };  
        void         exit(void)  { };  /* no exit actions */

    private:
        const char *  _stateName ="NoName";
        void logEvent(BaseEvent const & e) {
            ROS_INFO("State %s: event %s", getStateName(), e.getEvtName());
        }
};

typedef tinyfsm::FsmList<RobocarsStateMachine> fsm_list;

template<typename E>
void send_event(E const & event)
{
  fsm_list::template dispatch<E>(event);
}



class RosInterface
{
    public :
        RosInterface() : node_("~") {
            initParam();
            updateParam();
            it = new image_transport::ImageTransport(node_);
        };


        void initParam();
        void updateParam();
        void initSub();
        void initPub();

        void publishPredict(_Float32 steering, _Float32 throttling, _Float32 braking, __uint32_t seq, __uint32_t carId);

        void initStats();
        void reportStats();
        bool updateStats(uint32_t received, uint32_t missed);

    private:
        void state_msg_cb(const robocars_msgs::robocars_brain_state::ConstPtr& msg);

        ros::NodeHandle node_;
        ros::Subscriber state_sub;
        ros::Publisher autopilot_throttling_pub;
        ros::Publisher autopilot_steering_pub;
        ros::Publisher autopilot_braking_pub;


        void callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info);
        void callbackNoCameraInfo(const sensor_msgs::ImageConstPtr& image_msg);
        void mark_msg_cb(const robocars_msgs::robocars_mark::ConstPtr& msg);
        void telem_msg_cb(const robocars_msgs::robocars_telemetry::ConstPtr& msg);
        bool reloadModel_cb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);

        template <class T> void resize (T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels);
        template <class T> float unbind(T* prediction, int prediction_size, size_t num_results,
               TfLiteType input_type);

        ros::Subscriber mark_sub;
        ros::Subscriber telem_sub;
        image_transport::ImageTransport * it;
        image_transport::CameraSubscriber sub_image_and_camera;
        image_transport::Subscriber sub_image;

        ros::ServiceServer reloadModel_svc;
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
        int input_img;
        int input_telem_speed;
        int output_steering;
        int output_throttling;
        int output_mark;
        int wanted_height;
        int wanted_width;
        int wanted_channels;
        TfLiteType model_input_img_type;
        TfLiteType model_input_telem_speed_type;
        TfLiteType model_output_steering_type;
        TfLiteType model_output_throttling_type;
        TfLiteType model_output_mark_type;
        int output_steering_size;
        int output_throttling_size;
        int output_mark_size;

        bool modelLoaded=false;

        // stats
        uint32_t totalImages=0;
        uint32_t missedImages=0;
        ros::Publisher stats_pub;

};

