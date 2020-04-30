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
    PredictEvent(const _Float32 steeringValue, const _Float32 throttlingValue) : steering_value(steeringValue), throttling_value(throttlingValue), BaseEvent("PredictEvent") {};
    _Float32 steering_value; 
    _Float32 throttling_value; 
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

        void predict();
        void publishPredict(_Float32 steering, _Float32 throttling);

        void initStats();
        void reportStats();
        bool updateStats(uint32_t received, uint32_t missed);

    private:
        void state_msg_cb(const robocars_msgs::robocars_brain_state::ConstPtr& msg);

        ros::NodeHandle node_;
        ros::Subscriber state_sub;
        ros::Publisher autopilot_throttling_pub;
        ros::Publisher autopilot_steering_pub;


        void callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info);
        void tof1_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg);
        void tof2_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg);
        bool reloadModel_cb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
        
        template <class T> void resize (T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels);
        template <class T> float unbind(T* prediction, int prediction_size, size_t num_results,
               TfLiteType input_type);
        ros::Subscriber tof1_sub;
        ros::Subscriber tof2_sub;
        image_transport::ImageTransport * it;
        image_transport::CameraSubscriber sub_image_and_camera;

        ros::ServiceServer reloadModel_svc;
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        int input;
        int output_steering;
        int output_throttling;
        int wanted_height;
        int wanted_width;
        int wanted_channels;
        TfLiteType model_input_type;
        int output_steering_size;
        int output_throttling_size;

        bool modelLoaded=false;

        // stats
        uint32_t totalImages=0;
        uint32_t missedImages=0;
        ros::Publisher stats_pub;

};

