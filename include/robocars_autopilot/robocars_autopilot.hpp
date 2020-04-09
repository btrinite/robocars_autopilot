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
            tensorflow::SessionOptions options = tensorflow::SessionOptions();
            options.config.mutable_gpu_options()->set_allow_growth(true);
            tensorflow::Status status = tensorflow::NewSession(options, &session);
            if (!status.ok()) {

            } else {
                ROS_INFO("Autopilot: TF Session initialized");
            }
        };


        void initParam();
        void updateParam();
        void initSub();
        void initPub();

        void predict();
        void publishPredict(_Float32 steering, _Float32 throttling);

        void initStats();
        void reportStats();
        boolean updateStats(uint32_t received, uint32_t missed);
    private:
        void state_msg_cb(const robocars_msgs::robocars_brain_state::ConstPtr& msg);

        ros::NodeHandle node_;
        ros::Subscriber state_sub;
        ros::Publisher autopilot_throttling_pub;
        ros::Publisher autopilot_steering_pub;


        void callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info);
        void tof1_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg);
        void tof2_msg_cb(const robocars_msgs::robocars_tof::ConstPtr& msg);

        ros::Subscriber tof1_sub;
        ros::Subscriber tof2_sub;
        image_transport::ImageTransport * it;
        image_transport::CameraSubscriber sub_image_and_camera;

        tensorflow::Session* session;
        tensorflow::GraphDef graph_def;

        // stats
        uint32_t totalImages=0;
        uint32_t missedImages=0;
        ros::Publisher stats_pub;

};

