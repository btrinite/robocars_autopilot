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
#include <vector>
#include <queue>

#include <boost/format.hpp>

#include <date.h>
#include <json.hpp>


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <std_srvs/Empty.h>

#include <robocars_msgs/robocars_brain_state.h>
#include <robocars_msgs/robocars_tof.h>
#include <robocars_msgs/robocars_autopilot_output.h>

#include "edgetpu.h"

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include <robocars_autopilot/robocars_autopilot_stats.h>
#include <robocars_autopilot.hpp>

RosInterface * ri;

// Paran
static int loop_hz;
static std::string model_filename;
static std::string model_path;
static float throttling_fixed_value;

bool edgetpu_found = false;

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
            ri->publishPredict(e.steering_value, e.throttling_value);
            RobocarsStateMachine::react(e);
        };

        void react( IdleStatusEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react(TickEvent const & e) override {
            RobocarsStateMachine::react(e);
            __tick_count++;
            if (__tick_count%(2000/loop_hz)==0) {
                ri->reportStats();
            }
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
            ri->initStats();
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
            ri->initStats();
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
    throttlingMsg.norm = throttling;

    autopilot_throttling_pub.publish(throttlingMsg);
}

void RosInterface::initParam() {
    if (!node_.hasParam("loop_hz")) {
        node_.setParam ("loop_hz", 60);       
    }
    if (!node_.hasParam("model_filename")) {
        node_.setParam("model_filename",  std::string("modelcat.pb"));
    }
    if (!node_.hasParam("fix_autopilot_throttle_value")) {
        node_.setParam("fix_autopilot_throttle_value",0.35);
    }
}
void RosInterface::updateParam() {
    node_.getParam("loop_hz", loop_hz);
    node_.getParam("model_path", model_path);
    node_.getParam("model_filename", model_filename);
    node_.getParam("fix_autopilot_throttle_value", throttling_fixed_value);
}


void RosInterface::initSub () {
    sub_image_and_camera = it->subscribeCamera("/front_video_resize/image", 2, &RosInterface::callbackWithCameraInfo, this);
    tof1_sub = node_.subscribe<robocars_msgs::robocars_tof>("/sensors/tof1", 2, &RosInterface::tof1_msg_cb, this);
    tof2_sub = node_.subscribe<robocars_msgs::robocars_tof>("/sensors/tof2", 2, &RosInterface::tof2_msg_cb, this);
    state_sub = node_.subscribe<robocars_msgs::robocars_brain_state>("/robocars_brain_state", 2, &RosInterface::state_msg_cb, this);
    reloadModel_svc = node_.advertiseService("reloadModel", &RosInterface::reloadModel_cb, this);
}
void RosInterface::initPub() {
    autopilot_steering_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/steering", 10);
    autopilot_throttling_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/throttling", 10);
    stats_pub = node_.advertise<robocars_autopilot::robocars_autopilot_stats>("/autopilot/stats", 10);
}

static uint32_t lastTof1Value;
static uint32_t lastTof2Value;


template <class T> void RosInterface::resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    switch (model_input_type) {
      case kTfLiteFloat32:
        out[i] = output[i];
        break;
      case kTfLiteInt8:
        out[i] = static_cast<int8_t>(output[i] - 128);
        break;
      case kTfLiteUInt8:
        out[i] = static_cast<uint8_t>(output[i]);
        break;
      default:
        break;
    }
  }
}

float linear_unbin(int idx, int N=15, float offset=-1, float R=2.0) { 
    return (float) ((float)idx *(R/((float)N + offset)) + offset);
}

template <class T> float RosInterface::unbind(T* prediction, int prediction_size, size_t num_results,
               TfLiteType input_type) {
  // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
    top_result_pq;

    const long count = prediction_size;  // NOLINT(runtime/int)
    float a, maxa = -1, res;
    int idx=-1;
    for (int i = 0; i < count; ++i) {
        switch (input_type) {    
            case kTfLiteFloat32:
            a = prediction[i];
            break;
            case kTfLiteInt8:
            a = (prediction[i] + 128) / 256.0;
            break;
            case kTfLiteUInt8:
            a = prediction[i] / 255.0;;
            break;
            default:
            break;
        }
        if (a>maxa) { maxa = a; idx=i;}
    }   
    if (count>1) {
        res=linear_unbin(idx, prediction_size, -1, 2);
    } else {
        res=maxa;
    }
    return res;
}

void RosInterface::callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info) {
    static uint32_t lastSeq = 0;
    cv_bridge::CvImagePtr cv_ptr;

    if (updateStats(1, image_msg->header.seq-(lastSeq+1))) {
        ROS_WARN ("Autopilot: Losing images");
    };
    lastSeq = image_msg->header.seq;

    if (modelLoaded) {
        int image_width = 160;
        int image_height = 120;
        int image_channels = 3; 
        try {   
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8); //for tensorflow, using RGB8
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
            return;
        }

        std::vector<uchar> in;
        if (cv_ptr->image.isContinuous()) {
            // array.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
            in.assign(cv_ptr->image.data, cv_ptr->image.data + (cv_ptr->image.total()*cv_ptr->image.channels()));
        } else {
            ROS_ERROR_STREAM("Unable to convert image: ");
            return;
        }

        const std::vector<int> inputs = interpreter->inputs();
        const std::vector<int> outputs = interpreter->outputs();
        switch (model_input_type) {
            case kTfLiteFloat32:
                resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
            break;
            case kTfLiteInt8:
                resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
            break;
            case kTfLiteUInt8:
                resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
            break;
        }
        interpreter->Invoke();

        float predicted_Steering;
        switch (interpreter->tensor(output_steering)->type) {
            case kTfLiteFloat32:
                predicted_Steering = unbind<float>(interpreter->typed_output_tensor<float>(0), output_steering_size,
                    1, model_input_type);
            break;    
            case kTfLiteInt8:
                predicted_Steering = unbind<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
                        output_steering_size, 1, model_input_type);
                break;
            case kTfLiteUInt8:
                predicted_Steering = unbind<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                        output_steering_size, 1, model_input_type);
            break;
        }
        send_event(PredictEvent(predicted_Steering,throttling_fixed_value));
    } else {
        send_event(PredictEvent(0,0));
    }
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

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    ROS_INFO ("Failed to build interpreter.");
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ROS_INFO("Failed to allocate tensors.");
  }
  return interpreter;
}

std::unique_ptr<tflite::Interpreter> BuildLocalInterpreter(
    const tflite::FlatBufferModel& model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    ROS_INFO("Failed to build interpreter.");
  }
  interpreter->SetNumThreads(4);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ROS_INFO("Failed to allocate tensors.");
  }
  return interpreter;
}

bool RosInterface::reloadModel_cb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
    updateParam();
    modelLoaded = false;
    model = tflite::FlatBufferModel::BuildFromFile((model_path+"/"+model_filename).c_str());
    if (model) {
        ROS_INFO("Model loaded %s", (model_path+"/"+model_filename).c_str());
        model->error_reporter();
        ROS_INFO("Resolved reporter");
        if (edgetpu_found) {
            std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
                edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
            interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
        } else {
            interpreter = BuildLocalInterpreter(*model);
        }
        if (!interpreter) {
            ROS_INFO("Failed to construct interpreter");
        } else {
            interpreter->UseNNAPI(0);
            interpreter->SetAllowFp16PrecisionForFp32(0);
            ROS_INFO("tensors size: %ld", interpreter->tensors_size());
            ROS_INFO("nodes size: %ld", interpreter->nodes_size());
            ROS_INFO("inputs: %ld", interpreter->inputs().size());
            ROS_INFO("input(0) name: %s", interpreter->GetInputName(0));

            input = interpreter->inputs()[0];

            TfLiteIntArray* dims = interpreter->tensor(input)->dims;
            wanted_height = dims->data[1];
            wanted_width = dims->data[2];
            wanted_channels = dims->data[3];
            model_input_type = interpreter->tensor(input)->type;
            ROS_INFO("wanted_height: %d", wanted_height);
            ROS_INFO("wanted_width: %d", wanted_width);
            ROS_INFO("wanted_channels: %d", wanted_channels);
            ROS_INFO("Input Type : %d", model_input_type);
            
            if ((model_input_type != kTfLiteFloat32) && (model_input_type != kTfLiteInt8) && (model_input_type!= kTfLiteUInt8)) {
                ROS_INFO("cannot handle input type %d", interpreter->tensor(input)->type);
            }

            output_steering = interpreter->outputs()[0];
            TfLiteIntArray* output_dims = interpreter->tensor(output_steering)->dims;
            output_steering_size = output_dims->data[output_dims->size - 1];
            ROS_INFO("Output Steering Size : %d", output_steering_size);

            output_throttling = interpreter->outputs()[1];
            output_dims = interpreter->tensor(output_throttling)->dims;
            output_throttling_size = output_dims->data[output_dims->size - 1];
            ROS_INFO("Output Throttling Size : %d", output_throttling_size);

            modelLoaded = true;
        }
    } else {
        ROS_INFO("Failed to mmap model %s", (model_path+"/"+model_filename).c_str());
    }

    return true;
}

void RosInterface::initStats(void) {
    totalImages=0;
    missedImages=0;
}

bool RosInterface::updateStats(uint32_t received, uint32_t missed) {
    totalImages+=received;
    missedImages+=missed;
    if ((missedImages%10)==9) {
        return true;
    }
    return false;
}

void RosInterface::reportStats(void) {

    robocars_autopilot::robocars_autopilot_stats statsMsg;

    statsMsg.header.stamp = ros::Time::now();
    statsMsg.header.seq=1;
    statsMsg.header.frame_id = "stats";
    statsMsg.totalImages = totalImages;
    statsMsg.missedImages = missedImages;

    stats_pub.publish(statsMsg);

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "robocars_autopilot");
    
    ri = new RosInterface;

    ri->initPub();
    fsm_list::start();
    ri->initSub();

    ROS_INFO("Autopilot: Starting");
    std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> edgetpu_devices = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    ROS_INFO("Autopilot: TPU found %zu", edgetpu_devices.size());
    if (edgetpu_devices.size() > 0) {
        edgetpu_found = true;
    }

    // wait for FCU connection
    ros::Rate rate(loop_hz);
    while(ros::ok()){
        // ros::spin();
        ros::spinOnce();
        send_event (TickEvent());
        rate.sleep();
    }
}

