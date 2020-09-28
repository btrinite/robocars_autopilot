/**
 * @file robocars_autopilot.cpp
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
 * Derived work from :
 *  - https://github.com/Namburger/edgetpu-minimal-example
 *  
 * main topic subscribed : 
 *  - /front_video_resize/image : the resized front video image
 *  - /robocars_brain_state : not used, as soon as a model is loaded, prediction on each received image starts
 * 
 * Topic published :
 *  - /autopilot/steering : predicted steering
 *  - /autopilot/throttling : predicted throtting (actually a configured fixed value)
 *  - /autopilot/stats : statistics about lost frame
 * 
 * Parameters :
 *  - loop_hz : tick frequency, used by FSM to trigger recurrent jobs like uopdating node's configuration, this implementation does not provide best performance (response time)
 *  - model_path : path zhere fo find model
 *  - model_filename : name of the model file to load
 * 
 * Services
 *  - reloadModel : invoke this service to trigger model loading
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

#include "model_utils.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <geometry_msgs/Twist.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <std_srvs/Empty.h>

#include <robocars_msgs/robocars_brain_state.h>
#include <robocars_msgs/robocars_mark.h>
#include <robocars_msgs/robocars_autopilot_output.h>

#include "edgetpu.h"

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"
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
static float autobrake_steering_thresh;
static float autobrake_brake_factor;
static float autobrake_speed_max;
static float autobrake_speed_thresh;

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
            ri->publishPredict(e.steering_value, e.throttling_value, e.braking_value, e.seq_num);
            RobocarsStateMachine::react(e);
        };

        void react( IdleStatusEvent const & e) override { 
            RobocarsStateMachine::react(e);
        };

        void react(TickEvent const & e) override {
            RobocarsStateMachine::react(e);
            __tick_count++;
            if ((__tick_count%(uint32_t)(2000/loop_hz))==0) {
                ri->reportStats();
                ri->updateParam();
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

void RosInterface::publishPredict(_Float32 steering, _Float32 throttling, _Float32 braking, uint32_t seq) {
    robocars_msgs::robocars_autopilot_output steeringMsg;
    robocars_msgs::robocars_autopilot_output throttlingMsg;
    robocars_msgs::robocars_autopilot_output brakingMsg;

    steeringMsg.header.stamp = ros::Time::now();
    steeringMsg.header.seq=seq;
    steeringMsg.header.frame_id = "pilotSteering";
    steeringMsg.norm = steering;

    autopilot_steering_pub.publish(steeringMsg);

    throttlingMsg.header.stamp = ros::Time::now();
    throttlingMsg.header.seq=seq;
    throttlingMsg.header.frame_id = "pilotThrottling";
    throttlingMsg.norm = throttling;

    autopilot_throttling_pub.publish(throttlingMsg);

    brakingMsg.header.stamp = ros::Time::now();
    brakingMsg.header.seq=seq;
    brakingMsg.header.frame_id = "pilotBraking";
    brakingMsg.norm = braking;

    autopilot_braking_pub.publish(brakingMsg);
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
    if (!node_.hasParam("autobrake_steering_thresh")) {
        node_.setParam("autobrake_steering_thresh",0.2);
    }
    if (!node_.hasParam("autobrake_brake_factor")) {
        node_.setParam("autobrake_brake_factor",1.0);
    }
    if (!node_.hasParam("autobrake_speed_thresh")) {
        node_.setParam("autobrake_speed_thresh",6.0);
    }
    if (!node_.hasParam("autobrake_speed_thresh")) {
        node_.setParam("autobrake_speed_max",12.0);
    }

}
void RosInterface::updateParam() {
    node_.getParam("loop_hz", loop_hz);
    node_.getParam("model_path", model_path);
    node_.getParam("model_filename", model_filename);
    node_.getParam("fix_autopilot_throttle_value", throttling_fixed_value);
    node_.getParam("autobrake_steering_thresh", autobrake_steering_thresh);
    node_.getParam("autobrake_brake_factor", autobrake_brake_factor);
    node_.getParam("autobrake_speed_thresh", autobrake_speed_thresh);
    node_.getParam("autobrake_speed_max", autobrake_speed_max);
}


void RosInterface::initSub () {
    //sub_image_and_camera = it->subscribeCamera("/front_video_resize/image", 1, &RosInterface::callbackWithCameraInfo, this);
    sub_image = it->subscribe("/front_video_resize/image", 1, &RosInterface::callbackNoCameraInfo, this);
    state_sub = node_.subscribe<robocars_msgs::robocars_brain_state>("/robocars_brain_state", 1, &RosInterface::state_msg_cb, this);
    mark_sub = node_.subscribe<robocars_msgs::robocars_mark>("/mark", 1, &RosInterface::mark_msg_cb, this);
    speed_sub = node_.subscribe<geometry_msgs::Twist>("/gym/speed", 1, &RosInterface::speed_msg_cb, this);
    reloadModel_svc = node_.advertiseService("reloadModel", &RosInterface::reloadModel_cb, this);
}
void RosInterface::initPub() {
    autopilot_steering_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/steering", 1);
    autopilot_throttling_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/throttling", 1);
    autopilot_braking_pub = node_.advertise<robocars_msgs::robocars_autopilot_output>("/autopilot/braking", 1);
    stats_pub = node_.advertise<robocars_autopilot::robocars_autopilot_stats>("/autopilot/stats", 1);
}

static uint32_t lastBrakeValue = 0;
static uint32_t lastSpeedValue = 0;

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

void RosInterface::callbackNoCameraInfo(const sensor_msgs::ImageConstPtr& image_msg) {
    static uint32_t lastSeq = 0;
    cv_bridge::CvImagePtr cv_ptr;
    static uint32_t missingSeq = 0;

    missingSeq = image_msg->header.seq-(lastSeq+1);
    if (missingSeq > 1) {
        updateStats(1, missingSeq);
        ROS_WARN ("Autopilot: Losing %d images(Seq %d)", missingSeq, image_msg->header.seq);
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
            {
                /* resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
                */
                /*std::transform(in.begin(), in.end(), interpreter->typed_tensor<float>(input),
                 [](uchar i) { return i / 254; });
                */
                float* fillInput = interpreter->typed_tensor<float>(input);
                for (int i=0; i<in.size();i++) {
                    //fillInput[i] = ((float)in[i]-127.5)/127.5;
                    fillInput[i] = (float)in[i];
                }
            }
            break;
            case kTfLiteInt8:
            {
                resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
            }
            break;
            case kTfLiteUInt8:
            {
                /*WIP resize code (based on tf operators) not working on edge tpu for now, since image input is already at the correct size, just copy image as is in tensor*/
                  uint8_t* fillInput = interpreter->typed_input_tensor<uint8_t>(input);
                  std::memcpy(fillInput, in.data(), in.size());
                /*
                resize<uint8_t>(interpreter->typed_input_tensor<uint8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
                    */
            }
            break;
        }

        interpreter->Invoke(); // this is where the magick happen
        float predicted_Steering;
        switch (interpreter->tensor(output_steering)->type) {
            case kTfLiteFloat32:
                predicted_Steering = unbind<float>(interpreter->typed_tensor<float>(output_steering), output_steering_size,
                    1, model_output_steering_type);
            break;    
            case kTfLiteInt8:
                predicted_Steering = unbind<int8_t>(interpreter->typed_tensor<int8_t>(output_steering),
                        output_steering_size, 1, model_output_steering_type);
                break;
            case kTfLiteUInt8:
                predicted_Steering = unbind<uint8_t>(interpreter->typed_tensor<uint8_t>(output_steering),
                        output_steering_size, 1, model_output_steering_type);
                predicted_Steering = -1.0 + (predicted_Steering*2.0);        
            break;
        }

        float predicted_Brake=0.0;
        if (interpreter->outputs().size()>2) {
            switch (interpreter->tensor(output_brake)->type) {
                case kTfLiteFloat32:
                    predicted_Brake = unbind<float>(interpreter->typed_output_tensor<float>(2), output_brake_size,
                        1, model_output_brake_type);
                break;    
                case kTfLiteInt8:
                    predicted_Brake = unbind<int8_t>(interpreter->typed_output_tensor<int8_t>(2),
                            output_brake_size, 1, model_output_brake_type);
                    break;
                case kTfLiteUInt8:
                    predicted_Brake = unbind<uint8_t>(interpreter->typed_output_tensor<uint8_t>(2),
                            output_brake_size, 1, model_output_brake_type);
                break;
            }
        } else {
            //Model do not provide brake, implement basic logic
            if (fabs(predicted_Steering)> autobrake_steering_thresh) {
                if (lastSpeedValue>autobrake_speed_thresh) {
                    predicted_Brake = mapRange (autobrake_speed_thresh,autobrake_speed_max,0,-1,(lastSpeedValue-autobrake_speed_thresh)) * autobrake_brake_factor;
                    ROS_INFO("Autopilot : apply brake: %f", predicted_Brake);
                }
            }
        }
        send_event(PredictEvent(predicted_Steering,throttling_fixed_value, predicted_Brake, image_msg->header.seq));
    } else {
        send_event(PredictEvent(0.0,0.0,0.0,0));
    }
}


void RosInterface::callbackWithCameraInfo(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info) {
    callbackNoCameraInfo (image_msg);
}

void RosInterface::mark_msg_cb(const robocars_msgs::robocars_mark::ConstPtr& msg){
    lastBrakeValue = msg->mark;
}

void RosInterface::speed_msg_cb(const geometry_msgs::Twist::ConstPtr& msg){
    lastSpeedValue = msg->linear.x;
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

bool RosInterface::reloadModel_cb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
    updateParam();
    modelLoaded = false;
    model = tflite::FlatBufferModel::BuildFromFile((model_path+"/"+model_filename).c_str());
    if (model) {
        ROS_INFO("Model loaded %s", (model_path+"/"+model_filename).c_str());
        model->error_reporter();
        ROS_INFO("Resolved reporter");
        if (edgetpu_found) {
            edgetpu_context =
                edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        }
        if (!edgetpu_context) {
            interpreter = std::move(coral::BuildInterpreter(*model));
        } else {
            interpreter = std::move(coral::BuildEdgeTpuInterpreter(*model, edgetpu_context.get()));
        }

        if (!interpreter) {
            ROS_INFO("Failed to construct interpreter");
        } else {

            ROS_INFO("tensors size: %ld", interpreter->tensors_size());
            ROS_INFO("nodes size: %ld", interpreter->nodes_size());
            ROS_INFO("inputs: %ld", interpreter->inputs().size());
            for (uint idx=0;idx<interpreter->inputs().size();idx++) {
                ROS_INFO("  input(%d) name: %s", idx, interpreter->GetInputName(idx));
            }
            ROS_INFO("outputs: %ld", interpreter->outputs().size());
            for (uint idx=0;idx<interpreter->outputs().size();idx++) {
                ROS_INFO("  output(%d) name: %s", idx, interpreter->GetOutputName(idx));
            }

            input = interpreter->inputs()[0];
            const auto& required_shape = coral::GetInputShape(*interpreter, 0);

            TfLiteIntArray* dims = interpreter->tensor(input)->dims;
            model_input_type = interpreter->tensor(input)->type;

            ROS_INFO("Input idx: %d", input);
            ROS_INFO("wanted_height: %d", required_shape[0]);
            ROS_INFO("wanted_width: %d", required_shape[1]);
            ROS_INFO("wanted_channels: %d", required_shape[2]);
            ROS_INFO("Input Type : %d (%d %d %d)", model_input_type, kTfLiteFloat32, kTfLiteInt8, kTfLiteUInt8);
            
            if ((model_input_type != kTfLiteFloat32) && (model_input_type != kTfLiteInt8) && (model_input_type!= kTfLiteUInt8)) {
                ROS_INFO("cannot handle input type %d", interpreter->tensor(input)->type);
            }

            output_steering = interpreter->outputs()[0];
            TfLiteIntArray* output_dims = interpreter->tensor(output_steering)->dims;
            output_steering_size = output_dims->data[output_dims->size - 1];
            model_output_steering_type = interpreter->tensor(output_steering)->type;
            ROS_INFO("Output Steering Idx : %d", output_steering);
            ROS_INFO("Output Steering Size : %d", output_steering_size);
            ROS_INFO("Output Steering Type : %d", model_output_steering_type);

            output_throttling = interpreter->outputs()[1];
            output_dims = interpreter->tensor(output_throttling)->dims;
            output_throttling_size = output_dims->data[output_dims->size - 1];
            model_output_throttling_type = interpreter->tensor(output_throttling)->type;
            ROS_INFO("Output Throttling Size : %d", output_throttling_size);
            ROS_INFO("Output Throttling Type : %d", model_output_throttling_type);

            if (interpreter->outputs().size()>2) {
                output_brake = interpreter->outputs()[2];
                output_dims = interpreter->tensor(output_brake)->dims;
                output_brake_size = output_dims->data[output_dims->size - 1];
                model_output_brake_type = interpreter->tensor(output_brake)->type;
                ROS_INFO("Output Brake Size : %d", output_brake_size);
                ROS_INFO("Output Brake Type : %d", model_output_brake_type);
            }

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
    ROS_INFO("TF Version %s", TFLITE_VERSION_STRING);
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

