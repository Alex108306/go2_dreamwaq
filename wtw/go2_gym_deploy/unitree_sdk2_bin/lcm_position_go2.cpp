// lcm related headfile
#include <lcm/lcm-cpp.hpp>
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
// standard headfile
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>
// unitree_sdk2 related headfile
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

// To keep the project code stable and easy to understand, the `using namespace` statements used in unitree_sdk2 are not used

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);


// No changes needed: motor CRC check function provided by Unitree
uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{   
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}


// Remote controller key-value union, taken from unitree_sdk2, no changes needed
typedef union
{
  struct
  {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t start : 1;
    uint8_t select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t F1 : 1;
    uint8_t F2 : 1;
    uint8_t A : 1;
    uint8_t B : 1;
    uint8_t X : 1;
    uint8_t Y : 1;
    uint8_t up : 1;
    uint8_t right : 1;
    uint8_t down : 1;
    uint8_t left : 1;
  } components;
  uint16_t value;
} xKeySwitchUnion;


class Custom
{
public:
    explicit Custom(){}
    ~Custom(){}

    void Init();
    void InitLowCmd();
    void Loop();
    void LowStateMessageHandler(const void* messages);
    void JoystickHandler(const void *message);
    void InitRobotStateClient();
    void activateService(const std::string& serviceName,int activate);
    void lcm_send();
    void lcm_receive();
    void lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt* msg);
    void LowCmdWrite();
    void SetNominalPose();
    int queryServiceStatus(const std::string& serviceName);

    leg_control_data_lcmt leg_control_lcm_data = {0};
    state_estimator_lcmt body_state_simple = {0};
    pd_tau_targets_lcmt joint_command_simple = {0};
    rc_command_lcmt rc_command = {0};

    unitree_go::msg::dds_::LowState_ low_state{};
    unitree_go::msg::dds_::LowCmd_ low_cmd{};     
    unitree_go::msg::dds_::WirelessController_ joystick{};
    unitree::robot::go2::RobotStateClient rsc;
    
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_suber;
    lcm::LCM lc;
    
    xKeySwitchUnion key;
    int mode = 0;
    int motiontime = 0;
    float dt = 0.002; // unit [second]
    bool _firstRun;

    /*LowCmd write thread*/
    // Pointer to the DDS low-level command sending thread
    unitree::common::ThreadPtr LcmSendThreadPtr;
    unitree::common::ThreadPtr LcmRecevThreadPtr;
    unitree::common::ThreadPtr lowCmdWriteThreadPtr;

};

void Custom::InitRobotStateClient()
{
    rsc.SetTimeout(5.0f); 
    rsc.Init();
}

int Custom::queryServiceStatus(const std::string& serviceName)
{
    std::vector<unitree::robot::go2::ServiceState> serviceStateList;
    int ret,serviceStatus;
    ret = rsc.ServiceList(serviceStateList);
    size_t i, count=serviceStateList.size();
    for (i=0; i<count; i++)
    {
        const unitree::robot::go2::ServiceState& serviceState = serviceStateList[i];
        if(serviceState.name == serviceName)
        {
            if(serviceState.status == 0)
            {
                std::cout << "name: " << serviceState.name <<" is activate"<<std::endl;
                serviceStatus = 1;
            }
            else
            {
                std::cout << "name:" << serviceState.name <<" is deactivate"<<std::endl;
                serviceStatus = 0;
            } 
        }    
    }
    return serviceStatus;
    
}

void Custom::activateService(const std::string& serviceName,int activate)
{
    rsc.ServiceSwitch(serviceName, activate);  
}

void Custom::LowStateMessageHandler(const void* message)
{
    // Low-level state read via sdk2
    low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

void Custom::JoystickHandler(const void *message)
{
    // Remote controller signal
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    key.value = joystick.keys();
}

// -------------------------------------------------------------------------------
// Thread 1: LCM send thread
// Purpose: read low_state and joystick signals in real time via unitree_sdk2 and send them to the LCM middleware
void Custom::lcm_send(){
    // leg_control_lcm_data
    for (int i = 0; i < 12; i++)
    {
        leg_control_lcm_data.q[i] = low_state.motor_state()[i].q();
        leg_control_lcm_data.qd[i] = low_state.motor_state()[i].dq();
        leg_control_lcm_data.tau_est[i] = low_state.motor_state()[i].tau_est();
    }
    // Read orientation from the IMU
    for(int i = 0; i < 4; i++){
        // Quaternion
        body_state_simple.quat[i] = low_state.imu_state().quaternion()[i]; 
    }
    for(int i = 0; i < 3; i++){
        // roll pitch yaw
        body_state_simple.rpy[i] = low_state.imu_state().rpy()[i];
        // IMU 3-axis acceleration
        body_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];
        // IMU 3-axis linear acceleration
        body_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];
    }
    for(int i = 0; i < 4; i++){
        // Foot contact force
        body_state_simple.contact_estimate[i] = low_state.foot_force()[i];
    }
    // Remote controller button values and joystick values
    rc_command.left_stick[0] = joystick.lx();
    rc_command.left_stick[1] = joystick.ly();
    rc_command.right_stick[0] = joystick.rx();
    rc_command.right_stick[1] = joystick.ry();
    rc_command.right_lower_right_switch = key.components.R2;
    rc_command.right_upper_switch = key.components.R1;
    rc_command.left_lower_left_switch = key.components.L2;
    rc_command.left_upper_switch = key.components.L1;

    if(key.components.A > 0){
        mode = 0;
    } else if(key.components.B > 0){
        mode = 1;
    }else if(key.components.X > 0){
        mode = 2;
    }else if(key.components.Y > 0){
        mode = 3;
    }else if(key.components.up > 0){
        mode = 4;
    }else if(key.components.right > 0){
        mode = 5;
    }else if(key.components.down > 0){
        mode = 6;
    }else if(key.components.left > 0){
        mode = 7;
    }

    rc_command.mode = mode;


    lc.publish("leg_control_data", &leg_control_lcm_data);
    lc.publish("state_estimator_data", &body_state_simple);
    lc.publish("rc_command", &rc_command);

    // std::cout << "loop: messsages are sending ......" << std::endl;
}


// -------------------------------------------------------------------------------
// Thread 2: LCM receive thread
// Purpose: read, in real time via the LCM middleware, the desired joint control signals (q, qd, kp, kd, tau_ff) output by the PyTorch neural network
// Looking at go2_gym_deploy/envs/lcm_agent.py you can see that:
// the neural network only outputs the desired q; kp and kd can be set freely, and qd and tau_ff are set to 0
void Custom::lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt* msg){
    (void) rbuf;
    (void) chan;
    joint_command_simple = *msg; // Receive the joint signals output by the neural network
}

// Following the standard format recommended by LCM: loop and handle incoming LCM messages
void Custom::lcm_receive(){
    while(true){
        lc.handle();
    }
}


// -------------------------------------------------------------------------------
// Thread 3: unitree_sdk2 command write thread
// Purpose: initialize low_cmd; after a proper state machine, the motors execute the neural network output
void Custom::InitLowCmd()
{
    // The head member of the LowCmd type is the frame header,
    // which is used for the CRC check. head, levelFlag, gpio, etc. can just be set to the defaults shown in the examples.
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    /* The LowCmd type has 20 motorCmd members;
    each member's command controls one corresponding motor on the Go2 robot,
    but the Go2 robot only has 12 motors,
    so only the first 12 are valid and the remaining 8 are reserved. */
    for(int i=0; i<20; i++)
    {
        /* This line sets the mode variable of the motorCmd member to 0x01;
        0x01 sets the motor to servo mode.
        If you find during debugging that you cannot control the Go2 joint motors,
        check whether this variable is 0x01. */
        low_cmd.motor_cmd()[i].mode() = (0x01);   // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

void Custom::SetNominalPose(){
    // Running this cpp file not only initializes communication,
    // it also sets the initial joint angles for the lying-down pose
    // and puts every motor into position mode
    for(int i = 0; i < 12; i++){
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
        joint_command_simple.kp[i] = 20; 
        joint_command_simple.kd[i] = 0.5; 
    }

    // Joint angles when lying down
    joint_command_simple.q_des[0] = -0.3;
    joint_command_simple.q_des[1] = 1.2;
    joint_command_simple.q_des[2] = -2.721;
    joint_command_simple.q_des[3] = 0.3;
    joint_command_simple.q_des[4] = 1.2;
    joint_command_simple.q_des[5] = -2.721;
    joint_command_simple.q_des[6] = -0.3;
    joint_command_simple.q_des[7] = 1.2;
    joint_command_simple.q_des[8] = -2.721;
    joint_command_simple.q_des[9] = 0.3;
    joint_command_simple.q_des[10] = 1.2;
    joint_command_simple.q_des[11] = -2.721;

    std::cout<<"SET NOMINAL POSE"<<std::endl;
}

void Custom::LowCmdWrite(){
    motiontime++;
    
    if(_firstRun && leg_control_lcm_data.q[0] != 0){
        for(int i = 0; i < 12; i++){
            // When the program reaches here for the first time,
            // set the current joint angles as the target angles
            joint_command_simple.q_des[i] = leg_control_lcm_data.q[i];
            // Initialize L2+B to prevent damping from being triggered by mistake
            key.components.Y = 0;
            key.components.A = 0;
            key.components.B = 0;
            key.components.L2 = 0;
        }
        _firstRun = false;
    } 

    // A piece of safety-redundancy code:
    // when the roll angle exceeds its limit, or the pitch angle exceeds its limit, or L2+B is pressed on the remote
    // if (  low_state.imu_state().rpy()[0] > 0.5 || low_state.imu_state().rpy()[1] > 0.5 || ((int)key.components.B==1 && (int)key.components.L2==1))
    if ( std::abs(low_state.imu_state().rpy()[0]) > 0.8 || std::abs(low_state.imu_state().rpy()[1]) > 0.8 || ((int)key.components.B==1 && (int)key.components.L2==1))
    {       
        for (int i = 0; i < 12; i++){
            // Enter damping mode
            low_cmd.motor_cmd()[i].q() = 0;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 0;
            low_cmd.motor_cmd()[i].kd() = 5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }  
        std::cout << "======= Switched to Damping Mode, and the thread is sleeping ========"<<std::endl;
        sleep(1.5);

        while (true)
        {   
            
            // sleep(0.1);

            if (((int)key.components.B==1 && (int)key.components.L2==1) ) {
                // [L2+B] is pressed again
                std::cout << "======= [L2+B] is pressed again, the script is about to exit========" <<std::endl;
                exit(0);
            } else if (((int)key.components.A==1 && (int)key.components.L2==1) ){
                rsc.ServiceSwitch("sport_mode", 1);
                std::cout << "======= activate sport_mode service and exit========" <<std::endl;
                sleep(0.5);
                exit(0);
            } else{   
                if (((int)key.components.Y==1 && (int)key.components.L2==1) ){
                    std::cout << "=======  Switch to Walk These Ways ========"<<std::endl;
                    std::cout<<"Communicatino is set up successfully" << std::endl;
                    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;
                    break;
                    
                }else{
                    std::cout << "======= Press [L2+B] again to exit ========"<<std::endl;
                    std::cout << "======= Press [L2+Y] again to switch to WTW ========"<<std::endl;
                    std::cout << "======= Press [L2+A] again to activate sport_mode service========"<<std::endl;
                    sleep(0.01);
                }
                
            }

        }
        
    } 
    else{
        for (int i = 0; i < 12; i++){
            // Only execute the neural network output once safety is ensured
            low_cmd.motor_cmd()[i].q() = joint_command_simple.q_des[i];
            low_cmd.motor_cmd()[i].dq() = joint_command_simple.qd_des[i];
            low_cmd.motor_cmd()[i].kp() = joint_command_simple.kp[i];
            low_cmd.motor_cmd()[i].kd() = joint_command_simple.kd[i];
            low_cmd.motor_cmd()[i].tau() = joint_command_simple.tau_ff[i];
        }  
    }
    
    /* The first line of this block computes the CRC checksum.
    The last line calls lowcmd_publisher's Write() to send the control command to the Go2 robot. */
    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    lowcmd_publisher->Write(low_cmd);
}


//
// End of the function definitions related to the looping worker threads
//----------------------------------------------------------------------



void Custom::Init(){
    _firstRun = true;
    InitLowCmd();
    SetNominalPose();

    // This determines which message is subscribed to and what action is taken when lc.handle() is called
    // Message subscribed to: "pd_plustau_targets"
    // Action taken: lcm_receive_Handler
    lc.subscribe("pd_plustau_targets", &Custom::lcm_receive_Handler, this);

    /*create low_cmd publisher*/
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();
    /*create low_state dds subscriber*/
    lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);
    /*create joystick dds subscriber*/
    joystick_suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_suber->InitChannel(std::bind(&Custom::JoystickHandler, this, std::placeholders::_1), 1);
}


void Custom::Loop(){
    // Adding threads implements the loop-function behavior

    // intervalMicrosec: 1 microsecond = 0.000001 seconds
    // When dt = 0.002 s
    // ntervalMicrosec = 2000us
    /*lcm send thread*/
    LcmSendThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_send_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::lcm_send, this);
    /*lcm receive thread*/
    LcmRecevThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_recev_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::lcm_receive, this);
    /*low command write thread*/
    lowCmdWriteThreadPtr = unitree::common::CreateRecurrentThreadEx("dds_write_thread", UT_CPU_ID_NONE, dt*1e6, &Custom::LowCmdWrite, this);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Caution: The scripts is about to shutdown Unitree sport_mode Service." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]); // Pass in this machine's network interface (PC or Jetson Orin)

    Custom custom;

    custom.InitRobotStateClient();
    if(custom.queryServiceStatus("sport_mode"))
    {
        std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        custom.activateService("sport_mode",0);
        sleep(0.5);
        if(!custom.queryServiceStatus("sport_mode")){
            std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        }
    } else{
        std::cout <<"sportd_mode is already deactivated now" << std::endl
                  <<"next step is setting up communication" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
    }


    custom.Init();

    std::cout<<"Communicatino is set up successfully" << std::endl;
    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;

    custom.Loop();

    while (true)
    {
        sleep(10);
    }

    return 0;
}