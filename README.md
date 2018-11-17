# openvino_speech_recognition
This project is a speech recognization demo that uses OpenVINO for inference, deploys TensorFlow Speech Recognition model, and can be  on Intel's CPU, GPU, and NCS2. 
## System Configuration

### Software
* [Intel(R) Distribution of OpenVINO(TM) Toolkit](https://software.intel.com/en-us/openvino-toolkit)
* [Install mraa](https://github.com/intel-iot-devkit/mraa)
* Install TensorFlow 1.5 
Note: later versions of Tensorflow require AVX instruction set which isn't supported by the Up-board.

### Hardware 
* [Up-board](https://up-board.org/)
* 5-DOF Humanoid Robotic Arm and Hand
* AdaFruit 16-channel 12-bit PWM/Servo Shield - I2C interface

