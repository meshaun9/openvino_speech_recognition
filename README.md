# OpenVINO Speech Recognition Demo for NIPS 2018

This is a speech recognization demo that uses OpenVINO for inference, uses [TensorFlow Speech Recognition model](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf), and can be deployed on Intel's CPU, GPU, and NCS2. Based on the inference results the robotic arm will attempt sign language and hand gestures.

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

