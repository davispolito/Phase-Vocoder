// Music 256a / CS 476a | fall 2016
// CCRMA, Stanford University
//
// Author: Romain Michon (rmichonATccrmaDOTstanfordDOTedu)
// Description: Simple sine wave oscillator

#ifndef SINE_H_INCLUDED
#define SINE_H_INCLUDED


class Sine{
private:
    float currentSampleRate, currentAngle, angleDelta;
    const float PI = 3.1415926535897931;
    
public:
    Sine():currentSampleRate(0.0),currentAngle(0.0),angleDelta(0.0){}
    
    ~Sine(){}
    
    // sampling rate must be set to get an accurate frequency
    void setSamplingRate(int samplingRate){
        currentSampleRate = samplingRate;
    }
    
    int getSamplingRate(){
        return currentSampleRate;
    }
    
    void setFrequency(double frequency){
        const float cyclesPerSample = frequency / (float) currentSampleRate;
        angleDelta = cyclesPerSample * 2.0 * PI;
    }
    
    // compute one sample
    float tick(){
        const float currentSample = (float) std::sinf (currentAngle);
        currentAngle += angleDelta;
        return currentSample;
    }
};


#endif  // SINE_H_INCLUDED
