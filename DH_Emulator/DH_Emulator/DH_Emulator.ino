#include "Arduino.h"
#include "math.h"
#include "ccl_data.h"
/**
CMD packet structure to command switches and other 'wireline' tools

To Control Depth:
[DEPTHCODE]:[RATE0]:[RATE1]:[DIRECTION]:[CHKSUM]
  Response: [DEPTHCODE]:[ACK/NACK]:[CHKSUM]

To Control Switches:
[ADDR]:[CMD]:[CHKSUM]
  Response: [ADDR]:[ACK/NACK]


**/

#define FLOAT_TO_BYTES(N) (*(float*)(N))
#define BYTES_TO_FLOAT(N) (*(float*)(N))
#define SEC_IN_MIN (60)
#define BIT8_TO_16BIT(H, L) ((H << 8) | (L & 0xff))

#define DEPTH_UPDATE_TIME (100)
#define CCL_DELTA_DEPTH (25)
#define LEN(N) (sizeof(N)/sizeof(N[0]))

/******* Declarations *****************/
double nextDouble();
static double randn(double mu, double sigma);
/**************************************/


class Switch {
public:
    byte Address;

    explicit Switch(byte address) {
        this->Address = address;
    }

};


const Switch switches[5] = {
        Switch(0x01),
        Switch(0x02),
        Switch(0x03),
        Switch(0x04),
        Switch(0x05)
};


class WirelineTool {
public:
    float run();
};


class Depth : public WirelineTool {


    unsigned long timer = millis();

    float Rate = 50;

public:


    float Depth = 0;
    float DesiredRate = Rate;
    float RatePerClick = 20;
    float PrevRate = 0;
    bool automatic = false;
    float DesiredDepth = -1;

    float run() {

        float cur_depth = this->Depth;

        if ((millis() - this->timer) >= DEPTH_UPDATE_TIME) {
            this->timer += DEPTH_UPDATE_TIME;
            this->update_depth_manual();
        }


        //Serial.write(cur_depth, HEX);
        //Serial.print(" ");
        return cur_depth;
    }


private:

    float get_rate_analog(int rate_input) {
        return (float) map(rate_input, 0, 4095, 0, 350);
    }

    void tune_rate(float steps = 20) {
        float step_per_click = steps;

        if (this->Rate != this->DesiredRate) {
//      Serial1.println("TUNE_RATE");
//      Serial1.print(" Rate: ");
//      Serial1.print(this->Rate);
//      Serial1.print(" DesiredRate: ");
//      Serial1.println(this->DesiredRate);

            if (abs(this->Rate - this->DesiredRate) < step_per_click) {
                this->tune_rate(steps = steps / 2);
                //this->Rate = this->DesiredRate;
            } else if (this->Rate > this->DesiredRate) {
                this->Rate = this->Rate - step_per_click;
            } else if (this->Rate < this->DesiredRate) {
                this->Rate = this->Rate + step_per_click;
            }
        }
    }

    void tune_depth(float max_rate = 350, float depth_threshold = 50, float tuner = 1.01) {
        if (this->DesiredDepth > -1) {
            float delta_depth = this->Depth - this->DesiredDepth;

            if (abs(delta_depth) < 0.05) {
                this->DesiredRate = 0;
                return;
            }

            if (delta_depth > 0 && max_rate > 0) {
                max_rate *= -1;
            }

            if (abs(delta_depth) <= depth_threshold) {
                this->tune_depth(max_rate = max_rate / tuner, depth_threshold = depth_threshold / tuner);
            } else {
                this->DesiredRate = max_rate;
            }
        }
    }

    void update_depth_manual(int threshold = 100) {
        int rate_input = analogRead(A0);

        if (!this->automatic) {
            this->DesiredDepth = -1;
            this->DesiredRate = get_rate_analog(rate_input);
            this->tune_rate();
        } else {
            if (abs(this->PrevRate - rate_input) > threshold) {
                Serial1.print(this->PrevRate);
                Serial1.print(" ");
                Serial1.println(rate_input);
                this->automatic = false;
            }
        }

        this->PrevRate = rate_input;
        this->tune_rate();
        this->tune_depth();

        float rate_per_update = ((float) this->Rate / (float) SEC_IN_MIN) / (1000.0 / (float) DEPTH_UPDATE_TIME);

        this->Depth += rate_per_update;
        if (this->Depth < 0)
            this->Depth = 0;
    }

};


class TempTool : public WirelineTool {
    float thermalGradientPerFoot = 0.02346959924;
    float Temp = 25;
    const Depth *depth;

public:
    float run(const Depth &_depth) {
        this->depth = &_depth;

        this->temp_calc();
        return this->Temp;
    }

private:

    void temp_calc() {
        /**
          temp = thermalGradientPerFoot(depth) + 77(F)
        **/

        //this->Temp = (float) ((this->thermalGradientPerFoot * this->depth->Depth) + (float) 77);
        this->Temp = (float) (pow(this->depth->Depth/100.0, 2) + 77);
    }
};


class CCLTool : public WirelineTool {
    //float first = pgm_read_float_near(ccl_data+4);


public:

    float prev_depth = 0;
    byte blip = 0;
    float threshold= 40;
    int counter = 0;
    int anom_counter = 0;
    float gain = 1;
    float a = 0.1;
    bool anom_detected = false;
    float run(const Depth &depth) {
//        if (this->prev_depth < 0) {
//            this->prev_depth = depth.Depth;
//            this->threshold += randn(0,0.5);
//        }
//
//        if (abs(depth.Depth - this->prev_depth) >= threshold ) {
//            this->blip = 1;
//            this->prev_depth = depth.Depth;
//
//            if (random(0,10000) == 0){
//                // a 0.01% of happening
//                this->threshold = 40 + randn(10, 4);
//            }
//
//            this->threshold = 40 + randn(0, 0.5);
//        }
//        else{
//            this->blip = 0;
//        }
//        //Serial.println(this->blip);
//        return this->blip;
//    
//        int cur_depth = (unsigned int)(depth.Depth * 10.0);
//        float ccl_by_depth = pgm_read_float_near(ccl_data + cur_depth);
//        float depth_rate = abs(depth.Depth -this->prev_depth);
//        float depth_rate_ftmin = depth_rate * 60 * (1000.0/ DEPTH_UPDATE_TIME);
//
//        if(this->anom_detected == true){
//          this->a = 0.06;
//          if (this->anom_counter++ == 10){
//            this->a = 0.01;
//          }
//          this->anom_detected = false;
//        }
          
        if (this->counter > 50){
          this->counter = 0;
        }

//        if (this->counter >= 46){
//          // max is 15 min is 1
//          gain = 0.10 * (depth_rate_ftmin) + 1;
//        }
//        else{
//          gain = 1;
//        }
//
//        if (random(0,101) < 50 && gain == 1 && this->anom_detected == false){
//          this->anom_detected = true;
//        }
        
        float rads = 2.0*M_PI*(float)this->counter++/100.0;
        float ccl_by_depth = this->a*sin(20.0*rads); //* (random(-10000,10000)/10000.0);
        this->prev_depth = depth.Depth;
        return ccl_by_depth;
    }

};

class Packager {
    /**
    Data to send 'uphole' on serial in the following packet form:
  
    [DEPTH0]:[DEPTH1]:[TEMP0]:[TEMP1]:[CCL]:[CHKSUM]
    
    **/
    byte depth_data[sizeof(float)];
    byte prev_depth_data[sizeof(float)];
    byte desireddepth_data[sizeof(float)];
    byte temp_data[sizeof(float)];
    byte ccl_data[sizeof(float)];
    byte meta_data[sizeof(float)];
    byte automatic_flag;

public:
    void sendData(Depth &depth, TempTool &temp, CCLTool &ccl) {
        FLOAT_TO_BYTES(this->depth_data) = depth.run();
        FLOAT_TO_BYTES(this->temp_data) = temp.run(depth);
        FLOAT_TO_BYTES(this->meta_data) = (float) DEPTH_UPDATE_TIME;
        FLOAT_TO_BYTES(this->desireddepth_data) = depth.DesiredDepth;
        this->automatic_flag = (byte) depth.automatic;

        //this->ccl_blip = (byte)ccl.run(depth);
        FLOAT_TO_BYTES(this->ccl_data) = (float)ccl.run(depth);
        FLOAT_TO_BYTES(this->prev_depth_data) = (float)ccl.prev_depth;


        this->_send_data();
//        Serial.print("Depth: ");
//        Serial.print(depth.Depth);
//        Serial.print(" CCL_DATA: ");
//        Serial.println(ccl.run(depth));

    }

private:

    void _send_data() {
        this->_send_depth();
        this->_send_prev_depth();
        this->_send_temp();
        this->_send_meta_data();
        this->_send_desired_depth();
        this->_send_ccl_data();
        this->_send_automatic_flag();

    }

    void _send_prev_depth(){
      for(byte i: this->prev_depth_data){
        Serial.write(i);
      }
    }

    void _send_ccl_data(){
        for(byte i: this->ccl_data){
            Serial.write(i);
        }
    }


    void _send_desired_depth() {
        for (byte i: this->desireddepth_data) {
            Serial.write(i);
        }
    }

    void _send_automatic_flag() {
        Serial.write(this->automatic_flag);
    }

    void _send_depth() {
        for (byte i : this->depth_data) {
            Serial.write(i);
        }
    }

    void _send_temp() {
        for (byte i : this->temp_data) {
            Serial.write(i);
        }
    }

    void _send_meta_data() {
        for (byte i : this->meta_data) {
            Serial.write(i);
        }
    }

    byte _checksum() {
    }
};


class DepthController {

    Depth *depth;
    byte data_in[2];
    int end_depth;

public:
    explicit DepthController(Depth *obj) {
        this->depth = obj;
    }

    void listen() {
        if (!Serial1) {
            return;
        }

        if (Serial1.available() > 0) {
            for (byte &i : this->data_in) {
                i = Serial1.read();
            }

            this->end_depth = (int) (BIT8_TO_16BIT(this->data_in[0], this->data_in[1]));
            depth->automatic = true;
            depth->DesiredDepth = (float) this->end_depth;
        }
    }
};


Depth depth;
TempTool temp;
CCLTool ccl;
Packager packager;

DepthController dc(&depth);

double nextDouble() {
    double r = random(-1000,1000) / 1000.0;
    return r;
}

static double randn(double mu, double sigma){
    double x = 2.0 * nextDouble() -1.0;
    double y = 2.0 * nextDouble() - 1.0;
    double s = sq(x) + sq(y);
    while( s > 1.0){
        x = 2.0 * nextDouble() - 1.0;
        y = 2.0 * nextDouble() - 1.0;
        s = sq(x) + sq(y);
    }

    double xGaus = sqrt(-2.0 * log(s) / s) * x * sigma + mu;
    return xGaus;
}

void setup() {
    analogReadResolution(12);
    randomSeed(123);

    Serial.begin(115200);
    while (!Serial) {}

    Serial1.begin(115200);
    while (!Serial1) {}

    pinMode(13, OUTPUT);
    digitalWrite(13, HIGH);

    //float first = pgm_read_float_near(ccl_data+4);
    //Serial.println(first);

}


#define TOGGLE_PIN(N) ((digitalRead(13)) ? digitalWrite(13, LOW) : digitalWrite(13, HIGH))
void loop() {
    TOGGLE_PIN(13);
    packager.sendData(depth, temp, ccl);
    dc.listen();
    delay(50);
   
}
