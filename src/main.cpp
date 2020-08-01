/******************************************************************************
 *                                                                            *
 * Copyright (C) 2020 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

#include <cstdlib>
#include <string>
#include <vector>
#include <limits>

#include <yarp/os/Network.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Value.h>
#include <yarp/os/Property.h>
#include <yarp/os/RpcServer.h>
#include <yarp/os/Time.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/CartesianControl.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/sig/Vector.h>
#include <yarp/math/Math.h>

#include "grasp_IDL.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;
using namespace yarp::math;

/******************************************************************************/
class Grasper : public RFModule, public grasp_IDL {
    PolyDriver arm;
    PolyDriver hand;
    PolyDriver gaze;
    RpcServer rpc;

    /**************************************************************************/
    bool attach(RpcServer& source) override {
        return this->yarp().attachAsServer(source);
    }

    /**************************************************************************/
    bool configure(ResourceFinder& rf) override {
        string name = "icub-grasp";

        Property arm_options;
        arm_options.put("device", "cartesiancontrollerclient");
        arm_options.put("local", "/"+name+"/arm");
        arm_options.put("remote", "/icubSim/cartesianController/right_arm");
        if (!arm.open(arm_options)) {
            yError() << "Unable to open right arm driver!";
            return false;
        }

        Property hand_options;
        hand_options.put("device", "remote_controlboard");
        hand_options.put("local", "/"+name+"/hand");
        hand_options.put("remote", "/icubSim/right_arm");
        if (!hand.open(hand_options)) {
            yError() << "Unable to open right hand driver!";
            arm.close();
            return false;
        }

        Property gaze_options;
        gaze_options.put("device", "gazecontrollerclient");
        gaze_options.put("local", "/"+name+"/gaze");
        gaze_options.put("remote", "/iKinGazeCtrl");
        if (!gaze.open(gaze_options)) {
            yError() << "Unable to open gaze driver!";
            hand.close();
            arm.close();
            return false;
        }

        rpc.open("/"+name+"/rpc");
        attach(rpc);

        return true;
    }

    /**************************************************************************/
    bool grasp() override {
        // enable position control of the fingers
        IControlMode* imod;
        hand.view(imod);
        vector<int> fingers = {7, 8, 9, 10, 11, 12, 13, 14, 15};
        imod->setControlModes(fingers.size(), fingers.data(), vector<int>(fingers.size(), VOCAB_CM_POSITION).data());

        // target pose that allows grasing the object
        Vector x({-.24, .18, -.03});
        Vector o({-.14, -.79, .59, 3.07});

        // keep gazing at the object
        IGazeControl* igaze;
        gaze.view(igaze);
        igaze->setTrackingMode(true);
        igaze->lookAtFixationPoint(x);

        // reach for the pre-grasp pose
        ICartesianControl* iarm;
        arm.view(iarm);
        Vector dof({1, 0, 1, 1, 1, 1, 1, 1, 1, 1});
        iarm->setDOF(dof, dof);
        iarm->setTrajTime(.6);
        iarm->goToPoseSync(x + Vector({.07, 0., .03}), o);
        iarm->waitMotionDone(.1, 3.);

        // put the hand in the pre-grasp configuration
        IPositionControl* ihand;
        IControlLimits* ilim;
        hand.view(ihand);
        hand.view(ilim);
        double pinkie_min, pinkie_max;
        ilim->getLimits(15, &pinkie_min, &pinkie_max);
        ihand->setRefAccelerations(fingers.size(), fingers.data(), vector<double>(fingers.size(), numeric_limits<double>::infinity()).data());
        ihand->setRefSpeeds(fingers.size(), fingers.data(), vector<double>({60., 60., 60., 60., 60., 60., 60., 60., 200.}).data());
        ihand->positionMove(fingers.size(), fingers.data(), vector<double>({60., 80., 0., 0., 0., 0., 0., 0., pinkie_max}).data());
        Time::delay(5.);

        // reach for the object
        iarm->goToPoseSync(x, o);
        iarm->waitMotionDone(.1, 3.);

        // close fingers
        ihand->positionMove(fingers.size(), fingers.data(), vector<double>({60., 80., 40., 35., 40., 35., 40., 35., pinkie_max}).data());

        // give time to adjust the contacts
        Time::delay(5.);

        // lift up the object
        x += Vector({-.07, 0., .1});
        igaze->lookAtFixationPoint(x);
        iarm->goToPoseSync(x, o);
        iarm->waitMotionDone(.1, 3.);

        return true;
    }

    /**************************************************************************/
    double getPeriod() override {
        return 1.0;
    }

    /**************************************************************************/
    bool updateModule() override {
        return true;
    }

    /**************************************************************************/
    bool close() override {
        rpc.close();
        gaze.close();
        hand.close();
        arm.close();
        return true;
    }
};

/******************************************************************************/
int main(int argc, char *argv[]) {
    Network yarp;
    if (!yarp.checkNetwork()) {
        yError() << "Unable to find YARP server!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    rf.configure(argc,argv);

    Grasper grasper;
    return grasper.runModule(rf);
}
