/******************************************************************************
 *                                                                            *
 * Copyright (C) 2020 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>
#include <vector>
#include <cmath>
#include <limits>

#include <vtkSmartPointer.h>
#include <vtkCommand.h>
#include <vtkPolyDataMapper.h>
#include <vtkPlaneSource.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkSuperquadric.h>
#include <vtkSampleFunction.h>
#include <vtkContourFilter.h>
#include <vtkArrowSource.h>
#include <vtkTransform.h>
#include <vtkProperty.h>
#include <vtkCaptionActor2D.h>
#include <vtkTextProperty.h>
#include <vtkTextActor.h>
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleSwitch.h>
#include <vtkPLYReader.h>

#include <yarp/os/Value.h>
#include <yarp/os/Bottle.h>
#include <yarp/sig/PointCloud.h>
#include <yarp/math/Math.h>

#include "cardinal_points_grasp.h"

namespace viewer {

static std::mutex mtx;

/******************************************************************************/
class UpdateCommand : public vtkCommand {
    bool shutdown{false};

public:
    /**************************************************************************/
    vtkTypeMacro(UpdateCommand, vtkCommand);

    /**************************************************************************/
    static UpdateCommand *New() {
        return new UpdateCommand;
    }

    /**************************************************************************/
    void shutDown() {
        shutdown = true;
    }

    /**************************************************************************/
    void Execute(vtkObject* caller, unsigned long vtkNotUsed(eventId),
                 void* vtkNotUsed(callData)) {
        std::lock_guard<std::mutex> lck(mtx);
        vtkRenderWindowInteractor* iren = static_cast<vtkRenderWindowInteractor*>(caller);
        if (shutdown) {
            iren->GetRenderWindow()->Finalize();
            iren->TerminateApp();
        } else {
            iren->Render();
        }
    }
};

/******************************************************************************/
class Viewer {
    vtkSmartPointer<vtkRenderer>                    vtk_renderer{nullptr};
    vtkSmartPointer<vtkRenderWindow>                vtk_renderWindow{nullptr};
    vtkSmartPointer<vtkRenderWindowInteractor>      vtk_renderWindowInteractor{nullptr};
    vtkSmartPointer<UpdateCommand>                  vtk_updateCallback{nullptr};
    vtkSmartPointer<vtkAxesActor>                   vtk_axes{nullptr};
    vtkSmartPointer<vtkInteractorStyleSwitch>       vtk_style{nullptr};
    vtkSmartPointer<vtkCamera>                      vtk_camera{nullptr};
    vtkSmartPointer<vtkPlaneSource>                 vtk_floor{nullptr};
    vtkSmartPointer<vtkPolyDataMapper>              vtk_floor_mapper{nullptr};
    vtkSmartPointer<vtkActor>                       vtk_floor_actor{nullptr};
    vtkSmartPointer<vtkPolyDataMapper>              vtk_object_mapper{nullptr};
    vtkSmartPointer<vtkPoints>                      vtk_object_points{nullptr};
    vtkSmartPointer<vtkUnsignedCharArray>           vtk_object_colors{nullptr};
    vtkSmartPointer<vtkPolyData>                    vtk_object_polydata{nullptr};
    vtkSmartPointer<vtkVertexGlyphFilter>           vtk_object_filter{nullptr};
    vtkSmartPointer<vtkActor>                       vtk_object_actor{nullptr};
    vtkSmartPointer<vtkSuperquadric>                vtk_superquadric{nullptr};
    vtkSmartPointer<vtkSampleFunction>              vtk_superquadric_sample{nullptr};
    vtkSmartPointer<vtkContourFilter>               vtk_superquadric_contours{nullptr};
    vtkSmartPointer<vtkTransform>                   vtk_superquadric_transform{nullptr};
    vtkSmartPointer<vtkPolyDataMapper>              vtk_superquadric_mapper{nullptr};
    vtkSmartPointer<vtkActor>                       vtk_superquadric_actor{nullptr};
    std::vector<vtkSmartPointer<vtkArrowSource>>    vtk_arrows;
    std::vector<vtkSmartPointer<vtkPolyDataMapper>> vtk_arrows_mappers;
    std::vector<vtkSmartPointer<vtkTransform>>      vtk_arrows_transforms;
    std::vector<vtkSmartPointer<vtkActor>>          vtk_arrows_actors;

    vtkSmartPointer<vtkPLYReader>                   vtk_model_reader{nullptr};
    vtkSmartPointer<vtkPolyDataMapper>              vtk_model_mapper{nullptr};
    vtkSmartPointer<vtkActor>                       vtk_model_actor{nullptr};
    vtkSmartPointer<vtkTransform>                   vtk_model_transform{nullptr};

    yarp::os::Bottle sqParams;

public:
    /**************************************************************************/
    Viewer() = delete;

    /**************************************************************************/
    Viewer(const int x, const int y, const int w, const int h) {
        vtk_renderer = vtkSmartPointer<vtkRenderer>::New();
        vtk_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        vtk_renderWindow->SetPosition(x, y);
        vtk_renderWindow->SetSize(w, h);
        vtk_renderWindow->SetWindowName("VTK 3D Viewer");
        vtk_renderWindow->AddRenderer(vtk_renderer);
        vtk_renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        vtk_renderWindowInteractor->SetRenderWindow(vtk_renderWindow);
        vtk_renderer->SetBackground(std::vector<double>({.7, .7, .7}).data());

        vtk_axes = vtkSmartPointer<vtkAxesActor>::New();
        vtk_axes->GetXAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
        vtk_axes->GetYAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
        vtk_axes->GetZAxisCaptionActor2D()->GetTextActor()->SetTextScaleModeToNone();
        vtk_axes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(10);
        vtk_axes->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(10);
        vtk_axes->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(10);
        vtk_axes->SetTotalLength(std::vector<double>({.1, .1, .1}).data());
        vtk_renderer->AddActor(vtk_axes);

        vtk_style = vtkSmartPointer<vtkInteractorStyleSwitch>::New();
        vtk_style->SetCurrentStyleToTrackballCamera();
        vtk_renderWindowInteractor->SetInteractorStyle(vtk_style);
    }

    /**************************************************************************/
    void start() {
        vtk_renderWindowInteractor->Initialize();
        vtk_renderWindowInteractor->CreateRepeatingTimer(10);
        vtk_updateCallback = vtkSmartPointer<UpdateCommand>::New();
        vtk_renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, vtk_updateCallback);
        vtk_renderWindowInteractor->Start();
    }

    /**************************************************************************/
    void stop() {
        vtk_updateCallback->shutDown();
    }

    /**************************************************************************/
    void addCamera(const std::vector<double>& position, const std::vector<double>& focalpoint,
                   const std::vector<double>& viewup, const double view_angle) {
        std::lock_guard<std::mutex> lck(mtx);
        vtk_camera = vtkSmartPointer<vtkCamera>::New();
        vtk_camera->SetPosition(position.data());
        vtk_camera->SetFocalPoint(focalpoint.data());
        vtk_camera->SetViewUp(viewup.data());
        vtk_camera->SetViewAngle(view_angle);
        vtk_renderer->SetActiveCamera(vtk_camera);
    }

    /**************************************************************************/
    void addTable(const std::vector<double>& center, const std::vector<double>& normal) {
        std::lock_guard<std::mutex> lck(mtx);
        if (vtk_floor_actor) {
            vtk_renderer->RemoveActor(vtk_floor_actor);
        }

        vtk_floor = vtkSmartPointer<vtkPlaneSource>::New();
        vtk_floor->SetOrigin(0., 0., 0.);
        vtk_floor->SetPoint1(.5, 0., 0.);
        vtk_floor->SetPoint2(0., .5, 0.);
        vtk_floor->SetResolution(20, 20);
        vtk_floor->SetCenter(const_cast<std::vector<double>&>(center).data());
        vtk_floor->SetNormal(const_cast<std::vector<double>&>(normal).data());
        vtk_floor->Update();

        vtk_floor_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        vtk_floor_mapper->SetInputData(vtk_floor->GetOutput());
        
        vtk_floor_actor = vtkSmartPointer<vtkActor>::New();
        vtk_floor_actor->SetMapper(vtk_floor_mapper);
        vtk_floor_actor->GetProperty()->SetRepresentationToWireframe();
        
        vtk_renderer->AddActor(vtk_floor_actor);
    }

    /**************************************************************************/
    void addObject(std::shared_ptr<yarp::sig::PointCloud<yarp::sig::DataXYZRGBA>> pc) {
        std::lock_guard<std::mutex> lck(mtx);
        if (vtk_object_actor) {
            vtk_renderer->RemoveActor(vtk_object_actor);
        }

        if (vtk_model_actor) {
            vtk_renderer->RemoveActor(vtk_model_actor);
        }

        vtk_object_points = vtkSmartPointer<vtkPoints>::New();
        vtk_object_colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        vtk_object_colors->SetNumberOfComponents(3);

        std::vector<unsigned char> color(3);
        for (size_t i = 0; i < pc->size(); i++) {
            const auto& p = (*pc)(i);
            vtk_object_points->InsertNextPoint(p.x, p.y, p.z);

            color = {p.r, p.g, p.b};
            vtk_object_colors->InsertNextTypedTuple(color.data());
        }

        vtk_object_polydata = vtkSmartPointer<vtkPolyData>::New();
        vtk_object_polydata->SetPoints(vtk_object_points);
        vtk_object_polydata->GetPointData()->SetScalars(vtk_object_colors);

        vtk_object_filter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vtk_object_filter->SetInputData(vtk_object_polydata);
        vtk_object_filter->Update();

        vtk_object_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        vtk_object_mapper->SetInputConnection(vtk_object_filter->GetOutputPort());

        vtk_object_actor = vtkSmartPointer<vtkActor>::New();
        vtk_object_actor->SetMapper(vtk_object_mapper);
        vtk_object_actor->GetProperty()->SetPointSize(1);

        vtk_renderer->AddActor(vtk_object_actor);
    }

    /**************************************************************************/
    void addModel(const std::string &model_name, const yarp::sig::Matrix &T,
                  std::vector<double> &color) {
        std::lock_guard<std::mutex> lck(mtx);
        if (vtk_model_actor) {
            vtk_renderer->RemoveActor(vtk_model_actor);
        }

        vtk_model_reader = vtkSmartPointer<vtkPLYReader>::New();
        vtk_model_reader->SetFileName( model_name.c_str() );
        vtk_model_reader->Update();

        vtk_model_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        vtk_model_mapper->SetInputConnection(vtk_model_reader->GetOutputPort());

        vtk_model_actor = vtkSmartPointer<vtkActor>::New();
        vtk_model_actor->SetMapper(vtk_model_mapper);
        vtk_model_actor->GetProperty()->SetColor(color.data());
        vtk_model_actor->GetProperty()->SetOpacity(.6);

        vtk_model_transform = vtkSmartPointer<vtkTransform>::New();
        const auto axisangle = yarp::math::dcm2axis(T);
        double angle = ((180. / M_PI) * axisangle[3]);
        vtk_model_transform->Translate(T.subcol(0, 3, 3).data());
        vtk_model_transform->RotateWXYZ(angle, axisangle.subVector(0, 2).data());
        vtk_model_actor->SetUserTransform(vtk_model_transform);

        vtk_renderer->AddActor(vtk_model_actor);
    }

    /**************************************************************************/
    void addSuperquadric(const yarp::os::Bottle& params) {
        std::lock_guard<std::mutex> lck(mtx);
        if (vtk_superquadric_actor) {
            vtk_renderer->RemoveActor(vtk_superquadric_actor);
        }

        sqParams = params;
        // Note: roundness parameter for axes x and y is shared in SQ model,
        //       but VTK shares axes x and z (ThetaRoundness).
        //       To get a good display, directions of axes y and z need to be swapped
        //       => parameters for y and z are inverted and a rotation of -90 degrees around x is added
        const auto x = sqParams.get(0).asDouble();
        const auto y = sqParams.get(1).asDouble();
        const auto z = sqParams.get(2).asDouble();
        const auto angle = sqParams.get(3).asDouble();
        const auto bx = sqParams.get(4).asDouble();
        const auto by = sqParams.get(6).asDouble();
        const auto bz = sqParams.get(5).asDouble();
        const auto eps_1 = sqParams.get(7).asDouble();
        const auto eps_2 = sqParams.get(8).asDouble();

        vtk_superquadric = vtkSmartPointer<vtkSuperquadric>::New();
        vtk_superquadric->ToroidalOff();
        vtk_superquadric->SetSize(1.);
        vtk_superquadric->SetCenter(0., 0., 0.);
        vtk_superquadric->SetScale(bx, by, bz);
        vtk_superquadric->SetPhiRoundness(eps_1);
        vtk_superquadric->SetThetaRoundness(eps_2);

        vtk_superquadric_sample = vtkSmartPointer<vtkSampleFunction>::New();
        vtk_superquadric_sample->SetSampleDimensions(50, 50, 50);
        vtk_superquadric_sample->SetImplicitFunction(vtk_superquadric);
        vtk_superquadric_sample->SetModelBounds(-bx, bx, -by, by, -bz, bz);

        vtk_superquadric_contours = vtkSmartPointer<vtkContourFilter>::New();
        vtk_superquadric_contours->SetInputConnection(vtk_superquadric_sample->GetOutputPort());
        vtk_superquadric_contours->GenerateValues(1, 0., 0.);

        vtk_superquadric_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        vtk_superquadric_mapper->SetInputConnection(vtk_superquadric_contours->GetOutputPort());
        vtk_superquadric_mapper->ScalarVisibilityOff();

        vtk_superquadric_actor = vtkSmartPointer<vtkActor>::New();
        vtk_superquadric_actor->SetMapper(vtk_superquadric_mapper);
        vtk_superquadric_actor->GetProperty()->SetColor(0., 1., 0.);
        vtk_superquadric_actor->GetProperty()->SetOpacity(.4);

        vtk_superquadric_transform = vtkSmartPointer<vtkTransform>::New();
        vtk_superquadric_transform->Translate(x, y, z);
        vtk_superquadric_transform->RotateZ(angle);
        vtk_superquadric_transform->RotateX(-90.);
        vtk_superquadric_actor->SetUserTransform(vtk_superquadric_transform);

        vtk_renderer->AddActor(vtk_superquadric_actor);
    }

    /**************************************************************************/
    void focusOnSuperquadric() {
        std::lock_guard<std::mutex> lck(mtx);
        std::vector<double> centroid(3);
        vtk_superquadric_transform->GetPosition(centroid.data());
        vtk_camera->SetPosition(0., 0., centroid[2] + .15);
        vtk_camera->SetFocalPoint(centroid.data());
    }

    /**************************************************************************/
    bool showCandidates(const std::vector<cardinal_points_grasp::rankable_candidate>& candidates) {
        std::lock_guard<std::mutex> lck(mtx);
        if (!vtk_arrows_actors.empty()) {
            for (auto vtk_actor:vtk_arrows_actors) {
                vtk_renderer->RemoveActor(vtk_actor);
            }
            vtk_arrows.clear();
            vtk_arrows_mappers.clear();
            vtk_arrows_transforms.clear();
            vtk_arrows_actors.clear();
        }

        if (sqParams.size() == 0){
            return false;
        }

        const auto x = sqParams.get(0).asDouble();
        const auto y = sqParams.get(1).asDouble();
        const auto z = sqParams.get(2).asDouble();
        const auto angle = sqParams.get(3).asDouble() * (M_PI / 180.);
        const auto bx = sqParams.get(4).asDouble();
        const auto by = sqParams.get(5).asDouble();
        const auto bz = sqParams.get(6).asDouble();

        const yarp::sig::Vector sqCenter{x, y, z};
        std::vector<yarp::sig::Vector> sqPoints{{bx, 0., 0., 1.}, {0., -by, 0., 1.},
                                                {-bx, 0., 0., 1.}, {0., by, 0., 1.},
                                                {0., 0., bz, 1.}};
        for (auto& sq_p_:sqPoints) {
            sq_p_ = sqCenter + (yarp::math::axis2dcm({0., 0., 1., angle}) * sq_p_).subVector(0, 2);
        }

        for (const auto& c:candidates) {
            const auto& type = std::get<0>(c);
            const auto& err = std::get<1>(c);
            auto T = std::get<2>(c);
            const auto L = .1 * (1. - err); // arrows' max length

            const auto p = T.getCol(3).subVector(0, 2);
            yarp::sig::Vector sq_p;
            auto max_d{std::numeric_limits<double>::infinity()};
            for (const auto& sq_p_:sqPoints) {
                const auto d = yarp::math::norm(p - sq_p_);
                if (d < max_d) {
                    max_d = d;
                    sq_p = sq_p_;
                }
            }
            const auto axis_x = (sqCenter - sq_p) / yarp::math::norm(sqCenter - sq_p);
            // take a generic vector normal to axis_x
            yarp::sig::Vector axis_y;
            if (std::abs(axis_x[0]) > .001) {
                axis_y = yarp::sig::Vector{-axis_x[1] / axis_x[0], 1., 0.};
            } else if (std::abs(axis_x[1]) > .001) {
                axis_y = yarp::sig::Vector{1., -axis_x[0] / axis_x[1], 0.};
            } else {
                axis_y = yarp::sig::Vector{0., 1., -axis_x[1] / axis_x[2]};
            }
            axis_y /= yarp::math::norm(axis_y);
            const auto axis_z = yarp::math::cross(axis_x, axis_y);
            T.setSubcol(axis_x, 0, 0);
            T.setSubcol(axis_y, 0, 1);
            T.setSubcol(axis_z, 0, 2);
            T.setSubcol(sq_p, 0, 3);

            vtkSmartPointer<vtkArrowSource> vtk_arrow = vtkSmartPointer<vtkArrowSource>::New();
            vtk_arrow->SetTipResolution(10);
            vtk_arrow->SetShaftResolution(10);

            vtkSmartPointer<vtkPolyDataMapper> vtk_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            vtk_mapper->SetInputConnection(vtk_arrow->GetOutputPort());

            vtkSmartPointer<vtkActor> vtk_actor = vtkSmartPointer<vtkActor>::New();
            vtk_actor->SetMapper(vtk_mapper);
            if (type == "right") {
                vtk_actor->GetProperty()->SetColor(0., 0., 1.);
            } else {
                vtk_actor->GetProperty()->SetColor(1., 0., 0.);
            }
            vtk_actor->GetProperty()->SetOpacity(c == candidates.front() ? .8 : .25);

            vtkSmartPointer<vtkTransform> vtk_transform = vtkSmartPointer<vtkTransform>::New();
            vtk_transform->Translate(T.getCol(3).subVector(0, 2).data());
            const auto axisangle = yarp::math::dcm2axis(T);
            vtk_transform->RotateWXYZ((180. / M_PI) * axisangle[3], axisangle.subVector(0, 2).data());
            vtk_transform->Translate(-L, 0., 0.);
            vtk_transform->Scale(L, L, L);
            vtk_actor->SetUserTransform(vtk_transform);

            vtk_arrows.push_back(vtk_arrow);
            vtk_arrows_mappers.push_back(vtk_mapper);
            vtk_arrows_actors.push_back(vtk_actor);
            vtk_arrows_transforms.push_back(vtk_transform);

            vtk_renderer->AddActor(vtk_actor);
        }

        return true;
    }
};

}

#endif
