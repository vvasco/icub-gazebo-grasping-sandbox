# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Valentina Vasco
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

file(GLOB conf_ply ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.ply)
yarp_install(FILES ${conf_ply} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${PROJECT_NAME})
