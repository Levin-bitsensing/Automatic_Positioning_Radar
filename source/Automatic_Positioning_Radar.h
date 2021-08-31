#ifndef _FEATURE_MATCHING_TEST_H
#define _FEATURE_MATCHING_TEST_H

/* standard library */
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <utility>

using namespace std;


/* pcl 1.8.1 */
#include "pcl/visualization/cloud_viewer.h"
#include "pcl/io/io.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include <pcl/visualization/pcl_plotter.h>

using namespace pcl;
using namespace io;


/* opencv 4.5.1 */
#include "opencv2/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


/* Project source */
#include "util_calibration.h"


#endif