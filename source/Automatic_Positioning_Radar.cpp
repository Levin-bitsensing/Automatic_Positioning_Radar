/*************************************************************************************
* -----------------------------------   include   ----------------------------------- *
**************************************************************************************/

#include "Automatic_Positioning_Radar.h"

/**************************************************************************************
* -----------------------------------   define   ------------------------------------ *
**************************************************************************************/
#define IMG_WIDTH	1920
#define IMG_HEIGHT	1080


/**************************************************************************************
* ------------------------------   global variables   ------------------------------- *
**************************************************************************************/



/**************************************************************************************
* -----------------------------   function prototypes   ----------------------------- *
**************************************************************************************/

static Mat Rx(float theta);
static Mat Ry(float theta);
static Mat Rz(float theta);
static bool isRotationMatrix(Mat &R);
static Vec3f rot2eul(Mat R);
static Vec3f rot2euld(Mat R);

void initCamera(pcl::visualization::PCLVisualizer::Ptr& viewer);


/**************************************************************************************
* ------------------------------   main functions   ------------------------------- *
**************************************************************************************/


int main(int argc, char* argv[])
{

	/***********************
	**** configuration *****
	***********************/

	float32_t image_size[2] = { IMG_WIDTH, IMG_HEIGHT };
	float32_t intrinsic_parameter[5] = { 2119.137451, 2120.412109, 925.452271, 564.02832, -11.126279 };		/* fx, fy, cx, cy, skew */
	float32_t distortion_coefficient[5] = { -0.560545, 0.515465, -0.070978, -0.001732, -1.6e-05 };
	float32_t extrinsic_translation[3] = { -6.0, 0.0, 10.0 };												/* tx, ty, tz */
	float32_t installation_angle_offset[3] = { 0.0f, 11.0f, 4.5f };											/* rx, ry, rz */
	float32_t extrinsic_euler_angle[3] = { -1 * installation_angle_offset[0], -1 * installation_angle_offset[1], -1 * installation_angle_offset[2] };

	Mat R_tf = Rz(_deg2rad(-90.0f)) * Ry(_deg2rad(-90.0f)) *  Rx(_deg2rad(180.0f));

	char input_file_path_name[256];
	char input_file_path[256] = { "../input/" };
	char input_file_name[10][256] = {
		{"1_az0_el0_20210805160431_1"},
		{"2_az0_el1_20210805155600_62"},
		{"3_az0_el2_20210805155534_219"},
		{"4_az0_el3_20210805155509_40"},
		{"5_az0_el4_20210805155422_75"},
		{"6_az0_el5_20210805155356_235"},
		{"7_az1_el0_20210805155721_25"},
		{"8_az2_el0_20210805155744_170"},
		{"9_az-1_el0_20210805155845_205"},
		{"10_az-2_el0_20210805155912_16"},
	};
	char input_file_ext[10] = { ".jpg" };

	/* roi */
	Point poly[] = { 
		Point(0, 910), 
		Point(0, IMG_HEIGHT - 1),
		Point(IMG_WIDTH - 1, IMG_HEIGHT - 1),
		Point(IMG_WIDTH - 1, 596),
		Point(1305, 333), 
		Point(908, 333) };

	/* feature matching*/
	const float match_ratio_thresh = 0.75f;

	/* PnP, RANSAC parameters */
	int iterationsCount = 500;      // number of Ransac iterations.
	float reprojectionError = 6.0;  // maximum allowed distance to consider it an inlier.
	float confidence = 0.99;       // ransac successful confidence.
	bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as initial approximations of the rotation and translation vectors
	int pnpMethod = SOLVEPNP_ITERATIVE;

	/***********************
	**** initialization ****
	***********************/

	/* reference camera */
	PerspectiveCamera cmr_init(image_size, intrinsic_parameter, distortion_coefficient, extrinsic_euler_angle, extrinsic_translation);


	/* build camera matrix for distortion correction */
	Mat cameraMatrix = (Mat_<float>(3, 3) <<
		intrinsic_parameter[0], intrinsic_parameter[4], intrinsic_parameter[2],
		0,						intrinsic_parameter[1], intrinsic_parameter[3],
		0,						0,						1);

	Mat distCoeffs = (Mat_<float>(5,1) <<
		distortion_coefficient[0], distortion_coefficient[1], distortion_coefficient[3], distortion_coefficient[4], distortion_coefficient[2]);

	/* reference image file input */
	sprintf_s(input_file_path_name, "%s%s%s", input_file_path, input_file_name[0], input_file_ext);
	Mat img_ref = imread(input_file_path_name);


	/* image distortion correction */
	Mat img_buf;
	undistort(img_ref, img_buf, cameraMatrix, distCoeffs);
	img_ref = img_buf.clone();

	/* set roi */
	Mat roi_ref(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC1, Scalar(0));
	fillConvexPoly(roi_ref, poly, int(sizeof(poly) / sizeof(poly[0])), Scalar(255));

	/* feature detector */
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	
	/* feature detection */
	vector<KeyPoint> keypoints_ref;
	Mat descriptors_ref;
	detector->detectAndCompute(img_ref, roi_ref, keypoints_ref, descriptors_ref);
	
	/* output */
	Vec3f out_eul_angle[10];
	Mat out_tl[10];

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	initCamera(viewer);
	
	string win_name_matching = "Good Matches & Object detection";
	cv::namedWindow(win_name_matching, WINDOW_KEEPRATIO);
	
	for (unsigned int i_img = 1; i_img <= 9; i_img++)
	{
		/* target image file input */
		sprintf_s(input_file_path_name, "%s%s%s", input_file_path, input_file_name[i_img], input_file_ext);
		Mat img_tar = imread(input_file_path_name);

		/* image distortion correction */
		undistort(img_tar, img_buf, cameraMatrix, distCoeffs);
		img_tar = img_buf.clone();



		/***************************
		***** Feature matching *****
		***************************/
		/* Detect the keypoints using SURF Detector, compute the descriptors */
		vector<KeyPoint> keypoints_tar;
		Mat descriptors_tar;
		detector->detectAndCompute(img_tar, noArray(), keypoints_tar, descriptors_tar);

		/* Matching descriptor vectors with a FLANN based matcher */
		/* Since SURF is a floating-point descriptor NORM_L2 is used */
		std::vector< std::vector<DMatch> > knn_matches;
		
		matcher->radiusMatch(descriptors_ref, descriptors_tar, knn_matches, 2);
		
		/* Filter matches using the Lowe's ratio test */
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < match_ratio_thresh  * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}


		/* Homography estimation with RANSAC */
		std::vector<Point2f> point2d_ref;
		std::vector<Point2f> point2d_tar;
		for (size_t i = 0; i < good_matches.size(); i++)
		{
			point2d_ref.push_back(keypoints_ref[good_matches[i].queryIdx].pt);
			point2d_tar.push_back(keypoints_tar[good_matches[i].trainIdx].pt);
		}

		cv::Mat inlier_mask;
		Mat H = findHomography(point2d_ref, point2d_tar, RANSAC, 3, inlier_mask);

		/* Inlier filtering */
		std::vector<Point2f> point2d_ref_inlier1;
		std::vector<Point2f> point2d_tar_inlier1;
		std::vector<Point2f> point2d_ref_inlier2;
		std::vector<Point2f> point2d_tar_inlier2;

		for (size_t i = 0; i < inlier_mask.rows; ++i)
		{
			if (inlier_mask.at<char>(i) > 0.0)
			{
				point2d_ref_inlier1.push_back(point2d_ref[i]);
				point2d_tar_inlier1.push_back(point2d_tar[i]);
			}
		}


		/* World point reconstruction */
		std::vector<Point3f> point3d_ref_inlier1;
		std::vector<Point3f> point3d_ref_inlier2;
		point3d_ref_inlier1.resize(point2d_ref_inlier1.size());
		for (size_t i = 0; i < point2d_ref_inlier1.size(); i++)
		{
			cmr_init.img2wld((float32_t*)&point2d_ref_inlier1[i], (float32_t*)&point3d_ref_inlier1[i]);
		}


		/*********************************
		***** Camera pose estimation *****
		*********************************/
		if (good_matches.size() < 4) // OpenCV requires solvePnPRANSAC to minimally have 4 set of points
		{
			printf("\n[CAM%d] Extrinsic parameters\n", i_img);
			printf("Fail to estimate camera pose due to data shortage\n", i_img);
			continue;
		}
		else
		{

			/* Estimate the pose using RANSAC approach */
			Mat inliers_idx;
			Mat rvec;    // output rotation vector
			Mat tvec;    // output translation vector
			Mat R_est;   // rotation matrix
			Mat R_inst;

			/* refine reference pose */
			if (i_img == 1)
			{
				solvePnPRansac(point3d_ref_inlier1, point2d_ref_inlier1, cameraMatrix, distCoeffs, rvec, tvec,
					useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers_idx, pnpMethod);

				Rodrigues(rvec, R_est);	 // converts Rotation Vector to Matrix

				R_est.convertTo(R_est, CV_32F);
				tvec.convertTo(tvec, CV_32F);
				R_inst = R_tf.t() * R_est;

				out_eul_angle[0] = rot2euld(R_inst);
				out_tl[0] = (-1.0f) * R_est.t() * tvec;

				printf("\n[CAM%d] Extrinsic parameters\n", 0);
				printf("Rotation angle deg (roll):  rx = [ %3.5f ]\n", -1.0f * out_eul_angle[0][0]);
				printf("Rotation angle deg (pitch): ry = [ %3.5f ]\n", -1.0f * out_eul_angle[0][1]);
				printf("Rotation angle deg (yaw):   rz = [ %3.5f ]\n", -1.0f * out_eul_angle[0][2]);
				printf("Translation vector:          t = [ %3.5f   %3.5f   %3.5f ]\n", out_tl[0].at<float>(0), out_tl[0].at<float>(1), out_tl[0].at<float>(2));
			}

			solvePnPRansac(point3d_ref_inlier1, point2d_tar_inlier1, cameraMatrix, distCoeffs, rvec, tvec,
				useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers_idx, pnpMethod);

			Rodrigues(rvec, R_est);	 // converts Rotation Vector to Matrix

			R_est.convertTo(R_est, CV_32F);
			tvec.convertTo(tvec, CV_32F);
			R_inst = R_tf.t() * R_est;

			out_eul_angle[i_img] = rot2euld(R_inst);
			out_tl[i_img] = (-1.0f) * R_est.t() * tvec;

			printf("\n[CAM%d] Extrinsic parameters\n", i_img);
			printf("Rotation angle deg (roll):  rx = [ %3.5f ]\n", -1.0f * out_eul_angle[i_img][0]);
			printf("Rotation angle deg (pitch): ry = [ %3.5f ]\n", -1.0f * out_eul_angle[i_img][1]);
			printf("Rotation angle deg (yaw):   rz = [ %3.5f ]\n", -1.0f * out_eul_angle[i_img][2]);
			printf("Translation vector:          t = [ %3.5f   %3.5f   %3.5f ]\n", out_tl[i_img].at<float>(0), out_tl[i_img].at<float>(1), out_tl[i_img].at<float>(2));


			// -- Step 4: Catch the inliers keypoints to draw
			for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
			{
				int n = inliers_idx.at<int>(inliers_index);					// i-inlier
				point2d_ref_inlier2.push_back(point2d_ref_inlier1[n]);       // add i-inlier to list
				point2d_tar_inlier2.push_back(point2d_tar_inlier1[n]);       // add i-inlier to list
				point3d_ref_inlier2.push_back(point3d_ref_inlier1[n]);
			}
		}



		/*********************************
		************* Output *************
		*********************************/

		/* 3d data buffering for 3d visualization */
		viewer->removeAllPointClouds();

		PointCloud<PointXYZ> cloud;
		cloud.width = point3d_ref_inlier2.size();
		cloud.height = 1;
		cloud.is_dense = false;
		cloud.points.resize(cloud.width * cloud.height);
		for (size_t i = 0; i < point3d_ref_inlier2.size(); i++)
		{

			cloud.points[i].x = point3d_ref_inlier2[i].x;
			cloud.points[i].y = point3d_ref_inlier2[i].y;
			cloud.points[i].z = point3d_ref_inlier2[i].z;
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		*ptr_cloud = cloud;
		viewer->addPointCloud<pcl::PointXYZ>(ptr_cloud, "ground_plane");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "ground_plane");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.95, 0.95, 0.95, "ground_plane");
		viewer->spinOnce();


		/* Draw matches */
		Mat img_matches;
		cv::drawMatches(img_ref, keypoints_ref, img_tar, keypoints_tar, good_matches, img_matches, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


		/* Emphasize reference inliers */
		for (size_t i = 0; i < point2d_ref_inlier2.size(); i++)
		{
			circle(img_matches, point2d_ref_inlier2[i], 2, Scalar(0, 255, 255), 3);
		}

		/* Emphasize target inliers */
		for (size_t i = 0; i < point2d_tar_inlier2.size(); i++)
		{
			Point2f tar_pt = point2d_tar_inlier2[i];
			tar_pt.x += image_size[0];
			circle(img_matches, tar_pt, 2, Scalar(0, 255, 255), 3);
		}


		/* world point reprojection in reference image */
		PerspectiveCamera cmr_ref(image_size, intrinsic_parameter, distortion_coefficient, (float32_t *)&out_eul_angle[0], (float32_t *)out_tl[0].data);
		std::vector<Point2f> point2d_ref_reproj;
		point2d_ref_reproj.resize(point3d_ref_inlier2.size());
		for (size_t i = 0; i < point3d_ref_inlier2.size(); i++)
		{
			cmr_ref.wld2img((float32_t*)&point3d_ref_inlier2[i], (float32_t*)&point2d_ref_reproj[i]);
			circle(img_matches, point2d_ref_reproj[i], 1, Scalar(0, 0, 255), 2);
		}

		/* world point reconstruction */
		PerspectiveCamera cmr_tar(image_size, intrinsic_parameter, distortion_coefficient, (float32_t *)&out_eul_angle[i_img], (float32_t *)out_tl[i_img].data);
		std::vector<Point2f> point2d_tar_reproj;
		point2d_tar_reproj.resize(point3d_ref_inlier2.size());
		for (size_t i = 0; i < point3d_ref_inlier2.size(); i++)
		{
			cmr_tar.wld2img((float32_t*)&point3d_ref_inlier2[i], (float32_t*)&point2d_tar_reproj[i]);
			Point2f tar_pt = point2d_tar_inlier2[i];
			tar_pt.x += image_size[0];
			circle(img_matches, tar_pt, 1, Scalar(0, 0, 255), 2);
		}

		/* Show roi of the reference image */
		for (size_t i = 0; i < int(sizeof(poly) / sizeof(poly[0])); i++)
		{
			if (i != int(sizeof(poly) / sizeof(poly[0])) - 1)
			{
				line(img_matches, poly[i], poly[i + 1], Scalar(0, 255, 255), 2);
			}
			else
			{
				line(img_matches, poly[i], poly[0], Scalar(0, 255, 255), 2);
			}
		}
		
		/* reference image area visualizaion on the target image */
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(0, 0);
		obj_corners[1] = Point2f((float)img_ref.cols, 0);
		obj_corners[2] = Point2f((float)img_ref.cols, (float)img_ref.rows);
		obj_corners[3] = Point2f(0, (float)img_ref.rows);
		std::vector<Point2f> scene_corners(4);
		cv::perspectiveTransform(obj_corners, scene_corners, H);

		cv::line(img_matches, scene_corners[0] + Point2f((float)img_ref.cols, 0),
			scene_corners[1] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[1] + Point2f((float)img_ref.cols, 0),
			scene_corners[2] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[2] + Point2f((float)img_ref.cols, 0),
			scene_corners[3] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[3] + Point2f((float)img_ref.cols, 0),
			scene_corners[0] + Point2f((float)img_ref.cols, 0), Scalar(0, 255, 0), 4);


		cv::imshow(win_name_matching, img_matches);
		cv::waitKey();

	}

	return 0;
}


static Mat Rx(float theta)
{
	Mat rot_x = (Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cos(theta), -sin(theta),
		0, sin(theta), cos(theta));

	return rot_x;
}

static Mat Ry(float theta)
{
	Mat rot_y = (Mat_<float>(3, 3) <<
		cos(theta), 0, sin(theta),
		0, 1, 0,
		-sin(theta), 0, cos(theta));

	return rot_y;
}

static Mat Rz(float theta)
{
	Mat rot_z = (Mat_<float>(3, 3) <<
		cos(theta), -sin(theta), 0,
		sin(theta), cos(theta), 0,
		0, 0, 1);

	return rot_z;
}

// Checks if a matrix is a valid rotation matrix.
static bool isRotationMatrix(Mat &R)
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;
}


static Vec3f rot2eul(Mat R)
{
	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
	bool singular = sy < 1e-6; // If

	float x, y, z;

	if (!singular)
	{
		x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
		y = atan2(-R.at<float>(2, 0), sy);
		z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
	}

	else
	{
		x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
		y = atan2(-R.at<float>(2, 0), sy);
		z = 0;
	}

	return Vec3f(x, y, z);
}


static Vec3f rot2euld(Mat R)
{
	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
	bool singular = sy < 1e-6; // If

	float x, y, z;

	if (!singular)
	{
		x = _rad2deg(atan2(R.at<float>(2, 1), R.at<float>(2, 2)));
		y = _rad2deg(atan2(-R.at<float>(2, 0), sy));
		z = _rad2deg(atan2(R.at<float>(1, 0), R.at<float>(0, 0)));
	}

	else
	{
		x = _rad2deg(atan2(-R.at<float>(1, 2), R.at<float>(1, 1)));
		y = _rad2deg(atan2(-R.at<float>(2, 0), sy));
		z = 0;
	}

	return Vec3f(x, y, z);
}


void initCamera(pcl::visualization::PCLVisualizer::Ptr& viewer)
{
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(-26.6f, -10.5f, 14.6f, 3.02f, -3.33f, 5.86f, 0.275f, 0.025f, 0.96f);
	viewer->addCoordinateSystem(1.0);
}
