/*************************************************************************************
* -----------------------------------   include   ----------------------------------- *
**************************************************************************************/

#include "util_calibration.h"


/**************************************************************************************
* ------------------------------   global variables   ------------------------------- *
**************************************************************************************/



/**************************************************************************************
-------------------------------   function prototypes   -------------------------------
**************************************************************************************/


PerspectiveCamera::PerspectiveCamera(float32_t image_size[2],
	float32_t intrinsic_parameter[5],
	float32_t distortion_coefficient[5], 
	float32_t rvec[3],
	float32_t tvec[3])
{
	image_size_[0] = image_size[0];
	image_size_[1] = image_size[1];

	fx_ = intrinsic_parameter[0];
	fy_ = intrinsic_parameter[1];
	cx_ = intrinsic_parameter[2];
	cy_ = intrinsic_parameter[3];
	skew_ = intrinsic_parameter[4];

	k1_ = distortion_coefficient[0];
	k2_ = distortion_coefficient[1];
	k3_ = distortion_coefficient[2];
	p1_ = distortion_coefficient[3];
	p2_ = distortion_coefficient[4];

	rvec_[0] = rvec[0];
	rvec_[1] = rvec[1];
	rvec_[2] = rvec[2];

	tvec_[0] = tvec[0];
	tvec_[1] = tvec[1];
	tvec_[2] = tvec[2];

	InitCameraMatrix();
}

void PerspectiveCamera::InitCameraMatrix(void)
{
	float32_t temp3x3[3][3];
	float32_t temp3x4[3][4];
	float32_t temp3x1[3];
	float32_t R_tf[3][3];
	float32_t R_inst[3][3];
	float32_t Rx[3][3];
	float32_t Ry[3][3];
	float32_t Rz[3][3];


	/***************************
	***** Intrinsic matrix *****
	***************************/
	K_[0][0] = fx_;
	K_[0][1] = skew_;
	K_[0][2] = cx_;
	K_[1][0] = 0.0f;
	K_[1][1] = fy_;
	K_[1][2] = cy_;
	K_[2][0] = 0.0f;
	K_[2][1] = 0.0f;
	K_[2][2] = 1.0f;


	/***************************
	***** Extrinsic matrix *****
	***************************/

	_rot_z(_deg2rad(-90.0f), Rz);
	_rot_y(_deg2rad(-90.0f), Ry);
	_rot_x(_deg2rad(180.0f), Rx);

	_mat_mul_3x3and3x3(Rz, Ry, temp3x3);
	_mat_mul_3x3and3x3(temp3x3, Rx, R_tf);

	_rot_z(_deg2rad(rvec_[2]), Rz); /* yaw */
	_rot_y(_deg2rad(rvec_[1]), Ry); /* pitch */
	_rot_x(_deg2rad(rvec_[0]), Rx); /* roll */

	_mat_mul_3x3and3x3(Rz, Ry, temp3x3);
	_mat_mul_3x3and3x3(temp3x3, Rx, R_inst);

	_mat_mul_3x3and3x3(R_tf, R_inst, R_);


	/* Translation vector from world to camera in camera coordinates */
	temp3x1[0] = -tvec_[0];
	temp3x1[1] = -tvec_[1];
	temp3x1[2] = -tvec_[2];
	_mat_mul_3x3and3x1(R_, temp3x1, t_);


	/****************************
	***** projection matrix *****
	****************************/
	_mat_concat_3x3_3x1(R_, t_, temp3x4);
	_mat_mul_3x3and3x4(K_, temp3x4, P_);


	/************************************************************
	***** Homography matrix for Inverse perspective mapping *****
	*************************************************************/
	/* Discard z component */
	temp3x3[0][0] = P_[0][0];
	temp3x3[0][1] = P_[0][1];
	temp3x3[0][2] = P_[0][3];
	temp3x3[1][0] = P_[1][0];
	temp3x3[1][1] = P_[1][1];
	temp3x3[1][2] = P_[1][3];
	temp3x3[2][0] = P_[2][0];
	temp3x3[2][1] = P_[2][1];
	temp3x3[2][2] = P_[2][3];

	MatInv3x3(temp3x3, H_);

}



/* image to world projection on ground plane using homography matrix */
void PerspectiveCamera::img2wld(float32_t img[2], float32_t wld[3])
{
	float32_t img_homo[3] = { img[0], img[1], 1 };

	image_undistort(img_homo, img_homo);

	_mat_mul_3x3and3x1(H_, img_homo, wld);
	wld[0] = wld[0] / (wld[2] + _small_value);
	wld[1] = wld[1] / (wld[2] + _small_value);
	wld[2] = 0;
}


/* world to image projection using pojection matrix */
void PerspectiveCamera::wld2img(float32_t wld[3], float32_t img[2])
{
	float32_t wld_homo[4] = { wld[0], wld[1], wld[2], 1 };
	float32_t img_homo[3];

	_mat_mul_3x4and4x1(P_, wld_homo, img_homo);
	img[0] = img_homo[0] / (img_homo[2] + _small_value);
	img[1] = img_homo[1] / (img_homo[2] + _small_value);

	image_distort(img, img);
}


void MatInv3x3(float32_t src[][3], float32_t dst[][3])
{
	float32_t det;

	dst[0][0] = (src[1][1] * src[2][2] - src[1][2] * src[2][1]);
	dst[0][1] = (src[0][1] * src[2][2] - src[0][2] * src[2][1]) * (-1);
	dst[0][2] = (src[0][1] * src[1][2] - src[0][2] * src[1][1]);
	dst[1][0] = (src[1][0] * src[2][2] - src[1][2] * src[2][0]) * (-1);
	dst[1][1] = (src[0][0] * src[2][2] - src[0][2] * src[2][0]);
	dst[1][2] = (src[0][0] * src[1][2] - src[0][2] * src[1][0]) * (-1);
	dst[2][0] = (src[1][0] * src[2][1] - src[1][1] * src[2][0]);
	dst[2][1] = (src[0][0] * src[2][1] - src[0][1] * src[2][0]) * (-1);
	dst[2][2] = (src[0][0] * src[1][1] - src[0][1] * src[1][0]);

	det = (src[0][0] * dst[0][0]) + (src[0][1] * dst[1][0]) + (src[0][2] * dst[2][0]);

	dst[0][0] /= (det + _small_value);
	dst[0][1] /= (det + _small_value);
	dst[0][2] /= (det + _small_value);
	dst[1][0] /= (det + _small_value);
	dst[1][1] /= (det + _small_value);
	dst[1][2] /= (det + _small_value);
	dst[2][0] /= (det + _small_value);
	dst[2][1] /= (det + _small_value);
	dst[2][2] /= (det + _small_value);
}


void PerspectiveCamera::image_undistort(float32_t src_pt[2], float32_t dst_pt[2])
{
	int16_t iter;
	int16_t nIter;
	float32_t pn[2];
	float32_t pn0[2];
	float32_t pd[2];
	float32_t err[2];

	nIter = 3;

	image_normalize(src_pt, pn);

	pn0[0] = pn[0];
	pn0[1] = pn[1];

	for (iter = 0; iter < nIter; iter++)
	{
		image_distort_normal(pn, pd);
		err[0] = pd[0] - pn0[0];
		err[1] = pd[1] - pn0[1];
		pn[0] -= err[0];
		pn[1] -= err[1];
	}

	image_denormalize(pn, dst_pt);
}


void PerspectiveCamera::image_distort(float32_t src_pt[2], float32_t dst_pt[2])
{
	float32_t pn[2];
	float32_t pd[2];

	image_normalize(src_pt, pn);

	image_distort_normal(pn, pd);

	image_denormalize(pd, dst_pt);
}


void PerspectiveCamera::image_normalize(float32_t pi[2], float32_t pn[2])
{
	pn[1] = (pi[1] - cy_) / fy_;
	pn[0] = (pi[0] - cx_ - skew_ * pn[1]) / fx_;
}


void PerspectiveCamera::image_denormalize(float32_t pn[2], float32_t pi[2])
{
	pi[0] = fx_ * pn[0] + cx_ + skew_ * pn[1];
	pi[1] = fy_ * pn[1] + cy_;
}


void PerspectiveCamera::image_distort_normal(float32_t pn[2], float32_t pd[2])
{
	float32_t r2;
	float32_t alpha;
	float32_t dxTangential;
	float32_t dyTangential;


	/* compute radial distortion */
	r2 = pn[0] * pn[0] + pn[1] * pn[1];

	alpha = k1_ * (r2)
		  + k2_ * (r2 * r2)
		  + k3_ * (r2 * r2 * r2);

	/* compute tangential distortion */
	dxTangential = 2 * p1_ * pn[0] * pn[1] + p2_ * (r2 + 2 * pn[0] * pn[0]);
	dyTangential = p1_ * (r2 + 2 * pn[1] * pn[1]) + 2 * p2_ * pn[0] * pn[1];

	pd[0] = pn[0] + pn[0] * alpha + dxTangential;
	pd[1] = pn[1] + pn[1] * alpha + dyTangential;
}
