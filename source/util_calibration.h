#ifndef UTIL_CALIBRATION_H
#define UTIL_CALIBRATION_H

/*************************************************************************************
* -----------------------------------   include   ----------------------------------- *
**************************************************************************************/

#include <math.h>
#include "type.h"

/**************************************************************************************
* ------------------------------   global variables   ------------------------------- *
**************************************************************************************/



/**************************************************************************************
-------------------------------   function prototypes   -------------------------------
**************************************************************************************/



/**************************************************************************************
* -----------------------------------   define   ------------------------------------ *
**************************************************************************************/

extern void MatInv3x3(float32_t src[][3], float32_t dst[][3]);

class PerspectiveCamera
{
public:
	explicit PerspectiveCamera(float32_t image_size[2],
		float32_t intrinsic_parameter[5],
		float32_t distortion_coefficient[5],
		float32_t rvec[3],
		float32_t tvec[3]);
	
	void InitCameraMatrix(void);

	/* image to world projection on ground plane using homography matrix */
	void img2wld(float32_t img[2], float32_t wld[3]);

	/* world to image projection using pojection matrix */
	void wld2img(float32_t wld[4], float32_t img[3]);



	/* Image size: width, height */
	int32_t image_size_[2];

	/* Intrinsic parameter */
	float32_t fx_;
	float32_t fy_;
	float32_t cx_;
	float32_t cy_;
	float32_t skew_;

	/* Radial distrotion coefficient */
	float32_t k1_;
	float32_t k2_;
	float32_t k3_;

	/* Tangential distortion coefficient */
	float32_t p1_;
	float32_t p2_;

	/* Extrinsic parameter */
	float32_t rvec_[3];
	float32_t tvec_[3];

	/* Extrinsic */
	float32_t R_[3][3];	/* Roatation matrix */
	float32_t t_[3];		/* Translation vector */

	/* Intrinsic matrix */
	float32_t K_[3][3];

	/* Projection matrix (wld2img) */
	float32_t P_[3][4];

	/* Homography matrix (img2wld) */
	float32_t H_[3][3];

private:

	void image_undistort(float32_t src_pt[2], float32_t dst_pt[2]);
	void image_distort(float32_t src_pt[2], float32_t dst_pt[2]);
	void image_normalize(float32_t pi[2], float32_t pn[2]);
	void image_denormalize(float32_t pn[2], float32_t pi[2]);
	void image_distort_normal(float32_t pn[2], float32_t pd[2]);
};



/**************************************************************************************
--------------------------------------   Macro   --------------------------------------
**************************************************************************************/

#define _mat_mul_3x3and3x1(a, b, c)			c[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2];\
											c[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2];\
											c[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2];\

#define _mat_mul_3x3and3x3(a, b, c)			c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0];\
											c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1];\
											c[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2];\
											c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0];\
											c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1];\
											c[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2];\
											c[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0];\
											c[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1];\
											c[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2];\

#define _mat_mul_3x3and3x4(a, b, c)			c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0];\
											c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1];\
											c[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2];\
											c[0][3] = a[0][0]*b[0][3] + a[0][1]*b[1][3] + a[0][2]*b[2][3];\
											c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0];\
											c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1];\
											c[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2];\
											c[1][3] = a[1][0]*b[0][3] + a[1][1]*b[1][3] + a[1][2]*b[2][3];\
											c[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0];\
											c[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1];\
											c[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2];\
											c[2][3] = a[2][0]*b[0][3] + a[2][1]*b[1][3] + a[2][2]*b[2][3];\

#define _mat_mul_3x4and4x1(a, b, c)			c[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2] + a[0][3]*b[3];\
											c[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2] + a[1][3]*b[3];\
											c[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2] + a[2][3]*b[3];\

#define _mat_concat_3x3_3x1(a, b, c)		c[0][0] = a[0][0];\
											c[0][1] = a[0][1];\
											c[0][2] = a[0][2];\
											c[0][3] = b[0];\
											c[1][0] = a[1][0];\
											c[1][1] = a[1][1];\
											c[1][2] = a[1][2];\
											c[1][3] = b[1]; \
											c[2][0] = a[2][0];\
											c[2][1] = a[2][1];\
											c[2][2] = a[2][2];\
											c[2][3] = b[2];\

#define _rot_x(rad, rot)					rot[0][0] = 1.0f;\
											rot[0][1] = 0.0f;\
											rot[0][2] = 0.0f;\
											rot[1][0] = 0.0f;\
											rot[1][1] = cosf(rad);\
											rot[1][2] = -sinf(rad);\
											rot[2][0] = 0.0f;\
											rot[2][1] = sinf(rad);\
											rot[2][2] = cosf(rad);\

#define _rot_y(rad, rot)					rot[0][0] = cosf(rad);\
											rot[0][1] = 0.0f;\
											rot[0][2] = sinf(rad);\
											rot[1][0] = 0.0f;\
											rot[1][1] = 1.0f;\
											rot[1][2] = 0.0f;\
											rot[2][0] = -sinf(rad);\
											rot[2][1] = 0.0f;\
											rot[2][2] = cosf(rad);\

#define _rot_z(rad, rot)					rot[0][0] = cosf(rad);\
											rot[0][1] = -sinf(rad);\
											rot[0][2] = 0.0f;\
											rot[1][0] = sinf(rad);\
											rot[1][1] = cosf(rad);\
											rot[1][2] = 0.0f;\
											rot[2][0] = 0.0f;\
											rot[2][1] = 0.0f;\
											rot[2][2] = 1.0f;\

#define _deg2rad(a)							( (a) * 0.017453292519943f )
#define _rad2deg(a)							( (a) * 57.295779513082323f )

#endif
