#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;
  previous_dt_ = 0;

  /*
   * INITIALIZING MATRICES
   */
  R_laser_ = MatrixXd(2, 2); // measurement uncertainty (covariance) of the laser measurement
  R_radar_ = MatrixXd(3, 3); // measurement uncertainty (covariance) of the radar measurement
  H_laser_ = MatrixXd(2, 4); // observation matrix of the laser
  Hj_ = MatrixXd(3, 4);      // Jacobian matrix
	
  // ekf_ is a Kalman filter instance (instantiated in the linear)
  ekf_.P_ = MatrixXd(4,4); // covariance matrix of the estimation
  ekf_.x_ = VectorXd(4); // state vector
  ekf_.F_ = MatrixXd(4,4); // state transition matrix

  R_laser_ << 0.0225, 0,
              0, 0.0225;

  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
 
  // Laser output is a measurement of position in cartesian coordinates (x,y)
  H_laser_ << 1, 0, 0, 0,
	      0, 1, 0, 0;

  Hj_ << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
    */
    // Initialize state vector with initial position measurement and zero velocity
    cout << "EKF: " << endl;
	ekf_.x_ << 0,0,0,0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      float px = rho*cos(phi);
      float py = rho*sin(phi);
	    
      // Derive vx and by by taking derivatives of px and py. Since we assume d(phi)/dt = 0 (inertial), the second term in derivative goes to 0
      float vx = rho_dot*cos(phi);
      float vy = rho_dot*sin(phi);
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
	
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  
  // Only update F_ and Q_ if dt has changed, otherwise these matrices are the same as in the previous step
  if (dt != previous_dt_) {
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;
    float dt_4o4 = dt_4/4.0;
    float dt_3o2 = dt_3/2.0;
  
    //Modify the F matrix so that the time is integrated
    ekf_.F_ << 1, 0, dt, 0,
	       0, 1, 0, dt,
	       0, 0, 1, 0,
	       0, 0, 0, 1;
			 
    // define and set process noise for accelerations in x and y
    float noise_ax = 9.;
    float noise_ay = 9.;
  
    // define process noise uncertainty (discrete time model)
    ekf_.Q_ = MatrixXd(4,4);
    ekf_.Q_ <<  dt_4o4*noise_ax, 0, dt_3o2*noise_ax, 0,
			    0, dt_4*noise_ay, 0, dt_3o2*noise_ay,
			    dt_3o2*noise_ax, 0, dt_2*noise_ax, 0,
			    0, dt_3o2*noise_ay, 0, dt_2*noise_ay;
  }

  ekf_.Predict();
  
  // Update timestamp values
  previous_timestamp_ = measurement_pack.timestamp_;
  previous_dt_ = dt;
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else {
    // Laser updates
    ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
