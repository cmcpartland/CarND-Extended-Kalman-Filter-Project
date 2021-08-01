#include "kalman_filter.h"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; // Current state
  P_ = P_in; // Covariance of the state estimation
  F_ = F_in; // State transition matrix - describes the dynamical equations governing the system
  H_ = H_in; // Observation matrix - converts system state into outputs via matrix transformation
  R_ = R_in; // Measurement uncertainty - covariance matrix of the measurement
  Q_ = Q_in; // Process noise uncertainty
}

void KalmanFilter::Predict() {
  /**
  Predict the state
  */
	
  // State extrapolation equation - F_ is applied to the current value of x_ to give the next predicted state of x_
  x_ = F_ * x_; 
	
  MatrixXd Ft = F_.transpose();
	
  // Covariance extrapolation equation - describes how the covariance of the estimation extrapolates to the next cycle
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  Update the state based on a new measurement
  */
	
  VectorXd z_pred = H_ * x_; // Apply the observation matrix to the current state
  VectorXd y = z - z_pred; // Difference between the measurement (z) and the predicted measurement we'd expect given the current state
  Estimate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // update the state by using Extended Kalman Filter equations
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  // cylindrical coordinates
  float rho = sqrt(px*px + py*py);
  float theta = atan2(py,px);
  float rho_dot = (px*vx + py*vy)/rho;
  
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
  
  VectorXd y = z-h;

  //Normalize the phi value to be between -pi and pi
  float phi = y[1];
  while (phi > M_PI || phi < -M_PI) {
    if (phi > M_PI) {
      phi -= M_PI;
    } 
	else {
      phi += M_PI;
    }
  }
  y[1] = phi;
  
  Estimate(y);  
}

void KalmanFilter::Estimate(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y); // update to the state based on filtering equation
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; // Update to the covariance matrix 
}
