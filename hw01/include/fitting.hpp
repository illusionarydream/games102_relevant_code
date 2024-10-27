#ifndef FITTING_HPP
#define FITTING_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <matplotlibcpp.h>

#define vector_d std::vector<double>
#define vector_i std::vector<int>
#define EMatrixXd Eigen::MatrixXd
#define EVectorXd Eigen::VectorXd

namespace plt = matplotlibcpp;

class Fitting {
   private:
    // * method 1: use MSE to fit a polynomial
    EVectorXd W_MSE;

    // * method 2: use RBF to fit a polynomial
    EVectorXd W_RBF;
    double sigma;

    // * all
    vector_d data_x;
    vector_d data_y;
    int method;

   public:
    Fitting() {}

   public:
    // * method 1: use MSE to fit a polynomial
    bool polygon_fitting(const vector_d &x, const vector_d &y, const int &degree) {
        // if the size of x and y are not equal, return false
        if (x.size() != y.size()) {
            return false;
        }
        // if the degree is less than 0 or greater than the size of x, return false
        if (degree < 0 || degree >= x.size()) {
            return false;
        }
        // set the method
        method = 1;
        // get the number of data points and the number of coefficients
        int n = x.size();
        int m = degree + 1;
        data_x = x;
        data_y = y;
        // min |A * w - y|
        // A: n * degree matrix
        EMatrixXd A(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                A(i, j) = std::pow(x[i], j);
            }
        }
        // y = [y0 y1 y2 ... y(n-1)]^T
        EVectorXd Y(n);
        for (int i = 0; i < n; i++) {
            Y(i) = y[i];
        }
        // w = [w0 w1 w2 ... w(degree-1)]^T
        EVectorXd W = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
        // store the result
        W_MSE = W;
        return true;
    }
    // * method 2: use RBF to fit a polynomial
    bool RBF_fitting(const vector_d &x, const vector_d &y, const double &sigma = 1) {
        // if the size of x and y are not equal, return false
        if (x.size() != y.size()) {
            return false;
        }
        // set the method
        method = 2;
        // get the number of data points and the number of coefficients
        int n = x.size();
        data_x = x;
        data_y = y;
        // min |A * w - y|
        // A: n * n matrix
        EMatrixXd A(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A(i, j) = std::exp(-std::pow(x[i] - x[j], 2) / (2 * std::pow(sigma, 2)));
            }
        }
        // y = [y0 y1 y2 ... y(n-1)]^T
        EVectorXd Y(n);
        for (int i = 0; i < n; i++) {
            Y(i) = y[i];
        }
        // w = [w0 w1 w2 ... w(n-1)]^T
        EVectorXd W = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
        // store the result
        W_RBF = W;

        return true;
    }
    // * draw the fitting curve
    void draw(const double &left, const double &right, const int &bins = 100) {
        vector_d curve_x;
        vector_d curve_y;
        if (method == 1) {
            // if the size of W_MSE is 0, return
            if (W_MSE.size() == 0) {
                return;
            }
            // get the coefficients of the polynomial
            vector_d W;
            for (int i = 0; i < W_MSE.size(); i++) {
                W.push_back(W_MSE(i));
            }
            // draw the fitting curve
            for (int i = 0; i < bins; i++) {
                double x = left + (right - left) / bins * i;
                double y = 0;
                for (int j = 0; j < W.size(); j++) {
                    y += W[j] * std::pow(x, j);
                }
                curve_x.push_back(x);
                curve_y.push_back(y);
            }
        }
        if (method == 2) {
            // if the size of W_RBF is 0, return
            if (W_RBF.size() == 0) {
                return;
            }
            // get the coefficients of the polynomial
            vector_d W;
            for (int i = 0; i < W_RBF.size(); i++) {
                W.push_back(W_RBF(i));
            }
            // draw the fitting curve
            for (int i = 0; i < bins; i++) {
                double x = left + (right - left) / bins * i;
                double y = 0;
                for (int j = 0; j < W.size(); j++) {
                    y += W[j] * std::exp(-std::pow(x - data_x[j], 2) / (2 * std::pow(sigma, 2)));
                }
                curve_x.push_back(x);
                curve_y.push_back(y);
            }
        }
        // draw the data points
        plt::plot(data_x, data_y, "ro");
        // draw the fitting curve
        plt::plot(curve_x, curve_y, "b");
        plt::show();
        return;
    }
};

#endif  // FITTING_HPP