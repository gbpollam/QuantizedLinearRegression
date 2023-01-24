//
// Created by giovanni on 23/01/23.
//

#ifndef LINEARREGRESSIONQUANTIZATIONCPP_INTLINEARREGRESSION_H
#define LINEARREGRESSIONQUANTIZATIONCPP_INTLINEARREGRESSION_H

#include <vector>
#include <iostream>


class IntLinearRegression {
private:
    //Weights:
    std::vector<int32_t> weights;

    //Bias:
    int32_t bias;

    //Scaling factor s_w
    double s_w;

    //Scaling factor s_x
    double s_x;

public:
    explicit IntLinearRegression(const std::vector<double> & d_weights, double bias, double s_x);

    void fit(const std::vector<std::vector<int32_t>> & data, std::vector<int32_t> y, int maxit, double alpha_lr);

    void print_weights() {
        std::cout << "Computed weights: " << std::endl;
        for (int32_t w: weights) {
            std::cout << w << std::endl;
        }
    }

    void print_weights(double scale) {
        std::cout << "Computed weights (rescaled with scale " << scale << "):" << std::endl;
        for (int32_t w: weights) {
            std::cout << w/scale << std::endl;
        }
    }
};


#endif //LINEARREGRESSIONQUANTIZATIONCPP_INTLINEARREGRESSION_H
