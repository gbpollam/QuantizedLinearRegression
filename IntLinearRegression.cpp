//
// Created by giovanni on 23/01/23.
//

#include "IntLinearRegression.h"

IntLinearRegression::IntLinearRegression(const std::vector<double> & d_weights, double passed_bias, double passed_s_x){
    this->s_x = passed_s_x;
    // Compute the scaling factor for the weights
    s_w = 0;
    for (double w : d_weights){
        if (std::abs(w) > s_w){
            s_w = w;
        }
    }
    // Save the quantized weights (as if they were int8, but in int32 to avoid converting in matrix multiplication)
    for (double w : d_weights){
        weights.push_back((int32_t)(w*127.0/s_w));
    }
    // Quantize the bias (int32)
    this->bias = (int32_t)(passed_bias*2147483648.0/(s_w*s_x));
}

void IntLinearRegression::fit(const std::vector<std::vector<int32_t>> & data, std::vector<int32_t> y, int maxit, double alpha_lr){
    int it=0;
    uint len = weights.size();
    uint data_len = data.size();
    while(it<maxit){
        it++;
        double learning_rate = alpha_lr/it;
        // int32_t sum = 0;
        std::vector<int32_t> y_hats;
        for (std::vector<int32_t> x : data){
            int32_t y_hat = 0;
            for(int j=0; j<len; j++){
                y_hat += (x[j] * weights[j]);
            }
            y_hat += bias;
            y_hats.push_back(y_hat);
        }
        std::vector<int32_t> deltas(len, 0);
        int32_t delta_bias = 0;
        for(int i=0; i<data_len; i++){
            for(int j=0; j<len; j++){
                deltas[j] += (int8_t)((-2)*data[i][j]*(y[i] - y_hats[i])/data_len);
            }
            delta_bias += (int8_t)((-2)*(y[i] - y_hats[i])/data_len);
        }

        for(int j=0; j<len; j++){
            weights[j] -= (int8_t)(learning_rate*deltas[j]/(s_w*s_w));
        }
        bias -= (int8_t)(learning_rate* delta_bias/(s_w*s_w*s_x*s_x));
    }
}