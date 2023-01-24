//
// Created by giovanni on 11/01/23.
//

#ifndef LINEARREGRESSIONQUANTIZATIONCPP_QLINEARREGRESSION_H
#define LINEARREGRESSIONQUANTIZATIONCPP_QLINEARREGRESSION_H

#include <vector>
#include <iostream>

template<typename dwtype, typename biastype> class QLinearRegression {
private:
    //Weights:
    std::vector<dwtype> weights;

    //Bias:
    biastype bias;

public:
    explicit QLinearRegression(int num_weights){
        for (int i=0; i<num_weights;i++){
            this->weights.push_back(1);
        }
        bias = 1;
    }

    void fit(const std::vector<std::vector<dwtype>> & data, std::vector<biastype> y, int maxit, double alpha_lr){
        int it=0;
        uint len = weights.size();
        uint data_len = data.size();
        while(it<maxit){
            it++;
            double learning_rate = alpha_lr/it;
            biastype sum = 0;
            std::vector<biastype> y_hats;
            for (std::vector<dwtype> x : data){
                biastype y_hat = 0;
                for(int j=0; j<len; j++){
                    y_hat += (x[j] * weights[j]);
                }
                y_hat += bias;
                y_hats.push_back(y_hat);
            }
            std::vector<dwtype> deltas(len, 0);
            biastype delta_bias = 0;
            for(uint i=0; i<data_len; i++){
                for(int j=0; j<len; j++){
                    deltas[j] += (-2)*data[i][j]*(y[i] - y_hats[i])/data_len;
                }
                delta_bias += (-2)*(y[i] - y_hats[i])/data_len;
            }

            for(int j=0; j<len; j++){
                weights[j] -= learning_rate*deltas[j];
            }
            // bias -= learning_rate*delta_bias;
        }
    }

    void print_weights() {
        std::cout << "Computed weights: " << std::endl;
        for (dwtype w: weights) {
            std::cout << w << std::endl;
        }
    }

    void print_weights(double scale) {
        std::cout << "Computed weights: " << std::endl;
        for (dwtype w: weights) {
            std::cout << w/scale << std::endl;
        }
    }

    std::vector<dwtype> return_weights(){
        return weights;
    }

    biastype return_bias(){
        return bias;
    }
};


#endif //LINEARREGRESSIONQUANTIZATIONCPP_QLINEARREGRESSION_H
