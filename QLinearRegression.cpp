//
// Created by giovanni on 11/01/23.
//

#include "QLinearRegression.h"
#include <iostream>

/*
template<typename dwtype, typename biastype>
void QLinearRegression<dwtype, biastype>::fit(const std::vector<std::vector<dwtype>> & data, std::vector<biastype> y,
                                              int maxit,
                                              double alpha_lr) {
    int it=0;
    int len = weights.size();
    while(it<maxit){
        it++;
        double learning_rate = alpha_lr/double(it);
        biastype sum = 0;
        std::vector<biastype> y_hats;
        for (dwtype x : data){
            biastype y_hat = 0;
            for(int i=0; i<len; i++){
                y_hat+=x[i] * weights[i];
            }
            y_hat += bias;
            y_hats.push_back(y_hat);
        }
        std::vector<dwtype> deltas(len, 0);
        for(int i=0; i<data.size(); i++){
            for(int j=0; j<len; j++){
                deltas[j] += (-2)*data[i][j]*(y[i] - y_hats[i]);
            }
        }

        for(int j=0; j<len; j++){
            weights[j] += learning_rate*deltas[j];
        }
    }
}
*/

/*
template<typename dwtype, typename biastype>
void QLinearRegression<dwtype, biastype>::print_weights() {
    std::cout << "Computed weights: " << std::endl;
    for(dwtype w : weights){
        std::cout << w << std::endl;
    }
}
*/
