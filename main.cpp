#include <iostream>
#include <random>
#include <cstdlib>

#include "QLinearRegression.h"
#include "IntLinearRegression.h"

int main() {
    int num_data = 100;
    int num_attr = 3;
    //rng
    unsigned seed = 42;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-20,20);
    std::normal_distribution<double> n_distribution(0.0,3.0);
    //Set the real weights:
    std::vector<double> r_weigths;
    for (int j=0; j<num_attr;j++){
        r_weigths.push_back(j*j*j * n_distribution(generator));
    }
    //Generate the dataset
    std::vector<double> placeholder{};
    std::vector<std::vector<double>> data(num_data, placeholder);
    std::vector<double> y(num_data, 0);
    for (int i=0; i<num_data;i++){
        double y_sum = 0;
        for (int j=0; j<num_attr;j++){
            data[i].push_back(distribution(generator));
            y_sum += data[i][j] * r_weigths[j];
        }
        y[i] = y_sum + n_distribution(generator);
    }

    QLinearRegression<double, double> regressor(3);

    // Save weights and biases after 3 iterations (pass them to IntLinearRegression later)
    regressor.fit(data, y, 3, 0.01);
    std::vector<double> saved_weights = regressor.return_weights();
    double saved_bias = regressor.return_bias();

    // Complete the regression
    regressor.fit(data, y, 3, 0.01);

    std::cout << "True weights: " << std::endl;
    for(double w : r_weigths){
        std::cout << w << std::endl;
    }

    regressor.print_weights();

    std::cout << "Regression with int: " << std::endl;
    std::vector<int32_t> int_placeholder{};
    std::vector<std::vector<int32_t>> int_data(num_data, int_placeholder);

    // Find the max and cast data to int32_t (but treat them as if they were int8_t)
    double s_x = 0;
    for (int i=0; i<num_data;i++) {
        for (int j=0; j<num_attr;j++){
            if (std::abs(data[i][j]) > s_x){
                s_x = data[i][j];
            }
        }
    }

    // Quantize the data
    for (int i=0; i<num_data;i++) {
        for (int j=0; j<num_attr;j++){
            int_data[i].push_back((int32_t)(data[i][j]*127.0/s_x));
            //std::cout << "Double data: " << data[i][j] << std::endl;
            //printf("Int data: %d \n", int_data[i][j]);
        }
    }

    // Find the max and cast targets to int32_t
    std::vector<int32_t> int_y{};

    double s_y=0;

    for (int i=0; i<num_data;i++){
        if (std::abs(y[i]) > s_y){
            s_y = y[i];
        }
    }

    for (int i=0; i<num_data;i++){
        int_y.push_back((int32_t)(y[i]*2147483648.0/s_x));
    }

    // Initialize the IntLinearRegression passing the saved weights and biases
    IntLinearRegression int_regressor(saved_weights, saved_bias, s_x);

    int_regressor.fit(int_data, int_y, 1000, 0.001);

    std::cout << "True weights: " << std::endl;
    for(double w : r_weigths){
        std::cout << w << std::endl;
    }

    int_regressor.print_weights();
    int_regressor.print_weights(127.0);
}
