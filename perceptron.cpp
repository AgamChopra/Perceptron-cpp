#include <vector>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>

std::vector <long double> sigmoid (const std::vector <long double> m1) {
    
    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    std::vector <long double> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + std::exp(-m1[ i ]));
    }
    
    return output;
}

long double sigmoid(const long double x){
    return 1 / (1 + std::exp(-x));
}

long double MSELoss(std::vector<long double> y, std::vector<long double> yp){
    // MSE LOSS
    long double loss = 0;
    for(int i=0; i<yp.size(); i++){
         loss += pow(y[i] - yp[i], 2);
    }
    loss /= 2 * yp.size();
    return loss;
 }

std::vector<long double> const_sum(std::vector<long double> x, const long double a){
    for(int i=0; i<x.size(); i++){
         x[i] += a;
     }
    return x;
 }

long double accuracy(std::vector<long double> y, std::vector<long double> yp, float threshold){
    long double acc = 0.0;
    for(int i=0; i<y.size(); i++){
        acc += ((y[i]>threshold&&yp[i]>threshold)||(y[i]<threshold&&yp[i]<threshold)) ? 1.0 : 0.0;
    }
    return acc/y.size();
}

class perceptron {
    private:
        long double b = 0.0;
        std::vector<long double> w;
        long double grad_b = 0.0;
        std::vector<long double> grad_w; 
    public:
        perceptron(const int in_size = 1, const bool init_show = false){
            std::srand(time(NULL));
            for (int i=0; i < in_size; i++){
                    w.push_back((long double)rand()/(RAND_MAX));
                    grad_w.push_back(0);
                }
            if(init_show){
                std::cout << "perceptron created" << std::endl;
                show_values();
            }
        }

        void show_values(){
            for (int i=0; i < w.size(); i++){
                    std::cout << "w[" << i << "] = " << w[i] << ", ";
                }
                std::cout << "b = " << b << std::endl;
                for (int i=0; i < w.size(); i++){
                    std::cout << "Δw[" << i << "] = " << grad_w[i] << ", ";
                }
                std::cout << " Δb = " << grad_b << std::endl;
        }

        std::vector<long double> forward(const std::vector<std::vector<long double>> x){
            std::vector<long double> a;
            for(int i=0; i<x.size(); i++){
                long double temp = 0.0;
                for(int j=0; j<x[0].size(); j++){
                    temp += x[i][j]*w[j];
                }
                temp += b;
                a.push_back(sigmoid(temp));
            }
            return a;
        }

        void backward(const std::vector<long double> a, const std::vector<long double> y, const std::vector<long double> yp, const long double epsilon = 1E-9){
            long double dJ = (MSELoss(y, const_sum(yp, epsilon)) - MSELoss(y, const_sum(yp, -epsilon)))/(2*epsilon);
            for(int i=0; i<grad_w.size(); i++){
                grad_w[i] = dJ * ((sigmoid(a[i]*(w[i]+epsilon)+b)-sigmoid(a[i]*(w[i]-epsilon)+b))/(2*epsilon));
            }
            grad_b = dJ * (sigmoid(a[0]*w[0]+(b+epsilon))-sigmoid(a[0]*w[0]+(b-epsilon)))/(2*epsilon);
        }

        void update_param(const long double alpha = 1E-2){
            b = b - alpha * grad_b;
            for(int i = 0; i<w.size(); i++){
                w[i] = w[i] - alpha * grad_w[i];
            }
        }
};

int main()
{
    long double alpha = 0.001; // learning rate
    std::vector<std::vector<long double>> input = { {0.1, 0.2, 0.3}, {0.4, 0.7, 0.6}, {0.7, 0.8, 0.9}, {0.4, 0.1, 0.2}, {0.3, 0.0, 0.4}, {0.9, 0.9, 0.7}, {1.0, 0.5, 0.9}, {0.0, 0.6, 0.0} };
    std::vector<long double> y = { 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
    perceptron p1(input[0].size(), true);
    std::vector<long double> yp;
    yp = p1.forward(input);
    std::cout << "accuracy before training: " << accuracy(y,yp,0.5) << std::endl;
    for(int eps = 0; eps < 30000; eps++){
        yp = p1.forward(input);
        p1.backward(input[0], y,yp);
        p1.update_param(alpha);
        if(eps%10000 == 0){
            std::cout << "Epoch: " << eps <<" Loss:" << MSELoss(y, yp) << std::endl;
        }
    }
    yp = p1.forward(input);
    /*
    for(int i=0; i<input.size(); i++){
        std::cout << "Expectated: " << y[i] << " Predicted: " << yp[i] << std::endl;
    }
    */
    p1.show_values();
    std::cout << "accuracy after training: " << accuracy(y,yp,0.5) << std::endl;
}
/*
***EXPECTED OUTPUT***
perceptron created
w[0] = 0.970794, w[1] = 0.48851, w[2] = 0.764397, b = 0
Δw[0] = 0, Δw[1] = 0, Δw[2] = 0,  Δb = 0
accuracy before training: 0.5
Epoch: 0 Loss:0.102175
Epoch: 10000 Loss:0.0802934
Epoch: 20000 Loss:0.0755043
w[0] = 0.897934, w[1] = 0.343088, w[2] = 0.543771, b = -0.728602
Δw[0] = 0.000690478, Δw[1] = 0.00137181, Δw[2] = 0.00211631,  Δb = 0.00690478
accuracy after training: 1
*/