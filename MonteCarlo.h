#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <random>
#include <vector>
#include "Autodiff.h"
#include "MathUtils.h"

template<typename T>
T europeanCallMC(T S0, T sigma, T K, T T_expiry, T r, 
                 int num_paths, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    T sum_payoff = S0 * 0.0; // let it inherit type
    
    for (int i = 0; i < num_paths; i++) {
        double Z = norm(gen);
        
        T S_T = S0 * exp((r - 0.5*sigma*sigma)*T_expiry + sigma*sqrt(T_expiry)*Z);
        
        T payoff = max(S_T - K, 0.0);
        
        sum_payoff = sum_payoff + payoff;
    }
    
    T price = exp(-r * T_expiry) * sum_payoff / static_cast<double>(num_paths);
    
    return price;
}

template<typename T>
T asianCallMC(T S0, T sigma, T K, T T_expiry, T r, 
              int num_paths, int num_steps, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    
    T dt = T_expiry / num_steps;  // time step size
    T sqrt_dt = sqrt(dt);
    
    T sum_payoff = S0 * 0.0;
    
    for (int path = 0; path < num_paths; path++) {
        T S = S0;
        T path_sum = S0;
        
        // simulate the path
        for (int step = 0; step < num_steps; step++) {
            double Z = norm(gen);
            
            S = S * exp((r - 0.5*sigma*sigma)*dt + sigma*sqrt_dt*Z);
            
            path_sum = path_sum + S;
        }
        
        // average price over path
        T avg_price = path_sum / static_cast<double>(num_steps + 1);
        
        T payoff = max(avg_price - K, 0.0);
        
        sum_payoff = sum_payoff + payoff;
    }
    
    T price = exp(-r * T_expiry) * sum_payoff / static_cast<double>(num_paths);
    
    return price;
}

struct GreeksResult {
    double price;
    double delta;
    double vega;
    double rho;
    double theta;
    double strike_sens;  // ∂V/∂K
};

GreeksResult asianCallMC_Batched(double S0, double sigma, double K, 
                                  double T_expiry, double r,
                                  int num_paths, int num_steps,
                                  int batch_size = 1000, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    double sum_price = 0.0;
    double sum_delta = 0.0;
    double sum_vega = 0.0;
    double sum_rho = 0.0;
    double sum_theta = 0.0;
    double sum_strike_sens = 0.0;
    
    int num_batches = (num_paths + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int current_batch_size = std::min(batch_size, num_paths - batch * batch_size);
        
        // fresh tape for batch
        auto tape = std::make_shared<Tape>();
        Var S(S0, tape);
        Var sig(sigma, tape);
        Var K_var(K, tape);
        Var T_var(T_expiry, tape);
        Var r_var(r, tape);
        
        Var dt = T_var / static_cast<double>(num_steps);
        Var sqrt_dt = sqrt(dt);
        Var sum_payoff = S * 0.0;
        
        for (int path = 0; path < current_batch_size; path++) {
            Var S_current = S;
            Var path_sum = S;
            
            // simulate a path
            for (int step = 0; step < num_steps; step++) {
                double Z = norm(gen);
                
                S_current = S_current * exp((r_var - 0.5*sig*sig)*dt + sig*sqrt_dt*Z);
                path_sum = path_sum + S_current;
            }
            
            // payoff for this path
            Var avg_price = path_sum / static_cast<double>(num_steps + 1);
            Var payoff = max(avg_price - K_var, 0.0);
            sum_payoff = sum_payoff + payoff;
        }
        
        Var batch_price = exp(-1 * r_var * T_var) * sum_payoff / static_cast<double>(current_batch_size);
        
        tape->backward(batch_price.index);
        
        // accumulate gradients
        double weight = static_cast<double>(current_batch_size);
        sum_price += batch_price.val * weight;
        sum_delta += tape->grad(S.index) * weight;
        sum_vega += tape->grad(sig.index) * weight;
        sum_rho += tape->grad(r_var.index) * weight;
        sum_theta += tape->grad(T_var.index) * weight;
        sum_strike_sens += tape->grad(K_var.index) * weight;

        // tape freed automatically
    }

    // Average across all paths
    double total_paths = static_cast<double>(num_paths);
    return {
        sum_price / total_paths,
        sum_delta / total_paths,
        sum_vega / total_paths,
        sum_rho / total_paths,
        -sum_theta / total_paths,  // Flip sign for theta
        sum_strike_sens / total_paths
    };
}

// Finite Difference Delta (for comparison only)
double asianCallDelta_FD(double S0, double sigma, double K, 
                         double T_expiry, double r, 
                         int num_paths, int num_steps, 
                         double h = 0.01, unsigned seed = 42) {
    
    double price_up = asianCallMC<double>(S0 + h, sigma, K, T_expiry, r, 
                                          num_paths, num_steps, seed);
    
    double price_down = asianCallMC<double>(S0 - h, sigma, K, T_expiry, r, 
                                            num_paths, num_steps, seed);
    
    return (price_up - price_down) / (2.0 * h);
}

// Generate correlated normal samples
template<typename T>
std::vector<T> generateCorrelatedNormals(
    const std::vector<std::vector<double>>& chol,
    std::mt19937& gen) {
    
    std::normal_distribution<double> norm(0.0, 1.0);
    int n = chol.size();
    
    // independent normals
    std::vector<double> Z_indep(n);
    for (int i = 0; i < n; i++) {
        Z_indep[i] = norm(gen);
    }
    
    // correlate them
    std::vector<T> Z_corr(n);
    for (int i = 0; i < n; i++) {
        T sum = T(0.0);
        for (int j = 0; j <= i; j++) {
            sum = sum + chol[i][j] * Z_indep[j];
        }
        Z_corr[i] = sum;
    }
    
    return Z_corr;
}

// multi-asset basket call option
template<typename T>
T basketCallMC(const std::vector<T>& S0,           // Initial prices
                const std::vector<T>& sigma,        // Volatilities
                const std::vector<double>& weights, // Basket weights
                double K,                           // Strike
                double T_expiry,                    // Time to expiry
                T r,                                // Risk-free rate
                const std::vector<std::vector<double>>& corr, // Correlation matrix
                int num_paths,
                unsigned seed = 42) {
    
    int n_assets = S0.size();
    std::mt19937 gen(seed);
    
    // precompute Cholesky
    auto chol = choleskyDecomposition(corr);
    
    T sum_payoff = S0[0] * 0.0;
    
    for (int path = 0; path < num_paths; path++) {
        // generate correlated randoms
        std::normal_distribution<double> norm(0.0, 1.0);
        std::vector<double> Z_indep(n_assets);
        for (int i = 0; i < n_assets; i++) {
            Z_indep[i] = norm(gen);
        }
        
        // get correlated normals
        std::vector<double> Z(n_assets, 0.0);
        for (int i = 0; i < n_assets; i++) {
            for (int j = 0; j <= i; j++) {
                Z[i] += chol[i][j] * Z_indep[j];
            }
        }
        
        // compute terminal asset prices
        T basket_value = S0[0] * 0.0;
        
        for (int i = 0; i < n_assets; i++) {
            // GBM terminal value for asset i
            T S_T = S0[i] * exp((r - 0.5*sigma[i]*sigma[i])*T_expiry + sigma[i]*sqrt(T_expiry)*Z[i]);
            
            // add to weighted basket
            basket_value = basket_value + weights[i] * S_T;
        }
        
        // basket call payoff
        T payoff = max(basket_value - K, 0.0);
        sum_payoff = sum_payoff + payoff;
    }
    
    // discount and average
    T price = exp(-r * T_expiry) * sum_payoff / static_cast<double>(num_paths);
    return price;
}

#endif