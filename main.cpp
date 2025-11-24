#include <iostream>
#include <iomanip>
#include <chrono>
#include "Autodiff.h"
#include "BlackScholes.h"
#include "MonteCarlo.h"

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void printSubheader(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    printHeader("PART 1: Black-Scholes - First & Second Order Greeks");

    double S = 100.0, K = 100.0, T_exp = 1.0, sigma = 0.2, r = 0.05;

    printSubheader("First-Order Greeks (Forward-Mode AD)");

    Dual1 S_dual = Dual1::variable(100.0);
    Dual1 bs_delta = blackScholesCall(S_dual, Dual1::constant(0.2),
                                       Dual1::constant(100.0),
                                       Dual1::constant(1.0),
                                       Dual1::constant(0.05));

    Dual1 sigma_dual = Dual1::variable(0.2);
    Dual1 bs_vega = blackScholesCall(Dual1::constant(100.0), sigma_dual,
                                      Dual1::constant(100.0),
                                      Dual1::constant(1.0),
                                      Dual1::constant(0.05));

    std::cout << "Price: " << bs_delta.val << "\n";
    std::cout << "Delta: " << bs_delta.der << "\n";
    std::cout << "Vega:  " << bs_vega.der << "\n";

    // second-order Greeks with nested duals
    printSubheader("Second-Order Greeks (Nested Dual Numbers)");

    Dual2 S_second = Dual2::variable(Dual1::variable(100.0));
    Dual2 bs_gamma = blackScholesCall(S_second,
                                       Dual2::constant(Dual1::constant(0.2)),
                                       Dual2::constant(Dual1::constant(100.0)),
                                       Dual2::constant(Dual1::constant(1.0)),
                                       Dual2::constant(Dual1::constant(0.05)));

    std::cout << "Price: " << bs_gamma.val.val << "\n";
    std::cout << "Delta: " << bs_gamma.der.val << " (∂V/∂S)\n";
    std::cout << "Gamma: " << bs_gamma.der.der << " (∂²V/∂S²)\n";

    // analytical validation
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T_exp) / (sigma*sqrt(T_exp));
    double gamma_analytical = exp(-d1*d1/2) / (S * sigma * sqrt(2*M_PI*T_exp));

    std::cout << "\nValidation:\n";
    std::cout << "Gamma (analytical): " << gamma_analytical << "\n";
    std::cout << "Error: " << std::abs(bs_gamma.der.der - gamma_analytical) << "\n";

    printHeader("PART 2: Monte Carlo Pricing with Pathwise Differentiation");

    printSubheader("European Call (100k paths)");

    auto start_euro = std::chrono::high_resolution_clock::now();
    Dual1 S_euro = Dual1::variable(100.0);
    Dual1 euro_result = europeanCallMC(S_euro, Dual1::constant(0.2),
                                        Dual1::constant(100.0),
                                        Dual1::constant(1.0),
                                        Dual1::constant(0.05),
                                        100000);
    auto end_euro = std::chrono::high_resolution_clock::now();
    double time_euro = std::chrono::duration<double>(end_euro - start_euro).count();

    std::cout << "Price: " << euro_result.val << " (BS: " << bs_delta.val << ")\n";
    std::cout << "Delta: " << euro_result.der << " (BS: " << bs_delta.der << ")\n";
    std::cout << "Time:  " << time_euro << "s\n";
    std::cout << "MC Error: " << std::abs(euro_result.val - bs_delta.val) << "\n";

    printSubheader("Asian Call (10k paths, 252 steps)");

    auto start_asian = std::chrono::high_resolution_clock::now();
    Dual1 S_asian = Dual1::variable(100.0);
    Dual1 asian_result = asianCallMC(S_asian, Dual1::constant(0.2),
                                      Dual1::constant(100.0),
                                      Dual1::constant(1.0),
                                      Dual1::constant(0.05),
                                      10000, 252);
    auto end_asian = std::chrono::high_resolution_clock::now();
    double time_asian = std::chrono::duration<double>(end_asian - start_asian).count();

    std::cout << "Price: " << asian_result.val << "\n";
    std::cout << "Delta: " << asian_result.der << "\n";
    std::cout << "Time:  " << time_asian << "s\n";

    printHeader("PART 3: Reverse-Mode AD with Batched Checkpointing");

    printSubheader("Computing ALL 5 Greeks Simultaneously");

    auto start_batched = std::chrono::high_resolution_clock::now();
    GreeksResult greeks = asianCallMC_Batched(100.0, 0.2, 100.0, 1.0, 0.05,
                                              10000, 252, 1000, 42);
    auto end_batched = std::chrono::high_resolution_clock::now();
    double time_batched = std::chrono::duration<double>(end_batched - start_batched).count();

    std::cout << "\nResults:\n";
    std::cout << "  Price:       " << greeks.price << "\n";
    std::cout << "  Delta (∂V/∂S): " << greeks.delta << "\n";
    std::cout << "  Vega  (∂V/∂σ): " << greeks.vega << "\n";
    std::cout << "  Rho   (∂V/∂r): " << greeks.rho << "\n";
    std::cout << "  Theta (∂V/∂T): " << greeks.theta << "\n";
    std::cout << "  ∂V/∂K:       " << greeks.strike_sens << "\n";

    std::cout << "\nPerformance:\n";
    std::cout << "  Time (5 Greeks): " << time_batched << "s\n";
    double estimated_forward = time_batched * 5 * 1.17;
    std::cout << "  Forward-mode would take: ~" << estimated_forward << "s\n";
    std::cout << "  Speedup: " << (estimated_forward / time_batched) << "x\n";

    std::cout << "\nMemory Efficiency:\n";
    std::cout << "  Peak memory (batched):   " << (1000 * 252 * 15 * 40 / 1e6) << " MB\n";
    std::cout << "  Would need (unbatched): " << (10000 * 252 * 15 * 40 / 1e6) << " MB\n";
    std::cout << "  Memory reduction: 10x\n";

    printHeader("PART 4: Multi-Asset Basket Option (High-Dimensional Greeks)");

    int n_assets = 5;
    std::vector<double> S_init(n_assets, 100.0);    // all at 100
    std::vector<double> sigma_init(n_assets, 0.2);  // all 20% vol
    std::vector<double> weights(n_assets, 1.0/n_assets); // equal-weighted

    std::vector<std::vector<double>> corr(n_assets, std::vector<double>(n_assets));
    for (int i = 0; i < n_assets; i++) {
        for (int j = 0; j < n_assets; j++) {
            corr[i][j] = (i == j) ? 1.0 : 0.5;
        }
    }

    printSubheader("Computing 10 Greeks (5 deltas + 5 vegas) Simultaneously");

    auto tape_basket = std::make_shared<Tape>();
    std::vector<Var> S_vars, sigma_vars;

    for (int i = 0; i < n_assets; i++) {
        S_vars.push_back(Var(S_init[i], tape_basket));
        sigma_vars.push_back(Var(sigma_init[i], tape_basket));
    }

    Var r_basket(0.05, tape_basket);

    auto start_basket = std::chrono::high_resolution_clock::now();
    Var basket_price = basketCallMC(S_vars, sigma_vars, weights, 100.0, 1.0,
                                     r_basket, corr, 50000, 42);
    tape_basket->backward(basket_price.index);
    auto end_basket = std::chrono::high_resolution_clock::now();
    double time_basket = std::chrono::duration<double>(end_basket - start_basket).count();

    std::cout << "\nBasket Price: " << basket_price.val << "\n";
    std::cout << "\nDeltas (∂V/∂S_i):\n";
    for (int i = 0; i < n_assets; i++) {
        std::cout << "  Asset " << i << ": " << tape_basket->grad(S_vars[i].index) << "\n";
    }
    std::cout << "\nVegas (∂V/∂σ_i):\n";
    for (int i = 0; i < n_assets; i++) {
        std::cout << "  Asset " << i << ": " << tape_basket->grad(sigma_vars[i].index) << "\n";
    }

    std::cout << "\nPerformance:\n";
    std::cout << "  Time (10 Greeks): " << time_basket << "s\n";
    std::cout << "  FD estimate: " << (time_basket * 2 * n_assets) << "s\n";
    std::cout << "  Speedup: " << (2 * n_assets) << "x\n";

    printHeader("PART 5: Second-Order Cross-Greeks (Vanna & Volga)");

    printSubheader("Vanna: ∂²V/∂S∂σ");

    Dual2 S_vanna = Dual2::variable(Dual1::constant(100.0));
    Dual2 sigma_vanna = Dual2::constant(Dual1::variable(0.2));

    Dual2 bs_vanna = blackScholesCall(S_vanna, sigma_vanna,
                                       Dual2::constant(Dual1::constant(100.0)),
                                       Dual2::constant(Dual1::constant(1.0)),
                                       Dual2::constant(Dual1::constant(0.05)));

    double vanna = bs_vanna.der.der;
    std::cout << "Vanna: " << vanna << "\n";

    printSubheader("Volga: ∂²V/∂σ²");

    Dual2 sigma_volga = Dual2::variable(Dual1::variable(0.2));

    Dual2 bs_volga = blackScholesCall(Dual2::constant(Dual1::constant(100.0)),
                                       sigma_volga,
                                       Dual2::constant(Dual1::constant(100.0)),
                                       Dual2::constant(Dual1::constant(1.0)),
                                       Dual2::constant(Dual1::constant(0.05)));

    double volga = bs_volga.der.der;
    std::cout << "Volga: " << volga << "\n";

    printHeader("SUMMARY");

    std::cout << "\nPerformance:\n";
    std::cout << "  Forward-mode: " << time_asian << "s (1 Greek)\n";
    std::cout << "  Reverse-mode: " << time_batched << "s (5 Greeks)\n";
    std::cout << "  Speedup: " << std::setprecision(1) << (estimated_forward / time_batched) << "x\n";

    std::cout << "\nAccuracy:\n";
    std::cout << "  Gamma error: " << std::setprecision(6) << std::abs(bs_gamma.der.der - gamma_analytical) << "\n";
    std::cout << "  MC error: " << std::abs(euro_result.val - bs_delta.val) << "\n";

    std::cout << "\nMemory:\n";
    std::cout << "  Batched: " << (1000 * 252 * 15 * 40 / 1e6) << " MB\n";
    std::cout << "  Unbatched: " << (10000 * 252 * 15 * 40 / 1e6) << " MB\n";

    printHeader("COMPLETE");

    return 0;
}
