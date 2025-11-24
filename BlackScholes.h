#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

template <typename T>

T blackScholesCall(T S, T sigma, T K, T T_expiry, T r)
{
    const T sigma_root_t = sigma * sqrt(T_expiry);
    const double inv_sqrt_two = (1/sqrt(2));

    T d1 = (log(S/K) + (r + (sigma * sigma)/2.0)*T_expiry) / sigma_root_t;
    T Nd1 = 0.5 * (1 + erf(d1 * inv_sqrt_two));

    T d2 = d1 - sigma_root_t;
    T Nd2 = 0.5 * (1 + erf(d2 * inv_sqrt_two));

    T discount = K * exp(-1 * r * T_expiry) * Nd2;

    return S * Nd1 - discount;
}

#endif