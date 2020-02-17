#pragma once

#include <RcppArmadillo.h>

using namespace arma;

struct Slope {
  const uword p;
  vec s;
  vec w;
  vec beta_sign;
  uvec beta_order;
  uvec idx_i;
  uvec idx_j;

  Slope(uword p)
    : p(p),
      s(p),
      w(p),
      beta_sign(p),
      beta_order(p),
      idx_i(p),
      idx_j(p)
    {}

  vec operator()(vec beta, vec lambda)
  {
    // collect sign of beta and work with sorted absolutes
    beta_sign = sign(beta);
    beta = abs(beta);
    beta_order = sort_index(beta, "descend");
    beta = beta(beta_order);

    uword k = 0;

    for (uword i = 0; i < p; i++) {
      idx_i(k) = i;
      idx_j(k) = i;
      s(k)     = beta(i) - lambda(i);
      w(k)     = s(k);

      while ((k > 0) && (w[k - 1] <= w(k))) {
        k--;
        idx_j(k)  = i;
        s(k)     += s(k + 1);
        w(k)      = s(k) / (i - idx_i(k) + 1.0);
      }
      k++;
    }

    for (uword j = 0; j < k; j++) {
      double d = std::max(w(j), 0.0);
      for (uword i = idx_i(j); i <= idx_j(j); i++) {
        beta(i) = d;
      }
    }

    // reset order
    beta(beta_order) = beta;

    // reset sign
    beta %= beta_sign;

    return beta;
  }
};

