#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List cd(arma::mat x,
        arma::vec y,
        arma::vec lambda,
        arma::uword max_passes)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  std::vector<double> time;
  std::vector<double> loss;

  Slope prox(p);

  wall_clock timer;
  timer.tic();

  vec beta(p, fill::zeros);
  vec gradient(p, fill::zeros);

  vec lin_pred(n);
  vec residual(n);
  vec partial_residual(n);

  uword passes = 0;

  vec xTx = sum(square(x)).t();

  while (passes < max_passes) {
    ++passes;

    lin_pred = x*beta;
    residual = y - lin_pred;

    double f = 0.5*pow(norm(residual), 2);
    double slope_norm = dot(sort(abs(beta), "descending"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    for (uword j = 0; j < p; ++j) {
      residual += x.col(j)*beta(j);
      beta(j) = dot(residual, x.col(j));

      vec beta_new = prox(beta/xTx, lambda/xTx);
      beta(j) = beta_new(j);

      residual -= x.col(j)*beta(j);
    }

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(beta)
  );
}


