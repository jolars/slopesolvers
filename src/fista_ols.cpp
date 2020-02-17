#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

//' OLS with FISTA
//'
//' @param x predictors
//' @param y response in
//' @param lambda regularization sequence
//' @param max_passes maximum number of passes
//' @param opt optimal value
//' @param opt_tol relative suboptimality tolerance
//' @export
// [[Rcpp::export]]
List fista_ols(arma::mat x,
               arma::vec y,
               arma::vec lambda,
               arma::uword max_passes,
               const double opt,
               const double opt_tol = 1e-4)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  std::vector<double> time;
  std::vector<double> loss;

  Slope prox(p);

  wall_clock timer;
  timer.tic();

  double learning_rate = 1;
  double eta = 0.5;

  vec beta(p, fill::zeros);
  vec beta_tilde(p);
  vec beta_old(beta);
  vec gradient(p, fill::zeros);

  vec lin_pred(n);

  double t = 0;
  double t_old = 0;

  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    lin_pred = x * beta;
    gradient = x.t() * (lin_pred - y);

    double f_old = 0.5*pow(norm(y - lin_pred), 2);
    double slope_norm = dot(sort(abs(beta), "descend"), lambda);

    loss.emplace_back(f_old + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f_old + slope_norm - opt)/opt) <= opt_tol)
      break;

    while (true) {

      beta_tilde = prox(beta - learning_rate*gradient, lambda*learning_rate);
      vec d = beta_tilde - beta;

      double f = 0.5*pow(norm(y - x*beta_tilde), 2);

      double q = f_old
        + dot(d, gradient)
        + (1.0/(2*learning_rate))*accu(square(d));

      if (q >= f*(1 - 1e-12))
        break;

      learning_rate *= eta;

      checkUserInterrupt();
    }

    t = 0.5*(1 + std::sqrt(1 + 4*t_old*t_old));

    beta = beta_tilde + (t_old - 1)/t * (beta_tilde - beta_old);

    beta_old = beta;
    t_old = t;

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(beta)
  );
}


