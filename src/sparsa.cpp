#include <RcppArmadillo.h>
#include "prox.h"
#include "project_to_OWL_ball.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List sparsa(arma::mat x,
            arma::vec y,
            arma::vec lambda,
            arma::uword max_passes,
            double epsilon,
            double opt)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  std::vector<double> time;
  std::vector<double> loss;

  Slope prox(p);

  wall_clock timer;
  timer.tic();

  vec beta(p, fill::zeros);
  vec beta_old(beta);
  vec beta_new(beta);
  vec gradient(p, fill::zeros);

  vec lin_pred(n);

  double eta = 2;

  double alpha = 1;
  double alpha_min = 1e-30;
  double alpha_max = 1/alpha_min;

  beta_old = beta;

  gradient = x.t() * (x*beta_old - y);
  beta = prox(beta_old - gradient/alpha, lambda/alpha);

  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    alpha = pow(norm(x*(beta - beta_old)), 2)/pow(norm(beta - beta_old), 2);
    alpha = std::max(alpha_min, std::min(alpha, alpha_max));

    double norm_old = norm(x*beta - y);
    double f = 0.5*pow(norm_old, 2);
    double slope_norm = dot(sort(abs(beta), "descend"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f + slope_norm - opt)/opt) <= 1e-6)
      break;

    gradient = x.t() * (x*beta - y);

    while (true) {
      checkUserInterrupt();

      // vec z = beta - gradient/alpha;
      // std::vector<double> z_in = conv_to<std::vector<double>>::from(z);
      // std::vector<double> lambda_stdvec = conv_to<std::vector<double>>::from(lambda);
      // std::vector<double> z_out(p);
      //
      // evaluateProx(z_in, lambda_stdvec, epsilon, z_out, false);
      // beta_new = conv_to<vec>::from(z_out);

      beta_new = prox(beta - gradient/alpha, lambda/alpha);

      if (norm(x*beta_new - y) <= norm_old)
        break;

      alpha *= eta;
    }

    beta_old = beta;
    beta = beta_new;

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(beta)
  );
}


