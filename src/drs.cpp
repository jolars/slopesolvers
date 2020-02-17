#include <RcppArmadillo.h>
#include "prox.h"
#include "project_to_OWL_ball.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List drs(arma::mat x,
         arma::vec y,
         arma::vec lambda,
         arma::uword max_passes,
         double epsilon,
         const double opt)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  std::vector<double> time;
  std::vector<double> loss;

  Slope prox(p);

  wall_clock timer;
  timer.tic();

  vec beta(p, fill::zeros);

  double gamma = 1;
  double alpha = 1;

  uword passes = 0;

  mat IpXtX = speye(p, p) + gamma*x.t()*x;
  mat Xty = x.t() * y;

  std::vector<double> lambda_stdvec = conv_to<std::vector<double>>::from(lambda);

  while (passes < max_passes) {
    ++passes;

    double f = 0.5*pow(norm(y - x*beta), 2);
    double slope_norm = dot(sort(abs(beta), "descend"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f + slope_norm - opt)/opt) <= 1e-6)
      break;

    vec beta_g = solve(IpXtX, beta + gamma*Xty);
    vec beta_t = 2*beta_g - beta;

    std::vector<double> z_in = conv_to<std::vector<double>>::from(beta_t);
    std::vector<double> z_out(p);

    evaluateProx(z_in, lambda_stdvec, epsilon, z_out, false);

    vec beta_f = conv_to<vec>::from(z_out);

    beta += alpha*(beta_f - beta_g);

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(beta)
  );
}


