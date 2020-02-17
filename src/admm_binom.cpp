#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

double objective(const vec& y,
                 const vec& lin_pred,
                 const vec& beta,
                 const vec& u,
                 const vec& z,
                 double rho)
{
  return accu(log(1.0 + exp(-y % lin_pred))) + 0.5*rho*std::pow(norm(beta - z + u), 2);
}

inline vec gradient(const mat& x,
                    const vec& y,
                    const vec& lin_pred)
{
  return x.t() * (-y / (1.0 + exp(y % lin_pred)));
}

inline mat hessian(const mat& x,
                   const vec& y,
                   const vec& lin_pred)
{
  vec sig = 1/(1 + exp(-y % lin_pred));
  return x.t() * diagmat(y % sig % (1 - sig) % y) * x;
}

void update_beta(vec& beta,
                 const mat& x,
                 const vec& y,
                 const vec& u,
                 const vec& z,
                 double rho)
{
  uword p = beta.n_elem;
  uword n = x.n_rows;

  vec beta_new(beta);
  vec dbeta(beta);

  vec lin_pred(n, fill::zeros);
  vec g(p, fill::zeros);
  mat H(p, p, fill::zeros);

  double df = 0;

  // newton parameters
  double alpha = 0.1;
  double gamma = 0.5;
  double tol = 1e-5;
  uword max_passes = 50;

  // ADMM loop
  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    lin_pred = x*beta;

    double f = objective(y, lin_pred, beta, u, z, rho);

    g = gradient(x, y, lin_pred) + rho*(beta - z + u);
    H = hessian(x, y, lin_pred);
    H.diag() += rho;

    dbeta = -solve(H, g);
    df = dot(g, dbeta);

    if (0.5*df*df < tol)
      break;

    double t = 1;

    while (true) {
      beta_new = beta + t*dbeta;
      lin_pred = x*beta_new;
      double f_new = objective(y, lin_pred, beta_new, u, z, rho);

      if (f_new <= (f + alpha*t*df))
        break;

      t *= gamma;

      checkUserInterrupt();
    }

    beta = beta_new;
  }
}

//' Logistic regression with ADMM
//'
//' @param x predictors
//' @param y response in {-1, 1}
//' @param lambda regularization sequence
//' @param max_passes maximum number of passes
//' @param opt optimal value
//' @param opt_tol relative suboptimality tolerance
//' @export
// [[Rcpp::export]]
List admm_binom(arma::mat x,
                arma::vec y,
                arma::vec lambda,
                arma::uword max_passes,
                const double opt,
                const double opt_tol = 1e-4)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  x.insert_cols(0, ones(n));

  Slope prox(p);

  std::vector<double> time;
  std::vector<double> loss;

  wall_clock timer;
  timer.tic();

  vec beta(p + 1, fill::zeros);
  vec beta_hat(beta);

  double alpha = 1;
  double rho = lambda.max();

  vec z(p + 1, fill::zeros);
  vec u(p + 1, fill::zeros);
  vec z_old(z);

  // ADMM loop
  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    double f = accu(log(1 + exp(-y % (x*beta))));
    double slope_norm = dot(sort(abs(z.tail(p)), "descend"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f + slope_norm - opt)/opt) <= opt_tol)
      break;

    update_beta(beta, x, y, u, z, rho);

    z_old = z;
    beta_hat = alpha*beta + (1 - alpha)*z_old;
    z = beta_hat + u;
    z.tail(p) = prox(z.tail(p), lambda/rho);

    u += (beta_hat - z);

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(z)
  );
}


