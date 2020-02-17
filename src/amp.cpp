#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

template <typename T>
uword n_unique_abs(const T& a)
{
  return unique(abs(a)).eval().n_elem;
}

double F(double tau,
         const vec& alpha,
         const vec& prior,
         uword iteration,
         double sigma,
         double delta)
{
  double result = 0;

  uword p = prior.n_elem;

  Slope prox(p);
  vec rn(p);

  for (uword i = 0; i < iteration; ++i) {
    for (uword j = 0; j < p; ++j) {
      rn(j) = Rf_rnorm(0, 1);
    }

    result += mean(square(prox(prior + tau*rn, tau*alpha) - prior))/iteration;
  }

  return pow(sigma, 2) + result/delta;
}

double alpha_to_tau(const vec& alpha_seq,
                    const vec& prior,
                    double sigma,
                    uword max_iter,
                    double delta)
{
  uword p = prior.n_elem;

  double tau = sqrt(sigma*sigma + mean(square(prior))/delta);
  vec record_tau(max_iter);

  for (uword t = 0; t < max_iter; ++t) {
    tau = sqrt(F(tau, alpha_seq, prior, 100, sigma, delta));
    record_tau(t) = tau;
  }

  return mean(record_tau);
}

vec alpha_to_lambda(const vec& alpha_seq,
                    const uword max_iter,
                    const vec& prior,
                    const uword second_iter,
                    const double sigma,
                    const double delta)
{
  uword p = prior.n_elem;

  double tau = alpha_to_tau(alpha_seq, prior, sigma, max_iter, delta);

  double E = 0;
  vec prox_solution(p);
  vec rn(p);

  Slope prox(p);

  for (uword i = 0; i < second_iter; ++i) {
    for (uword j = 0; j < p; ++j) {
      rn(j) = Rf_rnorm(0, 1);
    }

    prox_solution = prox(prior + tau*rn, alpha_seq*tau);
    E += n_unique_abs(prox_solution)/(p*delta*second_iter);
  }

  return (1 - E)*alpha_seq*tau;
}

vec lambda_to_alpha(vec lambda_seq,
                    const double tol,
                    const vec& prior,
                    const double sigma,
                    const double delta,
                    const uword max_iter,
                    const uword max_iter2)
{
  const uword p = prior.n_elem;

  lambda_seq = sort(lambda_seq, "descend");

  // standardize lambda sequence
  vec l = lambda_seq/lambda_seq(0);

  vec alpha1 = l*0.5;
  vec alpha2 = l*2.0;

  vec lambda1 = alpha_to_lambda(alpha1, max_iter, prior, sigma, max_iter2, delta);
  vec lambda2 = alpha_to_lambda(alpha2, max_iter, prior, sigma, max_iter2, delta);

  while ((lambda1(0) < lambda_seq(0)) * (lambda2(0) > lambda_seq(0)) == 0) {
    if (lambda1(0) < lambda_seq(0)) {
      alpha1 *= 2;
      alpha2 *= 2;
    } else {
      alpha1 *= 0.5;
      alpha2 *= 0.5;
    }
    lambda1 = alpha_to_lambda(alpha1, max_iter, prior, sigma, max_iter2, delta);
    lambda2 = alpha_to_lambda(alpha2, max_iter, prior, sigma, max_iter2, delta);
  }

  // bisection
  vec middle_alpha_seq(p);
  vec middle_lambda(p);

  while ((alpha2(0) - alpha1(0)) > tol) {
    middle_alpha_seq = (alpha1 + alpha2)*0.5;
    middle_lambda = alpha_to_lambda(middle_alpha_seq, max_iter, prior, sigma, max_iter2, delta);

    if (middle_lambda(0) > lambda_seq(0)) {
      alpha2 = middle_alpha_seq;
    } else if (middle_lambda(0) < lambda_seq(0)) {
      alpha1 = middle_alpha_seq;
    } else {
      break;
    }
  }

  return middle_alpha_seq;
}


// [[Rcpp::export]]
List amp(arma::mat x,
         arma::vec y,
         arma::vec lambda,
         arma::uword max_passes,
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
  vec gradient(p, fill::zeros);

  vec z = y;

  vec residual(n);
  vec lin_pred(n);

  double delta = static_cast<double>(n)/static_cast<double>(p);
  double sigma = 0;
  double eps = 0.1;
  double tau = std::sqrt(sigma*sigma + 4*eps/delta);

  // vec alpha(p, fill::zeros);
  // alpha.head(p/2).ones();

  vec prior(p, fill::zeros);

  // for (uword j = 0; j < p; ++j) {
  //   prior(j) = Rf_rnorm(0, 1)*Rf_rbinom(1, eps);
  // }

  double tol = 1e-5;

  vec alpha = lambda_to_alpha(lambda, tol, prior, sigma, delta, 100, 100);

  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    beta = prox(x.t() * z + beta, alpha*tau);
    z = y - x * beta + z*n_unique_abs(beta)/n;
    tau = std::sqrt(F(tau, alpha, prior, 1, sigma, delta));

    double norm_old = norm(x*beta - y);
    double f = 0.5*pow(norm_old, 2);
    double slope_norm = dot(sort(abs(beta), "descend"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f + slope_norm - opt)/opt) <= 1e-6)
      break;

    checkUserInterrupt();
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(beta)
  );
}


