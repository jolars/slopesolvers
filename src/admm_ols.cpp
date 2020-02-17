#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

//' OLS with ADMM
//'
//' @param x predictors
//' @param y response in {-1, 1}
//' @param lambda regularization sequence
//' @param max_passes maximum number of passes
//' @param opt optimal value
//' @param opt_tol relative suboptimality tolerance
//' @export
// [[Rcpp::export]]
List admm_ols(arma::mat x,
              arma::vec y,
              arma::vec lambda,
              arma::uword max_passes,
              const double opt,
              const double opt_tol = 1e-4)
{
  uword p = x.n_cols;
  uword n = x.n_rows;

  Slope prox(p);

  std::vector<double> time;
  std::vector<double> loss;

  wall_clock timer;
  timer.tic();

  vec beta(p, fill::zeros);
  vec beta_hat = beta;

	double alpha = 1.5;

	vec xTy = x.t() * y;
	double rho = lambda.max();
	mat L, U;

	if (n >= p) {
	  lu(L, U, x.t() * x + rho*speye(p, p));
	} else {
	  lu(L, U, speye(n, n) + (1/rho)*(x * x.t()));
	}

	vec z(p, fill::zeros);
	vec u(p, fill::zeros);
	vec z_old(z);
	vec q;

	// ADMM loop
	uword passes = 0;

	while (passes < max_passes) {
	  ++passes;

    double f = 0.5*pow(norm(y - x*z), 2);
    double slope_norm = dot(sort(abs(beta), "descend"), lambda);

    loss.emplace_back(f + slope_norm);
    time.emplace_back(timer.toc());

    if (std::abs((f + slope_norm - opt)/opt) <= opt_tol)
      break;

	  q = xTy + rho*(z - u);

	  if (n >= p) {
	    beta = solve(trimatu(U), solve(trimatl(L), q));
	  } else {
	    beta = q/rho - (x.t() * solve(trimatu(U), solve(trimatl(L), x*q)))/(rho*rho);
	  }

	  z_old = z;
	  beta_hat = alpha*beta + (1 - alpha)*z_old;

	  z = prox(beta_hat + u, lambda/rho);

	  u += (beta_hat - z);

		checkUserInterrupt();
	}

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(z)
  );
}


