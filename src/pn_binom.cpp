#include <RcppArmadillo.h>
#include "prox.h"

using namespace Rcpp;
using namespace arma;

double objective(const vec& b, const vec& Ax)
{
  return accu(log(1.0 + exp(-b % Ax)));
}

inline vec gradient(const mat& A,
                    const vec& b,
                    const vec& Ax)
{
  return A.t() * (-b / (1.0 + exp(b % Ax)));
}

inline mat hessian(const mat& A,
                   const vec& b,
                   const vec& Ax)
{
  vec sig = 1/(1 + exp(-b % Ax));
  return A.t() * diagmat(b % sig % (1 - sig) % b) * A;
}


//solve subproblem using admm
vec solve_subproblem(const vec& x,
                     const vec& gradient_x,
                     const mat& H,
                     const vec& lambda)
{
  const uword p = lambda.n_elem;

  Slope prox(p);

  vec z(p + 1, fill::zeros);
  vec z0(z);
  vec z1(z);
  double f = 0;
  double f0 = 0;
  double h = 0;
  double g = 0;
  double g0 = 1;
  double learning_rate = 1;
  vec gradient_z(z);

  double eta = 0.5;
  double t = 0;
  double t0 = 0;

  uword max_passes = 500;
  uword passes = 0;

  while (passes < max_passes) {
    ++passes;

    vec zmx = z - x;
    gradient_z = gradient_x + H*zmx;

    g = dot(gradient_x, zmx) + 0.5*as_scalar(zmx.t()*H*zmx);
    h = dot(sort(abs(z.tail(p)), "descend"), lambda);
    f = g + h;

    if (std::abs((f - f0)/f) < 1e-5)
      break;

    f0 = f;
    g0 = g;

    while (true) {

      z1 = z - learning_rate*gradient_z;
      z1.tail(p) = prox(z1.tail(p), lambda*learning_rate);

      vec d = z1 - z;
      vec z1mx = z1 - x;

      g = dot(gradient_x, z1mx) + 0.5*as_scalar(z1mx.t()*H*z1mx);
      double q = g0 + dot(d, gradient_z) + (1.0/(2*learning_rate))*accu(square(d));

      if (q >= g*(1 - 1e-12))
        break;

      learning_rate *= eta;

      checkUserInterrupt();
    }

    t = 0.5*(1.0 + std::sqrt(1.0 + 4.0*t0*t0));

    z = z1 + (t0 - 1.0)/t * (z1 - z0);
    z0 = z;
    t0 = t;

    checkUserInterrupt();
  }

  return z;
}

//' Logistic regression with proximal newton
//'
//' @param x predictors
//' @param y response
//' @param lambda regularization sequence
//' @param max_passes maximum number of passes
//' @param opt optimal value
//' @param opt_tol relative suboptimality tolerance
//' @export
// [[Rcpp::export]]
List pn_binom(arma::mat A,
              arma::vec b,
              arma::vec lambda,
              arma::uword max_passes,
              const double opt,
              const double opt_tol = 1e-4)
{
  uword p = A.n_cols;
  uword n = A.n_rows;

  A.insert_cols(0, ones(n));

  Slope prox(p);

  std::vector<double> time;
  std::vector<double> loss;

  wall_clock timer;
  timer.tic();

  vec x(p + 1, fill::zeros);
  vec x_new(x);
  vec x_old(x);
  vec dx(x);
  vec y(x);

  vec lin_pred(n, fill::zeros);

  double df = 0;

  // newton parameters
  double alpha = 0.001;
  double beta = 0.5;
  double xtol = 1e-9;

  lin_pred = A*x;

  double g_x = objective(b, lin_pred);
  vec grad_g_x = gradient(A, b, lin_pred);
  vec grad_g_old(grad_g_x);
  mat hess_g_x = hessian(A, b, lin_pred);
  double h_x = dot(sort(abs(x.tail(p)), "descend"), lambda);
  double f_x = g_x + h_x;

  double f_y, g_y, h_y = 0;
  vec grad_g_y(p+1, fill::zeros);
  mat hess_g_y(p+1, p+1, fill::zeros);

  uword passes = 0;

  while (passes < max_passes) {
    loss.emplace_back(f_x);
    time.emplace_back(timer.toc());

    if (std::abs((f_x - opt)/opt) <= opt_tol)
      break;

    vec v = solve_subproblem(x, grad_g_x, hess_g_x, lambda) - x;

    // linesearch
    double t = 1;
    double grad_g_x_v = dot(grad_g_x, v);

    while (true) {

      y = x + t*v;

      lin_pred = A*y;
      grad_g_y = gradient(A, b, lin_pred);
      hess_g_y = hessian(A, b, lin_pred);

      g_y = objective(b, lin_pred);
      h_y = dot(sort(abs(y.tail(p)), "descend"), lambda);
      f_y = g_y + h_y;

      if (f_y > (f_x + alpha*t*grad_g_x_v + alpha*(h_y - h_x)))
        t *= beta;
      else
        break;

      // if (f_y < (f_x + alpha*t*desc))
      //   break;
      // else if (t <= xtol)
      //   break;
      //
      // if (!is_finite(f_y) || std::abs(f_y - f_x - t*grad_g_x_d) <= 1e-9) {
      //   t *= beta;
      // } else {
      //   double t_interp = -(grad_g_x_d*t*t) / (2*(f_y - f_x - t*grad_g_x_d));
      //   if (t_interp >= 0.01 && t_interp <= 0.99*t)
      //     t = t_interp;
      //   else
      //     t *= beta;
      // }
      checkUserInterrupt();
    }

    std::swap(f_x, f_y);
    std::swap(x, y);
    std::swap(grad_g_x, grad_g_y);
    std::swap(hess_g_x, hess_g_y);

    ++passes;
  }

  return List::create(
    Named("loss") = wrap(loss),
    Named("time") = wrap(time),
    Named("beta") = wrap(x)
  );
}


