% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{fista_binom}
\alias{fista_binom}
\title{OLS with FISTA}
\usage{
fista_binom(x, y, lambda, max_passes, opt, opt_tol = 1e-04)
}
\arguments{
\item{x}{predictors}

\item{y}{response}

\item{lambda}{regularization sequence}

\item{max_passes}{maximum number of passes}

\item{opt}{optimal value}

\item{opt_tol}{relative suboptimality tolerance}
}
\description{
OLS with FISTA
}
