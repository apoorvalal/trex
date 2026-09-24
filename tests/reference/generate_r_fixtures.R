# Rebuild from the repository root: Rscript tests/reference/generate_r_fixtures.R
# Independent oracles: base stats::lm/glm, sandwich::vcovHC, fixest::feols.
stopifnot(requireNamespace('jsonlite'), requireNamespace('sandwich'), requireNamespace('fixest'))
set.seed(7342)
n <- 240
D <- data.frame(x1=rnorm(n), x2=rnorm(n), unit=rep(1:24,each=10), period=rep(1:10,24))
D$w <- exp(.7*D$x1)
D$y <- 1.2 + .8*D$x1 - .4*D$x2 + sin(D$unit) + .15*D$period + rnorm(n)*(.4+abs(D$x1))
D$binary <- rbinom(n,1,plogis(.2+.6*D$x1-.35*D$x2))
D$count <- rpois(n,exp(.2+.3*D$x1-.2*D$x2))
write.csv(D, 'tests/data/r_regression_input.csv',row.names=FALSE)
pack <- function(m) list(coef=unname(coef(m)), covariance=unname(vcov(m)), HC0=unname(sandwich::vcovHC(m,type='HC0')), HC1=unname(sandwich::vcovHC(m,type='HC1')))
refs <- list(ols=pack(lm(y~x1+x2,D)), no_intercept=pack(lm(y~x1+x2-1,D)), wls=pack(lm(y~x1+x2,D,weights=w)), fe=pack(lm(y~x1+x2+factor(unit)+factor(period),D,weights=w)), logistic=pack(glm(binary~x1+x2,D,family=binomial())), poisson=pack(glm(count~x1+x2,D,family=poisson())))
fx <- fixest::feols(y~x1+x2|unit+period,D,weights=~w,fixef.tol=1e-10)
ssc <- if ('K.adj' %in% names(formals(fixest::ssc))) fixest::ssc(K.adj=TRUE,K.fixef='full',K.exact=TRUE) else fixest::ssc(adj=TRUE,fixef.K='full',fixef.force_exact=TRUE)
refs$fixest <- list(coef=unname(coef(fx)), HC1=unname(vcov(fx,vcov='hetero',ssc=ssc)))
refs$provenance <- list(R=R.version.string, sandwich=as.character(packageVersion('sandwich')), fixest=as.character(packageVersion('fixest')), command='Rscript tests/reference/generate_r_fixtures.R', seed=7342)
jsonlite::write_json(refs,'tests/data/r_regression_reference.json',digits=16,pretty=TRUE,auto_unbox=TRUE)
