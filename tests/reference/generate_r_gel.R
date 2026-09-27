# Run from the repository root. Requires CRAN gmm and jsonlite.
set.seed(492)
y <- rnorm(300,1.3,.8)
write.csv(data.frame(y=y),'tests/data/r_gel_input.csv',row.names=FALSE)
g <- function(theta,x) cbind(x-theta[1], (x-theta[1])^2-theta[2], (x-theta[1])^3)
refs <- list()
for (type in c('EL','ET','CUE')) {
 m <- gmm::gel(g,y,c(mean(y),var(y)),type=type)
 refs[[type]] <- list(coef=unname(coef(m)), covariance=unname(vcov(m)))
}
refs$provenance <- list(R=R.version.string,gmm=as.character(packageVersion('gmm')),command='Rscript tests/reference/generate_r_gel.R',seed=492)
jsonlite::write_json(refs,'tests/data/r_gel_reference.json',digits=16,pretty=TRUE,auto_unbox=TRUE)
