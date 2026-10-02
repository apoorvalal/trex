# Paper source

Moreira, Marcelo J.; Newey, Whitney K.; Sharifvaghefi, Mahrad (2026).
*Continuously Updating GMM in Linear IV Models: A Polynomial Approach*.
[arXiv:2609.36445v1](https://arxiv.org/abs/2609.36445v1).

The pinned TeX archive was downloaded and read on 2026-09-30. `manifest.json`
records its SHA-256 and complete file list. Re-fetch with
`python benchmarks/cue_polynomial/source/fetch.py`. Local copies are retained
under ignored `download/`; the paper's copyrighted source is not redistributed
in this PR. That archive contains the main TeX, bibliography and two figures,
**not the online supplement or the authors' MATLAB simulation code**.

Implementation map (line numbers refer to `CU_GMM_v9g_arvix.tex`):

- 292–439, Section 2: projected IV moments, fixed joint reduced-form covariance,
  positive-definiteness assumption, common endpoint and just-identified case.
- 608–692, Section 3.1: determinant/adjugate polynomial ratio, degree ≤2k.
- 694–753, Section 3.2: stationary numerator p'q−pq', degree ≤4k−2;
  compare every real stationary point with infinity; constant criterion case.
- 755–828, Section 3.3: companion eigenvalues and polynomial interpolation.
- 1500–1568: n=800, beta=5, rho=.6, w=sqrt(.5), powers/Bernoulli-interaction
  instruments and intercept demeaning in the Monte Carlo design.

This is an independent, smaller implementation experiment, not a reproduction
of the paper's 10,000-replication tables. Chebyshev interpolation, covariance
preconditioning, reciprocal charts and float64 validation thresholds are our
implementation choices, not a claim about the missing supplement's exact code.
The paper's multiple-endogenous-variable elimination algorithm and its
rank-changing pseudoinverse extension are not implemented.
