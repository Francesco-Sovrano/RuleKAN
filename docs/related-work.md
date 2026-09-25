# Related work

RuleKAN combines ideas from KANs, separated representations, sparse pursuit and symbolic regression. The implementation is not a direct instance of any one of these methods: numerical support discovery, support-conditioned symbolic factorization, multiplicity/rank expansion and the powered outer grammar are separate components.

## Kolmogorov-Arnold Networks

KANs replace scalar edge weights with learnable univariate functions, commonly represented by splines [1]. RuleKAN uses the same basic premise for its numerical edge functions but changes the aggregation structure: each rule multiplies selected one-dimensional edge functions, and rules are summed.

KAN 2.0 introduced MultKAN, in which multiplication nodes extend the additive KAN architecture and are used for scientific and symbolic discovery [2]. RuleKAN is related through its explicit multiplicative structure, but its symbolic procedure separates three objects that are not assumed to coincide: the numerical support, the symbolic factor multiplicity within a support, and the additive symbolic rank assigned to that support.

The repository also provides Gaussian-RBF numerical edges. This is related to FastKAN, which replaces spline edge bases with Gaussian radial basis functions for faster KAN evaluation [15]. In RuleKAN, changing the numerical basis does not change the final symbolic vocabulary or the learned-support constraint. We refer to this Gaussian-RBF numerical-basis variant as **RuleKAN-RBF**; its internal benchmark identifier remains `rulekan_fast` for backward compatibility.

## Separated representations and tensor decompositions

A finite sum of products of univariate functions is a separated representation. Beylkin and Mohlenkamp developed separated representations for high-dimensional numerical analysis [3], and Beylkin, Garcke and Mohlenkamp applied sums of separable functions to multivariate regression [4]. RuleKAN belongs to this broad functional form, with sparse learned supports and data-adaptive one-dimensional numerical factors.

There is also a connection to canonical polyadic (CP/PARAFAC) tensor decomposition, which writes a tensor as a sum of rank-one outer products [16]. RuleKAN is not a tensor decomposition: samples need not lie on a tensor grid, supports can differ across rules, and each factor is a learned univariate function. The analogy is useful because both models separate additive rank from multiplicative factorization.

## Sparse pursuit and dictionary methods

Matching pursuit greedily builds a sparse approximation from a redundant dictionary [17]. Orthogonal Matching Pursuit (OMP) refits coefficients of previously selected atoms and has a large sparse-recovery literature [18]. RuleKAN's symbolic stage uses the same general sparse-pursuit pattern: propose an atom, evaluate it against the current residual/model, refit continuous coefficients, and decide whether to commit it.

The RuleKAN dictionary is structured rather than fixed. An atom is a product of symbolic factors with continuous affine parameters, and its admissible variable support is normally restricted by the numerical RuleKAN precursor. The OMP benchmark variants change the pursuit/refit policy while retaining this support-conditioned symbolic grammar.

Sparse Identification of Nonlinear Dynamics (SINDy) uses sparse regression over a chosen nonlinear function library to identify governing equations [19]. RuleKAN shares the idea that a restricted function library can make equation discovery identifiable, but differs in two ways: it learns variable-interaction supports before symbolic search, and its symbolic atoms contain continuously reparameterized symbolic factors rather than a fixed precomputed feature matrix.

## Symbolic regression

Classical symbolic regression searches expression structure directly. Schmidt and Lipson demonstrated recovery of free-form natural laws from experimental data [5]. Genetic-programming systems such as Operon evolve expression trees and optimize constants [10]. PySR/`SymbolicRegression.jl` combines evolutionary search, simplification and continuous constant optimization [9]. Benchmark surveys show that symbolic-regression performance depends strongly on operator vocabulary, constant fitting, noise, complexity control and compute budget [12,13].

RuleKAN instead amortizes part of the structural search through a numerical support detector. SISP removes that restriction and is the repository's direct structure-independent control: it uses the same sum-product symbolic language and pursuit machinery but supplies the complete variable-multiset grammar.

`RuleKAN-Comp` and `SISP-Comp` add one expression-tree-like recursion level: an outer unary symbolic operator can wrap a compact sum-product inner expression. This targets the compositional advantage visible in tree-based systems such as PySR without replacing RuleKAN's support-conditioned search with a fully unrestricted recursive grammar.

## Differentiable equation discovery and exact symbolic atoms

Equation Learner (EQL) networks train with a restricted set of analytic units and sparsity so that the learned network can be read as an equation [6]. Neural-symbolic distillation methods similarly use a numerical model to guide later symbolic recovery [7]. These approaches motivate the use of continuous optimization around a finite symbolic vocabulary.

Recent KAN-specific symbolic-regression methods are especially close comparators. Sovrano et al. [22] identify a limitation of the original KAN symbolic replacement procedure: candidate functions are normally fitted to one learned KAN edge at a time, even though replacement quality depends on the rest of the network. Their Greedy Symbolic Regression (GSR) evaluates edge replacements through end-to-end loss with brief fine-tuning, while Gated Matching Pursuit (GMP) trains differentiable gates over an operator library on each edge before discretization. The benchmark already includes these procedures as `gsr` and `gmp`. Here **edge-wise** therefore describes the unit being symbolized---a KAN edge---rather than a distinct KAN architecture.

SR-KAN [23] is closer to RuleKAN at the level of the overall symbolic-regression objective: both use KAN-style numerical models to guide recovery of a closed-form multivariate expression and both explicitly accommodate multiplicative structure. Their structural biases differ. SR-KAN follows a divide-and-conquer strategy based on single-layer sum, multiplication, and composite KANs, together with symmetry/separability tests, target transformations, brute-force search for low input dimension, and library-based symbolic extraction. RuleKAN instead first learns sparse product-rule supports and restricts the later symbolic search to those supports, while allowing symbolic multiplicity and additive rank to differ from the numerical precursor. Because SR-KAN is an executable end-to-end symbolic-regression method in the same tabular regression setting, the research benchmark includes the authors' implementation as `srkan`.

Symbolic-KAN [24] is related but tests a different design choice. It embeds analytic primitives, learned scalar projections, hierarchical gates, and symbolic regularization directly in the trainable network; the gates are progressively sharpened so that active units select individual primitives and yield a closed-form network without a separate post-hoc symbolic fitting stage. RuleKAN deliberately keeps numerical support discovery and symbolic factorization as separate stages. The `research_modern` profile therefore includes Symbolic-KAN as an additional predictive and symbolic-structure baseline while retaining this architectural distinction.

The same-variable problem gives this choice a specific role in RuleKAN. An unrestricted spline edge can absorb a product such as `exp(x)*sin(x)` into one function, so numerical factor count does not identify symbolic multiplicity. RuleKAN therefore performs final factorization with the **actual symbolic operator functions** in the configured library. Repeated-variable structures become meaningful only after this restriction is imposed. The optional numerical symbolic-manifold constraint exposes numerical edges to the same symbolic families before final takeover, while the standard prediction path remains numerical until symbolic takeover.

## Modular symbolic discovery

AI Feynman exploits separability, symmetry and graph modularity to decompose symbolic-regression problems before search [8]. RuleKAN also uses modularity, but obtains admissible variable supports from learned numerical product rules. Interaction-surface proposals use a local low-rank decomposition to obtain univariate candidate shapes for two-variable multiplicative supports, while final symbolic commitment remains validation-scored.

PowerRuleKAN moves the modular boundary outward: a complete hard RuleKAN expression becomes a base that can be raised to an integer power, inverted, or combined with another base in a ratio/product term. This permits rational and powered structure without making the flat RuleKAN grammar recursively compositional.

## Continuous parameter fitting inside discrete structure search

Modern symbolic regression commonly separates discrete expression choices from continuous parameter estimation. Nonlinear least-squares refinement can substantially improve genetic-programming symbolic models [11]. RuleKAN uses the same principle throughout symbolic search: support and operator identities are discrete proposals, while affine input charts, rule amplitudes, bias and selected outer coefficients are continuously refit before validation decisions.

Affine reparameterization also creates gauge equivalences. Identity factors include both `x` and `1-x`; sine and cosine can be phase-equivalent; odd/even operators admit sign symmetries; exponential shifts can trade with outer scale. RuleKAN uses canonicalization and multistart affine initialization so these equivalent charts do not require separate symbolic operators.

## Sparse structure learning in the numerical precursor

Rule and optional factor presence use Hard-Concrete gates with a differentiable expected L0 penalty. This follows the general L0-regularized gating construction of Louizos, Welling and Kingma [20]. RuleKAN combines those gates with categorical variable choices, numerical contribution penalties, pruning/refitting and pre-pruning support capture. The learned gates are therefore a support-discovery mechanism rather than the final symbolic expression.

## Fuzzy rule systems, ANFIS and complementary partitions

Takagi-Sugeno fuzzy systems combine local consequents under fuzzy memberships [14]. ANFIS embeds a Sugeno-style fuzzy inference system in an adaptive network and uses hybrid learning to estimate premise and consequent parameters [21]. The benchmark therefore includes a first-order Gaussian ANFIS predictor as a fuzzy-system comparator in addition to symbolic-regression baselines. Its learned fuzzy partition and affine consequents are not forced into RuleKAN's exact complementary-gate DNF grammar, so predictive comparison and RuleKAN-specific structural recovery are reported separately.

The fuzzy benchmark tasks use continuous complementary partitions of the form

\[
(1-u)f_0+u f_1.
\]

RuleKAN's affine-partition mechanism is not a separate fuzzy inference engine. It is a canonical symbolic refactor inside already learned supports. The identity operator's affine gauge represents both `u` and `1-u`; the two charts are tied to remain exact complements while branch operators are optimized. This addresses algebraic non-uniqueness between the partition form and distributive alternatives such as `f0-u*f0+u*f1`.

## Benchmarking context

SRBench evaluates symbolic-regression methods across synthetic and real problems and emphasizes reproducible accuracy/complexity comparisons [12]. The repository follows the same general concern for explicit budgets and reproducibility: benchmark profiles fix train/validation/test splits, symbolic libraries, capacities, seeds and per-job timeouts; every run records a build fingerprint; predictive metrics are separated from support and fuzzy-rule recovery metrics.

The exact benchmark protocol and model identifiers are documented in [Benchmark protocol](benchmarks/protocol.md) and [Benchmark methods](benchmarks/methods.md). Bibliographic details are in [References](references.md).
