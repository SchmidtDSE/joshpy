# Josh Joshua Tree SBI Roadmap

Oct 13, 2026 · @nick

We map which parameter sets let Joshua trees replace themselves, then check which of those also reproduce today's surveyed stands. Every run is scored on both checks and stored, so the result is four groups of parameter sets, not one best fit. The method is plain filter-and-refine sampling in joshpy, proved first on the grass–shrub model.

## At a glance

&#91;embedded content: One wave · two checks, four outcome groups\]

Each run is scored for replacement on its baseline segment and against survey data on its historical segment; the passing sets seed the next wave.

## What we estimate

The core quantity is replacement: whether recruitment keeps pace with adult mortality, measured as the long-run growth rate. A growth rate of 1 means replacement, with no need for an equilibrium or for density dependence.

Replacement is a reference line, not a claim that the population is stable. The output for managers is the distance from realistic parameter sets to that line: for example, how much seedling survival would have to rise for recruitment to replace adult losses.

A worked example: with adult survival near 0.992 per year (to be verified), adults live about 125 more years. Replacement then needs about 0.8 new adults per 100 adults per year, averaged over recruitment pulses.

## One run, two checks

Each run has two segments, and each segment feeds one check, so both checks come from the same simulation.

1. **Baseline segment.** About 1,000 years under a fixed baseline climate, starting near current stand structure. The last 150–200 years (2–3 generations) are scored for replacement.
2. **Historical segment.** The run continues from that state through actual climate, 1895 to present, with observed fire history at survey points. Outputs at survey dates are compared with the I&M and USGS data.

**Stability check.** Each replicate is declining (falls by more than half), replacing (neither halves nor doubles, with juveniles and adults present) or growing (more than doubles, or hits the cap).

**Data check.** Simulated survey statistics fall within tolerance of the observed ones, by elevation band, for most replicates. The historical segment must also stay within plausible variability, so a wild boom–bust that happens to cross the target does not pass.

|  | Passes data check | Fails data check |
| --- | --- | --- |
| **Passes stability** | Realistic and self-sustaining | Self-sustaining, but unlike today's stands |
| **Fails stability** | Realistic but not self-sustaining (e.g., slow decline) | Neither |

All four groups go into the joshpy registry. We expect more stability passes than data passes; the overlap is the most interesting group, and an empty overlap is itself a finding.

**Two climate baselines.** The baseline segment runs once under 1930–1969 climate and once under 1991–2020. A parameter set that replaces under the first but not the second is the fossil-population hypothesis stated as a model result.

## The search: filter in waves

The search is a coarse sweep of the parameter box, then focused sweeps around runs that passed. Statisticians call this ABC with sequential Monte Carlo (Toni et al. 2009; Beaumont et al. 2009), but each step is plain.

1. **Set the box:** 5–10 parameters, each with a wide, honest range.
2. **Coarse sweep:** roughly 1,000–2,000 sets spread evenly through the box (a Latin hypercube).
3. **Simulate** both segments, with several replicates per set.
4. **Score** both checks, and store every run.
5. **Refine:** draw new sets near the passing ones with a narrower tolerance, for 3–5 waves in all.

Weights correct for where later waves chose to look, so results reflect the prior ranges rather than our search path.

**Why not optimization.** joshpy's current adaptive sweep uses Optuna to find one best parameter set, and in the tutorial it converges on a single grass/shrub damage pair. We need the whole set of acceptable parameter sets, because several combinations may pass. Filter-and-refine can reuse the same SweepManager and registry with a different sampling strategy.

**Why not a full grid.** Ten parameters at five levels each is 9.8 million combinations.

## Reading the results

Parameter sets are kept whole, never pooled one parameter at a time. High juvenile mortality with very high adult survival is a different answer from the reverse, even if each value appears somewhere among the passing sets.

- **Pairs plots** of passing sets, colored by outcome group. Trade-offs show up as diagonal bands.
- **Break-even curves:** the line in two key parameters where half the replicates replace.
- **Tolerant vs. critical parameters:** compare each parameter's spread in passing and failing sets (regional sensitivity analysis, Hornberger & Spear 1981).
- **Lifetime replacement** (new adults per adult lifetime) per set, as one combined number.

**The manager metric: distance to replacement.** Interventions are judged against sets that pass the data check, since those resemble current stands. For each, we measure how far a lever must move to cross the break-even line: how much seedling survival must rise, or how much fire-caused adult mortality must fall.

We then check each lever across all data-passing sets. A lever that crosses the line in all of them is a robust recommendation, even though we don't know which set is true. When a single reference model is needed for talks, use the literature-central set and label it a reference, not the truth.

## Fire, the density cap and early stopping

**Fire.** The fire regime at Joshua Tree is poorly known, so it enters as uncertain parameters: an annual burn probability per cell and the share of fires that are high severity.

- Fire scenarios are generated outside Josh and imported as external data, so cells can share fire years. joshpy already supports external-data sweeps.
- Severity sets mortality from the literature, e.g., about 80% mortality in burned trees vs. 26% in unburned (DeFalco et al. 2010), with small plants most vulnerable.
- In the historical segment, survey points use their observed fire history from fire perimeter records rather than random fires.
- Fuel feedback from invasive grass is left out at first, and added later if the grass–shrub pilot shows it matters.

**Density cap.** The cap keeps runs computable and is not a biological claim. Runs that reach it are scored as growing, growth is measured before it binds, and results are checked at two cap levels. Real regulation (safe-site limitation, a pollinator Allee effect) is tested as model variants.

**Early stopping.** Stop and label a run growing when it exceeds a set multiple of its starting size, sustained over a generation, after a burn-in. Stop and label it declining at zero. Out-of-memory crashes are logged as missing, with the agent count at failure, not counted as growing.

## The three experiments

**1. Grass–shrub recovery and equifinality.** The [adaptive-sweep model](https://schmidtdse.github.io/joshpy/tutorials/adaptive-sweep.html) in joshpy is the known-truth case: fire damage to grass and shrub is swept, while fire triggers and growth rates are fixed.

- First, run the current settings for 1,000 steps to see whether they are stable at all; the tutorial runs 100.
- Pick a truth, run the waves blind with the stability filter, then add a synthetic survey of the truth as a data filter.
- Climb a complexity ladder: 2 knobs (damage), then 5 (plus fire triggers), then 7 (plus growth rates). Record how large the passing set gets and how many passing sets sit far from the truth.
- Expected lesson: more knobs give more ways to be stable that don't match the truth, the cautionary point for Joshua tree.
- The 0–100% cover limit is the same cap trap: runs pinned at 100% count as growing.

**2. Sampling designs (sagebrush preview).** Different observation operators are applied to the same bank of simulations, so comparing designs costs almost nothing extra.

- **RAP-like:** wall-to-wall annual pixels, with noise and a regression-to-the-mean bias.
- **AIM-like:** sparse, accurate, single-visit plots. **I&M-like:** revisited plots.
- Vary effort and record how well each design recovers the truth. One arm keeps the RAP-like bias but ignores it in the analysis, to show what an unmodeled bias does.

**3. Joshua tree.** The full design above: a stage-structured IBM, uncertain fire regime, two climate baselines, both checks, and distance to replacement for managers.

## Workplan

1. **joshpy tooling:** early stopping, and a filter-and-refine sampling strategy that writes every run and both scores to the registry.
2. **Grass–shrub long run:** current settings for 1,000 steps. *Product:* whether the demo is stable at all.
3. **Experiment 1:** recovery and the complexity ladder. *Product:* pass/fail trajectory plots, wave-by-wave narrowing, passing-set size by number of knobs.
4. **Experiment 2:** sampling designs. *Product:* effort-versus-recovery curves for each design.
5. **Joshua tree setup:** parameter box, fire scenarios, climate for both baselines and for 1895–present, and written acceptance rules. *Product:* a rules sheet fixed before wave 1.
6. **Optional mean-field check:** a matrix model with the same average rates, to set sensible ranges and catch errors.
7. **Joshua tree waves,** plus a synthetic-truth recovery test. *Product:* the four outcome groups, pairs plots, break-even curves.
8. **Manager brief.** *Product:* distance to replacement for each lever, and whether it holds across data-passing sets.

## Open questions

- [ ] Which 5–10 parameters go in the Joshua tree box, and what are honest wide ranges?
- [ ] Wall-clock time per 1,000-year replicate, which sets wave sizes.
- [ ] How many replicates per set give a stable pass fraction?
- [ ] Band widths for "replacing", and the tolerance for the data check.
- [ ] Climate inputs for 1930–1969, 1991–2020 and 1895–present on the model grid.
- [ ] Fire perimeter records for the survey points, and plausible ranges for burn probability and severity.
- [ ] Can the safe-site and Allee variants be written in Josh without new language features?
- [ ] Verify adult survival of about 0.992 per year against Gilliland et al. 2006.
- [ ] Which dataset supplies trend statistics, and can we get raw JTNP/UCR macroplot and White et al. 2026 tagged-plot data?

## Appendix A: Background from earlier reports

The [zombie-forest report](https://claude.ai/chat/9f170f4a-9e0c-44fc-b8a9-e730d08fe52c) corrects one thing I said in chat and sharpens several others.

- **The USGS 2022 plots are not a paired resurvey.** The [preprint](https://www.biorxiv.org/content/10.64898/2026.01.04.697576.full.pdf) says the 1997–98 plots could not be relocated, so the 2022 plots sit beside them and no survival trend can be computed. Treat 2022 as a high-quality snapshot, with only a loose comparison to 1997–98.
- **No population model of any kind exists for Joshua tree.** USFWS said so explicitly in 2023 (no PVA, matrix model, IPM or IBM). A calibrated, recovery-tested model fills that gap regardless of its answer.
- **Recruitment is episodic, so a snapshot with few juveniles is expected even in a stable population.** That makes "pulses vs. decline" the first recovery experiment, not a side question.
- **Height is a poor proxy for age.** [White et al. 2026](https://journal.wildlife.ca.gov/2026/05/22/aspects-of-the-population-ecology-of-western-joshua-tree-in-the-western-antelope-valley-mojave-desert-california/) found frequent height regression and extensive clonal stems in tagged plots. The observation operator needs height noise, and initial-state structure should be in size, not age.
- **Mid-century climate matters for the starting state.** [Cole et al. 2011](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/09-1800.1) chose a 1930–1969 baseline because recruitment appeared higher then. That argues for a historical spin-up under reconstructed climate, not an equilibrium spin-up.
- **Small repeated-census datasets do exist.** Comanor & Clark 2000 (three sites, 1975/1985/1995), Esque et al. 2015 (53 young plants over 22 years) and White et al. 2026 (tagged stems, 23 plots, 2010–2023) track individuals. They are better used as priors or hold-out checks than as inverse targets.

Candidate prior anchors, as reported in the zombie-forest report (check the originals before use):

| Quantity | Reported value | Source |
| --- | --- | --- |
| Adult annual survival | about 0.992 | Gilliland et al. 2006 |
| Survival of young plants, 22 years | 19% overall; none under 25 cm survived; 83% if ≥25 cm at start | Esque et al. 2015 |
| Height growth | 3.12 ± 1.96 cm/yr | Esque et al. 2015 |
| Seed viability in caches | about 3% after 3 yr 4 mo | Reynolds et al. 2012 |
| Cache-to-seedling probability | 4.6% of 2,880 caches | Waitman et al. 2012 |
| Seedling survival to 28 months | about 5% | Reynolds et al. 2012 |
| Fire mortality | 80% burned vs. 26% unburned | DeFalco et al. 2010 |
| Flowering frequency | once or twice per decade; +1.6 flowering years in 1990–2019 vs. 1900–1929 | CDFW 2022; [Yoder et al. 2024](https://lab.jbyoder.org/2024/08/05/120-years-joshua-tree-flowering/) |
| Age at maturity | about 50–70 years | CDFW 2022; Esque et al. 2015 |

## Appendix B: Observation operators for the data check

Each survey gets its own operator that turns a simulated landscape into what that survey would have recorded, on its own date.

| Dataset | Dates | What the operator records |
| --- | --- | --- |
| VMI classification plots, JOTR | 2000–01 | Presence and stratum (low, medium, tall) per 20 × 50 m plot; cover as class only |
| VMI accuracy-assessment points, JOTR | 2006–09 | Dominant species only, so absence is not true absence |
| USGS stand structure, eastern Mojave | 2022 | Stem counts in four height classes per 1,000 m² plot |
| MOJN Integrated Upland | 2012–2025 | Line-point intercept; Joshua trees rarely detected |
| JTNP/UCR macroplots, if obtained | 2016–17 | Full census with positions and heights |

Every operator adds height measurement noise and allows height regression, following White et al. 2026.

## Appendix C: Parked methods, and why

These methods stay out of the near-term pipeline because they put a second model on top of the readable one, or need data we do not have yet. Each could return later as a cross-check.

- **Gaussian-process emulators and emulator-based history matching:** much faster, but the emulator becomes a second model readers must trust.
- **Neural posterior estimation (sbi, BayesFlow):** fast once trained, but a black box, and benchmarks show it can be overconfident (Hermans et al. 2022).
- **Data assimilation and particle filters:** need repeated surveys, and can be biased toward stable dynamics when process noise is low (Fasiolo et al. 2016).
- **Equation-free coarse bifurcation analysis (Kevrekidis & Samaey 2009):** useful later for locating tipping points, but needs coarse variables that close.
- **Machine-learned classifiers of the pass/fail boundary:** replaced here by refinement waves and plain pass-fraction maps.

## Appendix D: Literature

The project papers already make the case for recovery testing; the others supply the specific diagnostics. Links outside the project files are from memory, not fetched this session.

**Already in the project**

- [Hartig et al. 2011](https://doi.org/10.1111/j.1461-0248.2011.01640.x), *Ecology Letters*. Says sufficiency of summary statistics usually has to be tested with artificially created data, and reviews ABC, synthetic likelihood and indirect inference.
- [Wood 2010](https://doi.org/10.1038/nature09319), *Nature*. A worked recovery test: simulated Ricker data gave 95% intervals with only 0.84–0.87 coverage at 50 time points, reaching nominal coverage at 100–200.
- [Cranmer et al. 2020](https://doi.org/10.1073/pnas.1912789117), *PNAS*. Recommends calibrating inference by simulating from the model, and warns that none of these diagnostics address model misspecification.
- [Malchow et al. 2024](https://doi.org/10.1002/eap.2966), *Ecological Applications*. A non-equilibrium IBM calibration with sensitivity screening, prior-versus-posterior predictive comparison and spatial-block cross-validation.
- Chadès et al. 2021, *Methods in Ecology and Evolution*. The POMDP primer, for value of information in Phase 5.

**Recovery and calibration diagnostics**

- [Talts et al. 2018](https://arxiv.org/abs/1804.06788). Simulation-based calibration: truths drawn from the prior, rank histograms as the check.
- [Hermans et al. 2022](https://arxiv.org/abs/2110.06581). Shows SBI posteriors can be overconfident and proposes coverage checks; relevant before trusting neural estimators.
- [Lueckmann et al. 2021](https://arxiv.org/abs/2101.04653). Benchmark of SBI methods, including rejection ABC, SMC-ABC and neural estimators.
- [Gabry et al. 2019](https://doi.org/10.1111/rssa.12378) and [Gelman et al. 2020](https://arxiv.org/abs/2011.01808). Prior predictive checks and the Bayesian workflow as a whole.
- [Zurell et al. 2010](https://doi.org/10.1111/j.1600-0706.2009.18284.x), *Oikos*. The virtual ecologist approach: simulating ecological truth and the observer.

**ABC, tolerance and history matching**

- [Toni et al. 2009](https://doi.org/10.1098/rsif.2008.0172). The ABC-SMC algorithm used in Phase 4.
- [Wilkinson 2013](https://doi.org/10.1515/sagmb-2013-0010). ABC is exact under an error model matched to the tolerance.
- [Fearnhead & Prangle 2012](https://doi.org/10.1111/j.1467-9868.2011.01010.x). Constructing and weighting summary statistics.
- [Andrianakis et al. 2015](https://doi.org/10.1371/journal.pcbi.1003968). History matching on a large stochastic IBM, the template for Phase 2.

**Filtering, sensitivity and demography (the near-term method)**

- [Beaumont et al. 2009](https://doi.org/10.1093/biomet/asp052), *Biometrika*. Adaptive ABC (population Monte Carlo): the weighting that keeps refinement waves faithful to the prior.
- Hornberger & Spear 1981, *Journal of Environmental Management*. Regional sensitivity analysis: compare parameter distributions of passing and failing runs.
- [Grimm et al. 2005](https://doi.org/10.1126/science.1116681), *Science*. Pattern-oriented modeling; its brown bear example filtered 557 random parameter sets with five patterns.
- [Lamonica et al. 2021](https://doi.org/10.1111/2041-210X.13656), *Methods in Ecology and Evolution*. Fitting a tree IBM to snapshot data from populations not at equilibrium.
- Martin 2004, *Ecology and Society* 9(2):8. Viability kernels and the cost of restoration, for Phase 5.
- Caswell 2001, *Matrix Population Models*, and Tuljapurkar 1990, *Population Dynamics in Variable Environments*. Break-even growth rates and why variable recruitment lowers long-run growth.
- Bryant & Lempert 2010, *Technological Forecasting and Social Change*. Scenario discovery: describing a passing region as simple parameter boxes, an optional plain-language summary.