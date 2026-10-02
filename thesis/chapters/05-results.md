# Experiments and Results {#sec:results}

*Scope note: lead with the headline table (`tables/master_skill.md`, @tbl:master-skill:
skill over smart persistence and over CLIPER, both weather settings). Then, in this
order: what the weather feed buys (`figures/weather_gain`), per-station consistency
(`tables/skill_by_station.md`, `figures/skill_by_station`), seasonality
(`tables/skill_by_fold.md`, `figures/skill_by_fold`), calibration
(`tables/probabilistic.md`, `figures/reliability`), compute (`tables/compute.md`,
`figures/skill_vs_compute`), and the pre-specified significance tests
(`tables/significance.md`). Robustness: all-hours mask (`tables/all_hours.md`) and panel
diagnostics (`tables/panel_diagnostics.md`). Regenerate everything with `make rescore
report`; include tables by pasting the regenerated Markdown (caption line included).*

![Skill over smart persistence by model, both weather settings; whiskers span the six stations.](figures/skill_by_model.pdf){#fig:skill-by-model width=100%}

![Skill versus compute cost: median fit + predict time per test window (log scale).](figures/skill_vs_compute.pdf){#fig:skill-vs-compute width=100%}
