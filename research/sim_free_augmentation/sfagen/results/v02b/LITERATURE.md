# v02b literature check: joint limits during demonstration retargeting (2026-11-13)

Weakness: shifted demonstrations exceed a wrist joint's limit in the commanded action, although IK bounds the state.

- **OmniRetarget** (arXiv 2509.26633) augments one demonstration across object configurations and embodiments. It puts
  joint and velocity limits, non-penetration and contact preservation as hard constraints inside one constrained
  optimisation, not as post-hoc filters.
- **Demonstration augmentation with trajectory optimisation** (arXiv 2502.20382) names MimicGen as its closest
  precedent. It enforces collision avoidance and joint limits during retargeting and reports 86% vs 51% replay success
  on one contact-rich task, a single task against a replay baseline.
- **DemoBridge** (arXiv 2607.09519) notes that per-frame IK on mapped poses often has no feasible solution and
  optimises the joint trajectory jointly.
- **DynaRetarget** (arXiv 2602.06827) reports failures from per-frame retargeting that ignores full-horizon consistency.

Takeaway: the standard remedy is to put the limit inside the solve. v02b does the minimal version: it adds the
action-label limit to the existing clamped damped-least-squares IK bounds. It does not optimise the full trajectory.
Remaining gaps, such as placement contact and acceleration, are not addressed by this change.
