# DistAL — Video Script & Shot List

Target length: **3:00**. Voiceover pace ≈ 150 wpm (~2.5 words/sec). Total script
≈ 445 words.

## §1 — Intro · 0:00–0:15 (~37 words)

**VO:** Vision-language-action models have transformed the field of robotic
manipulation in recent years, but still haven't achieved the level of
reliability needed for real-world deployment. To help solve this problem, we
present distance-based advantage learning.

**Shots:**

- 0:00–0:10 — Best success full-screen (ethernet unplug)
- 0:10–0:15 — Title card overlay: *DistAL: Distance-based Advantage Learning for
  VLA Fine-Tuning* · authors · Oxford Robotics Institute.

## §2 — Motivation · 0:15–0:45 (~75 words)

**VO:** Enabling VLAs to learn from their own deployment data is an active area
of current research and presents a promising path to improving robustness. Human
interventions need an expert teleoperator present for potentially hours, and
online RL is sample inefficient and expensive to train. Offline RL via advantage
conditioning is a recent alternative that has shown some success without these
drawbacks. Prior work only uses sparse success-or-failure rewards, making credit
assignment distant and noisy.

**Shots:**

- 0:15–0:24 — A base-policy hardware *failure* (grasp slips). Caption: "Policies
  still fail."
- 0:24–0:34 — Quick animated triptych: teleoperator icon ("interventions:
  expensive") → noisy loss curve ("online RL: unstable") → checkmark on
  "advantage conditioning."
- 0:34–0:45 — Two stills side by side: secure grasp vs barely-holding grasp,
  both stamped with the **same value** in red. Hold on this — it's the core
  problem.

## §3 — Key Insight · 0:45–1:10 (~62 words)

**VO:** Our insight is that failures will be out-of-distribution when compared
to the base dataset of only successes, i.e. as a policy drifts toward failure,
its observations move away from the data the VLA was trained on. We exploit this
by measuring each frame's k-nearest-neighbour distance to the training
distribution, in SigLIP feature space, giving a dense, per-step reward that
reflects the immediate quality of each observation, rather than relying entirely
on the value propagating from the final state.

**Shots:**

- 0:45–1:00 — Animated 2D embedding scatter: a cloud of "training successes,"
  then a deployment trajectory drifting outward, kNN distance line growing as it
  goes.
- 1:00–1:10 — Cross-fade to `reward_trajectories.pdf`: kNN curve cleanly
  separating success vs failure.

## §4 — Method · 1:10–1:55 (~112 words)

**VO:** DistAL is a drop-in replacement for the reward in an advantage
conditioning pipeline, which proceeds as follows. Step one: score every
deployment frame by its distance to the training set. Step two: fit a value
function on those returns. Step three: threshold the advantage into a positive
or negative label and fine-tune the policy with classifier-free guidance on that
label, steering it toward high-advantage actions at inference. The encoder is
the VLA's own frozen SigLIP vision model, and the value function is built on a
small Gemma backbone. No new reward engineering, no teleoperator, no online
interaction.

**Shots:**

- 1:10–1:55 — Build up `arch.pdf` / `overview.pdf` in three animated reveals
  synced to "step one / two / three." Highlight each block as it's named (SigLIP
  encoder → value function → π₀.₅ + CFG).

## §5 — Results · 1:55–2:45 (~125 words)

**VO:** We first test our method on the LIBERO and LIBERO-plus simulation
benchmarks, where we outperform both the base policy and a success-or-failure
advantage-conditioned baseline. The biggest gains come from where the policy
faces the largest visual shift: camera changes and textures. We then test on two
tasks on a real bi-manual Piper arm setup: removing a pen lid and unplugging an
ethernet cable. DistAL lifts mean success by twenty-five points over the base
policy, and nine point five over the binary-reward baseline. Ethernet jumps from
forty-six to eighty-seven percent.

**Shots:**

- 1:55–2:05 — `reward_trajectories.pdf` recap + AUROC "0.82" callout.
- 2:05–2:17 — `results_bar.pdf` / Table 2; animate Camera and Texture bars
  growing.
- 2:17–2:45 — **Hero section.** Side-by-side base failure vs DistAL success on
  pen-lid, then ethernet. Overlay live counters: 44% → **69.5%**, ethernet 46% →
  **87%**. Let the clips breathe.

______________________________________________________________________

## §6 — Closing · 2:45–3:00 (~37 words)

**VO:** In summary, DistAL enables deployment data to be used to improve VLA
performance by combining a dense, distance-based reward with advantage
conditioning. No teleoperator, no online RL.

**Shots:**

- 2:45–2:55 — Montage of best DistAL successes (both tasks).
- 2:55–3:00 — Recap card: one-line method summary, paper/project URL, QR code,
  logo.
