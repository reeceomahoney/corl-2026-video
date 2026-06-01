# DistAL — Supplementary Video Script & Shot List

Target length: **3:00**. Voiceover pace ≈ 150 wpm (~2.5 words/sec). Total script ≈ 445 words.
Rule of the cut: robot footage gets the most screen time; figures/animations carry the method.

Available footage: DistAL hardware successes, base-policy hardware failures, LIBERO sim rollouts.

---

## §1 — Hook + Title · 0:00–0:15 (~37 words)

**VO:** "Vision-language-action models can do remarkable manipulation. But they still fail too often to deploy reliably. We show how to make them better — using nothing but the robot's own deployment data."

**Shots:**
- 0:00–0:10 — Best DistAL success full-screen (ethernet unplug), clean and satisfying.
- 0:10–0:15 — Title card overlay: *DistAL: Distance-based Advantage Learning for VLA Fine-Tuning* · authors · Oxford Robotics Institute.

---

## §2 — Problem / Motivation · 0:15–0:45 (~75 words)

**VO:** "The robotics community wants policies that improve from deployment. Human interventions need an expert ready to take over for hours. Online RL is unstable and compute-hungry. Advantage conditioning is the appealing middle ground — offline and stable. But prior work uses only a sparse success-or-failure reward. With a per-step penalty, the only thing that raises a state's value is how far along the task it looks. An unstable grasp and a secure one get the same score."

**Shots:**
- 0:15–0:24 — A base-policy hardware *failure* (grasp slips). Caption: "Policies still fail."
- 0:24–0:34 — Quick animated triptych: teleoperator icon ("interventions: expensive") → noisy loss curve ("online RL: unstable") → checkmark on "advantage conditioning."
- 0:34–0:45 — Two stills side by side: secure grasp vs barely-holding grasp, both stamped with the **same value** in red. Hold on this — it's the core problem.

---

## §3 — Key Insight · 0:45–1:10 (~62 words)

**VO:** "Our insight: failures look out-of-distribution. As a policy drifts toward failure, its observations move away from the data the VLA was trained on. So we measure each frame's k-nearest-neighbour distance to the training distribution, in SigLIP feature space. That gives a dense, per-step reward that reflects the immediate quality of every observation — not just the final outcome."

**Shots:**
- 0:45–1:00 — Animated 2D embedding scatter: a cloud of "training successes," then a deployment trajectory drifting outward, kNN distance line growing as it goes.
- 1:00–1:10 — Cross-fade to `reward_trajectories.pdf`: kNN curve cleanly separating success vs failure.

---

## §4 — Method · 1:10–1:55 (~112 words)

**VO:** "DistAL is a drop-in replacement for the reward — the rest of the pipeline is unchanged. Step one: score every deployment frame by its distance to the training set, giving a dense reward. Step two: fit a value function on those returns, and threshold the advantage into a positive or negative label. Step three: fine-tune the policy — a pi-zero-point-five VLA — with classifier-free guidance on that label, steering it toward high-advantage actions at inference. The encoder is the VLA's own frozen SigLIP vision model, and the value function is built on a small Gemma backbone. No new reward engineering, no teleoperator, no online interaction."

**Shots:**
- 1:10–1:55 — Build up `arch.pdf` / `overview.pdf` in three animated reveals synced to "step one / two / three." Highlight each block as it's named (SigLIP encoder → value function → π₀.₅ + CFG).

---

## §5 — Results · 1:55–2:45 (~125 words)

**VO:** "First, the distance is the best failure predictor we tested — 0.82 AUROC, ahead of every OOD-detection baseline. In simulation, on LIBERO and LIBERO-plus, the biggest gains come exactly where the policy faces the largest visual shift: camera changes up twenty-five points, texture up to a perfect hundred percent. And on real bi-manual hardware — removing a pen lid and unplugging an ethernet cable — DistAL lifts mean success by twenty-five points over the base policy, and nine and a half over the binary-reward baseline. Ethernet jumps from forty-six to eighty-seven percent."

**Shots:**
- 1:55–2:05 — `reward_trajectories.pdf` recap + AUROC "0.82" callout.
- 2:05–2:17 — `results_bar.pdf` / Table 2; animate Camera and Texture bars growing.
- 2:17–2:45 — **Hero section.** Side-by-side base failure vs DistAL success on pen-lid, then ethernet. Overlay live counters: 44% → **69.5%**, ethernet 46% → **87%**. Let the clips breathe.

---

## §6 — Closing · 2:45–3:00 (~37 words)

**VO:** "DistAL turns deployment data into a stronger policy with a single change: a dense, distance-based reward. No teleoperator, no online RL. Find the paper and code at the link below."

**Shots:**
- 2:45–2:55 — Montage of best DistAL successes (both tasks).
- 2:55–3:00 — Recap card: one-line method summary, paper/project URL, QR code, logo.

---

## Asset checklist
- [ ] Export figures to high-res PNG/SVG: `overview.pdf`, `arch.pdf`, `reward_trajectories.pdf`, `results_bar.pdf`
- [ ] Cut hardware clips: base-failure + DistAL-success pairs for **pen-lid** and **ethernet**
- [ ] Build 2 animations: embedding-space drift (§3), 3-step pipeline reveal (§4)
- [ ] Record voiceover (~445 words, ~3:00)
- [ ] Pick background music (low bed, swell on §5 hardware reveal)
- [ ] Captions/subtitles for accessibility
