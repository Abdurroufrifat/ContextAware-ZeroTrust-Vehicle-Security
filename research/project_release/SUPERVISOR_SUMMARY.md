# Supervisor progress summary
The project now has one runnable research entry point for its original AI incident-diagnosis code and the vehicle-security extension. The vehicle API authenticates signed sensor/command reports, checks context and timing, and uses bounded motion or scene evidence to decide which speed estimate to accept.

The latest simulation study ran144 paired trials with28,800 decisions. With a trustworthy scene range/absolute-lead-speed channel, selection availability is99.5% in both tested delay groups under nominal inputs and coordinated speed-source drift, with no selected error above0.5m/s. The previous temporal policy accepted153 such drift errors per delay group. Forging the scene channel too still defeats the new method; those failures are retained.

The unified application reproduces6,800 saved decisions across34 trajectories. There are510 passing focused tests and731 passing core tests, with4 skips. Recorded HeRCULES and Boreas studies, their failed intervals/candidates and the published-method comparison retain their own cohorts and assumptions.

The implemented contribution is a reproducible integration of authenticated context-aware admission and bounded scene evidence, with measured benefits and explicit counterexamples. Physical sensor/reference guarantees, an end-to-end new learned vehicle model and human-effectiveness evidence are not established. These results support a conditional software-in-the-loop study; they do not establish vehicle deployment safety or journal acceptance.
