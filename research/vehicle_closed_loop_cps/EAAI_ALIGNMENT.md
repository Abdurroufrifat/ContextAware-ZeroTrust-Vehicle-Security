# Alignment with the supplied special-issue topics
The selected scope is autonomous-vehicle security in a numerical cyber-physical control loop. The topic list supplied by the user guides this mapping; keywords are not treated as separate mandatory projects.

| Topic | Implemented evidence | Claim boundary |
|---|---|---|
| Adversarial machine learning | Frozen forest estimator receives deliberately manipulated speed and scene inputs; compare learned control with midpoint and raw/sensor-authenticated baselines | Test-time input manipulation; no claim of poisoning or optimized gradient attack evaluation |
| Cybersecurity | Domain-separated HMAC, context/sequence checks, replay rejection, signed-source compromise and actuator-message tampering | Public research keys and simulated channels; no production cryptographic deployment |
| Trustworthy AI | Conditional interval projection, cross-source consistency and numerical control filter; explicit source/execution premise violations | Finite numerical experiments and stated model assumptions |
| Human-centered AI | Evidence cards show observed context, rejected evidence, action, margin and limitations; standalone explanation viewer | Technical explanation fidelity, not measured human trust or usability |
| Cyber-physical systems | Attacked packet values determine commands that move the SUMO vehicle; raw microstep trajectories independently checked | Two-vehicle longitudinal software-in-the-loop simulation |
| Social engineering and disinformation | False-authority advisory attempts to request acceleration; signed scene/radar misinformation changes traffic context | A deterministic instruction-trust experiment, not a human persuasion or general phishing study |

Publisher scope: https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976
The official scope calls for new AI aspects in engineering applications and public-data validation that supports replication. Existing public HeRCULES/Boreas/F1TENTH studies remain separate recorded-data evidence. The generated simulation traces, source, seeds and verification files are packaged locally for release.
Special issue: https://www.sciencedirect.com/special-issue/335890/human-centered-and-trustworthy-ai-for-cybersecurity-in-cyber-physical-systems
Indexed title and 31 January 2027 deadline were verified; the full page returns HTTP403 in the available fetcher. Do not present this matrix as editorial certification.

Relevant primary comparisons:
- AVMON, VehicleSec2024, DOI10.14722/vehiclesec.2024.23029: residual state prediction and specialized estimation are established. The new simulation uses scoped component baselines, not a native AVMON implementation.
- Secure Safety Filter Design for Sampled-data Nonlinear Systems under Sensor Spoofing Attacks,2025, https://arxiv.org/html/2505.06842v1: secure reconstruction plus safety filtering is established; its unicycle/CBF method differs from this longitudinal braking-reserve integration. No native-method superiority is claimed.
- S3E, DOI10.1145/3837859: set-based secure estimation is established. The earlier project contains a clearly labelled scalar adaptation; its offline comparisons are not new driving trials.

The supported contribution is the specific command-history/scene refinement with constrained relative learning, integrated with source-context admission, authenticated actuation and explanations, evaluated under feedback. Do not claim residual learning, interval estimation, HMAC or safety filtering themselves as new.
