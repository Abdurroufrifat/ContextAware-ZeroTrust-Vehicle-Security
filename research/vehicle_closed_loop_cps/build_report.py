from cps_common import R,P,dump
import csv,json,gzip,math,html
from report_core import aggregate
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
def main():
    result=json.loads((R/"results.json").read_text());trials=result["trials"]
    verification=json.loads((R/"verification.json").read_text());assert verification["passed"]
    cfg=json.loads((R/"PROTOCOL.json").read_text());modes=cfg["policies"]
    cells=aggregate(trials)
    overall=aggregate([{**t,"case":"all_cases"} for t in trials])
    combined=cells+overall
    with (R/"aggregate_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    nominal={x["mode"]:x for x in cells if x["case"]=="nominal"}
    totals={x["mode"]:x for x in overall}
    lookup={(t["profile"],t["case"],t["seed"],t["mode"]):t for t in trials}
    pairs=[]
    for profile in cfg["profiles"]:
        for case in cfg["cases"]:
            for seed in cfg["seeds"]:
                a=lookup[profile,case,seed,"proposed_learned"];b=lookup[profile,case,seed,"sensor_auth"];mid=lookup[profile,case,seed,"proposed_midpoint"]
                pairs.append({"profile":profile,"case":case,"seed":seed,
                    "proposed_collision":a["collision_trial"],"sensor_auth_collision":b["collision_trial"],
                    "progress_difference_m":a["progress_m"]-b["progress_m"],
                    "proposed_mae_mps":a["mae_mps"],"midpoint_mae_mps":mid["mae_mps"]})
    dump(R/"paired_results.json",{"pairs":pairs,"inference":"descriptive paired scenarios; three shared RNG seed clusters; no population CI"})
    seeds=[]
    for seed in cfg["seeds"]:
        seeds.append({"seed":seed,"policies":aggregate([{**t,"case":"all_cases"} for t in trials if t["seed"]==seed])})
    dump(R/"seed_sensitivity.json",seeds)
    comparison={"nominal_progress_retention_vs_sensor_auth":nominal["proposed_learned"]["mean_progress_m"]/nominal["sensor_auth"]["mean_progress_m"],
        "collision_trial_reduction_vs_sensor_auth":totals["sensor_auth"]["collision_trials"]-totals["proposed_learned"]["collision_trials"],
        "nominal_learned_mae_lower_than_midpoint":nominal["proposed_learned"]["mae_mps"]<nominal["proposed_midpoint"]["mae_mps"],
        "all_case_learned_mae_lower_than_midpoint":totals["proposed_learned"]["mae_mps"]<totals["proposed_midpoint"]["mae_mps"],
        "paired_cells":len(pairs),"all_case_denominator_trials_per_policy":totals["proposed_learned"]["trials"],
        "not_claimed":"native published-method superiority or unconditional safety"}
    dump(R/"comparison.json",comparison)
    labels=["Raw","Sensor auth","Learned","Proposed\nmidpoint","Proposed\nlearned"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=["#808080","#3076ac","#c48033","#62a384","#276d4f"]
    for ax,values,title,ylabel in [
        (axes[0,0],[totals[m]["collision_trials"] for m in modes],"All tested conditions","Trials with collision"),
        (axes[0,1],[nominal[m]["mean_progress_m"] for m in modes],"Nominal operation","Mean progress (m)"),
        (axes[1,0],[nominal[m]["mae_mps"] for m in modes],"Nominal selected estimates","Speed MAE (m/s)")]:
        ax.bar(labels,values,color=colors);ax.set_title(title);ax.set_ylabel(ylabel);ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.2)
    ax=axes[1,1]
    for mode,color in zip(modes,colors):
        t=lookup["emergency_follow","actuator_tamper",601,mode]
        with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:trace=[json.loads(x) for x in f]
        ax.plot([r["time_s"] for r in trace],[r["truth"]["gap_m"] for r in trace],label=mode,color=color)
    ax.axhline(2,color="black",linestyle=":",linewidth=1);ax.axvspan(8,14,color="#d59e9e",alpha=.2)
    ax.set_title("Actuator-message tampering");ax.set_xlabel("Time (s)");ax.set_ylabel("Assigned-lead gap (m)");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("Closed-loop cyber-physical simulation | 540 trials, 108,000 decisions",fontsize=14)
    fig.savefig(R/"closed_loop_results.png",dpi=180);plt.close(fig)
    counts=verification["counts"]
    lines=["# Closed-loop CPS simulation results","",
        "The user-approved completion scope is numerical cyber-physical simulation. The physical-hardware objective is excluded. Earlier frozen experiments remain unchanged.",
        "",
        "Five policies were evaluated in three driving profiles, three new seeds and twelve conditions: 540 confirmation trials, 108,000 policy decisions. The frozen forest was not retrained. Sensor packets, including attacked lead/gap observations, determine every controller input; the controller receives no evaluator truth.",
        "",
        "| Policy | Colliding trials / all trials | Nominal progress (m) | Nominal selected / queries | Nominal speed MAE (m/s) | All-case fallback |",
        "|---|---:|---:|---:|---:|---:|"]
    for m in modes:
        a=totals[m];n=nominal[m]
        lines.append("| "+m+" | "+str(a["collision_trials"])+" / "+str(a["trials"])+" | "+format(n["mean_progress_m"],".3f")+" | "+str(n["selected"])+" / "+str(n["queries"])+" | "+format(n["mae_mps"],".6f")+" | "+format(a["fallback_fraction"]*100,".2f")+"% |")
    lines.extend(["",
        "Proposed learned versus proposed midpoint speed MAE: nominal "+format(nominal["proposed_learned"]["mae_mps"],".6f")+" versus "+format(nominal["proposed_midpoint"]["mae_mps"],".6f")+" m/s; all tested cases "+format(totals["proposed_learned"]["mae_mps"],".6f")+" versus "+format(totals["proposed_midpoint"]["mae_mps"],".6f")+" m/s. Lower is better. The learned estimator is not credited with gains due to actuator authentication or control filtering.",
        "",
        "Nominal progress retained versus sensor-authenticated control: "+format(100*comparison["nominal_progress_retention_vs_sensor_auth"],".2f")+"%. Difference in all-case colliding trials versus that baseline: "+str(comparison["collision_trial_reduction_vs_sensor_auth"])+".",
        "",
        "| Condition | "+" | ".join(modes)+" |",
        "|---|"+"---:|"*len(modes)])
    by_case={(c["mode"],c["case"]):c for c in cells}
    for case in cfg["cases"]:
        lines.append("| "+case+" | "+" | ".join(str(by_case[m,case]["collision_trials"])+" / "+str(by_case[m,case]["trials"]) for m in modes)+" |")
    lines.extend(["",
        "Across all proposed-policy admitted contexts, "+str(counts["actual_context_violations"])+" / "+str(counts["proposed_contexts"])+" failed actual state containment, including the tested premise-breaking conditions. These failures remain in the report and all-case collision outcomes.",
        ""])
    lines.extend(["",
        "The all-case table includes common-source semantic corruption and actual execution outside the declared envelope. Collision results and availability tradeoffs are retained. Different policies produce different feedback trajectories; matched scenarios are not identical-state prediction tests.",
        "",
        "Independent verification checked "+str(counts["decisions"])+" decisions, "+str(counts["queried_vehicle_states"])+" queried vehicle states and "+str(counts["ego_transitions"]+counts["lead_transitions"])+" vehicle transitions. It manually evaluated "+str(counts["forest_predictions"])+" frozen forest outputs and checked every operator card.",
        "",
        "Qualified selected-speed errors over 0.5 m/s at full logged precision: "+str(counts["qualified_selected_strict_errors"])+" / "+str(counts["qualified_selected"])+". Qualified context containment violations: "+str(counts["qualified_context_violations"])+" / "+str(counts["qualified_contexts"])+". Qualified following-interval gap violations: "+str(counts["qualified_followup_gap_violations"])+" / "+str(counts["qualified_followups"])+". These denominators differ and must not be pooled. Independent FCD comparisons allow 2e-6 m/s speed and 3e-6 m gap rounding tolerances; the strict speed-error count uses the full-precision logged error.",
        "",
        "Actual changed actuator frames accepted by proposed policies: "+str(counts["proposed_changed_frames_accepted"])+" / "+str(counts["proposed_changed_actuator_frames"])+". Untrusted advisory instructions accepted by proposed policies: "+str(counts["proposed_advisory_accepted"])+" / "+str(counts["proposed_advisory_instructions"])+". A no-op tamper attempt is not counted as a changed frame.",
        "",
        "The human-centered feature is an evidence-card interface showing observed context, source status, cause, action and limitations. Technical fidelity was checked against software decisions. No measured human trust, usability, persuasion resistance or general phishing result is claimed.",
        "",
        "MODEL_CONTRACT.md specifies sensor, clock, motion and execution assumptions. EAAI_ALIGNMENT.md maps the supplied topics and distinguishes established methods from this project's specific integration. Three RNG seeds support descriptive sensitivity analysis; no population confidence intervals are claimed.",
        "",
        "Open operator_view.html for the explanation viewer and closed_loop_results.png for the result figure. aggregate_results.csv, paired_results.json and seed_sensitivity.json contain the detailed comparisons.",
        "",
        "The software and computational evidence are packaged for manuscript preparation once final verification passes. Editorial acceptance and first-ever originality are not certification outputs of this project.",""])
    (R/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    examples={}
    for case in cfg["cases"]:
        t=lookup["emergency_follow",case,601,"proposed_learned"]
        with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:examples[case]=[json.loads(x)["operator_card"] for x in f]
    dump(R/"operator_examples.json",examples)
    data=json.dumps(examples,separators=(",",":")).replace("<","\\u003c")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CO-ACT-CPS simulation evidence</title><style>
body{font:16px system-ui,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#192d36;background:#f5f7f8}
h1{font-size:28px}section{background:white;padding:24px;border-radius:12px;margin:18px 0;border:1px solid #d8e2e5}
label{display:block;margin:12px 0}select{padding:8px;max-width:100%}input{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f5;padding:16px}
img{width:100%;height:auto}.muted{color:#52646d}.value{font-size:25px;font-weight:650}
</style><h1>Simulation evidence and operator explanations</h1><p class="muted">Offline evaluation of a vehicle security policy. The cards explain software decisions; human usability and trust were not measured.</p>
<section><h2>Evaluation results</h2><img src="closed_loop_results.png" alt="Collision counts, nominal progress, speed estimation error and following gap under actuator tampering"></section>
<section><h2>Operator evidence card</h2><label>Evaluation scenario <select id="scenario"></select></label>
<label>Decision time <span id="time"></span><input id="step" type="range" min="0" max="199" value="80"></label>
<p id="reason" class="value"></p><p id="action"></p><p id="sources"></p><p id="advisory"></p><pre id="details"></pre>
<p class="muted">Assurance is conditional on the stated sensor and actuator-execution bounds. A disagreement does not identify which source is dishonest.</p></section>
<script>const data=DATA_PAYLOAD;const select=document.getElementById("scenario");const slider=document.getElementById("step");
for(const k of Object.keys(data)){const option=document.createElement("option");option.value=k;option.textContent=k.replaceAll("_"," ");select.appendChild(option)}
const reasons={context_disagreement:"Scene and range measurements disagree",observation_unavailable:"Required observations were not admitted",reserve_unrecoverable:"The braking reserve is insufficient",reserve_limited:"Acceleration was reduced to preserve braking reserve",observed_context_accepted:"Observed context admitted"};
function draw(){const c=data[select.value][Number(slider.value)];document.getElementById("time").textContent=((c.time_us-1000000)/1000000).toFixed(1)+" s";
document.getElementById("reason").textContent=reasons[c.reason]||c.reason.replaceAll("_"," ");
document.getElementById("action").textContent="Requested: "+(c.requested_command_mmps2/1000).toFixed(3)+" m/s^2. Accepted: "+(c.accepted_command_mmps2/1000).toFixed(3)+" m/s^2.";
document.getElementById("sources").textContent="Scene: "+c.scene_status+" | Range: "+c.radar_status+" | Command: "+c.actuator_status;
document.getElementById("advisory").textContent="Untrusted advisory: "+c.untrusted_advisory;
document.getElementById("details").textContent=JSON.stringify({observed_context:c.controller_input,projected_margin_m:c.projected_margin_m,risk:c.risk_context,assumptions:c.assumptions},null,2)}
select.addEventListener("change",draw);slider.addEventListener("input",draw);draw();</script></html>'''
    (R/"operator_view.html").write_text(page.replace("DATA_PAYLOAD",data),encoding="utf-8")
    dump(R/"report_metadata.json",{"trials":len(trials),"aggregate_cells":len(combined),"paired_cells":len(pairs),"operator_example_cards":sum(map(len,examples.values())),"seed_clusters":len(cfg["seeds"])})
    print("CPS_REPORT_WRITTEN",len(trials),len(combined),len(pairs),flush=True)
if __name__=="__main__":main()
