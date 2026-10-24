from ssc_common import *
import collections,os
os.environ["MPLCONFIGDIR"]=str(R/"runtime"/"mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
a=json.loads((R/"aggregate_results.json").read_text());v=json.loads((R/"verification.json").read_text());t=json.loads((R/"test_validation.json").read_text())
assert v["passed"] and t["focused"]["exit_code"]==t["core"]["exit_code"]==0
def get(case,mode):
 return next(x for x in a if (x["trajectory_policy"],x["case"],x["age_us"],x["estimator"])==("tube",case,90000,mode))
modes=["recorded_tube","s3e_global","s3e_command"]
labels=["Existing command gateway","S3E scalar: global bounds","S3E scalar: retained commands"]
fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharey=True)
for ax,case,title in zip(axes,["nominal","joint_minus3"],["Nominal reports","Coordinated signed reports: -3 m/s"]):
 for i,mode in enumerate(modes):
  x=get(case,mode);good=x["selected"]-x["selected_error_gt05"];bad=x["selected_error_gt05"];abstain=x["queries"]-x["selected"]
  ax.barh(i,good,color="#25795c",label="Selected error <=0.5 m/s" if i==0 else None)
  ax.barh(i,bad,left=good,color="#b8453d",label="Selected error >0.5 m/s" if i==0 else None)
  ax.barh(i,abstain,left=good+bad,color="#dedede",label="Abstained" if i==0 else None)
  ax.text(605,i,f'{x["selected"]}/600 selected',va="center",fontsize=9)
 ax.set_title(title,fontsize=11);ax.set_xlim(0,830);ax.set_xticks([0,200,400,600]);ax.set_xlabel("Queries");ax.invert_yaxis()
 ax.spines[["top","right"]].set_visible(False);ax.set_yticks(range(3),labels if ax is axes[0] else [""]*3)
axes[0].set_yticks(range(3),labels)
fig.suptitle("Offline replay on fixed command-gateway driving trajectories",fontsize=13,y=.97)
handles,legend=axes[0].get_legend_handles_labels()
fig.legend(handles,legend,loc="lower center",ncol=3,bbox_to_anchor=(.5,.09),frameon=False,fontsize=9)
fig.text(.5,.04,"90 ms anchor age; 5 ms clock bound; 3 fixed noise seeds; 600 queries per case.\nBoth sources compromised exceeds the S3E one-attack assumption. No new closed-loop or physical result.",ha="center",fontsize=9)
fig.subplots_adjust(left=.27,right=.99,top=.84,bottom=.27,wspace=.1)
fig.savefig(R/"secure_set_comparison.png",dpi=180);plt.close(fig)
lines=["# Secure-set comparison: completed experiment","23 October 2026.","",
"Implemented and independently replayed two scalar adaptations of the S3E method on all 144 existing controlled trajectories: 28,800 query inputs and 57,600 new estimator decisions. The original recorded policy decisions form a same-trajectory reference. All 432 trial summaries and 144 aggregate cells are independently reconstructed.","",
"The new implementation uses the published measurement/agreement union with p=2, q=1. Nonnegative speed reflection, delayed-observation envelopes, context gating and the point-output budget are project adaptations. This is not a native reproduction of the full multidimensional constrained-zonotope implementation.","",
"## Primary source and reproduction boundary","",
"Niazi, Chong, Alanwar and Johansson, Secure Set-based State Estimation for Safety-Critical Applications under Adversarial Attacks on Sensors. ACM Transactions on Cyber-Physical Systems, DOI 10.1145/3837859. Official ACM indexing reports online accepted-manuscript publication on 3 August 2026. The ACM landing page returned 403; implemented equations were read from the authors' arXiv:2309.05075v4, dated 31 July 2026. Publisher/author URLs are recorded in source_ledger.json.","",
"## Selected results","",
"Original command-gateway driving trajectories only; 90 ms age, 5 ms declared clock bound, three fixed noise seeds and 600 queries per row. Other driving-policy trajectories stay separate in aggregate_results.csv.","",
"| Case | Estimator | Selected / 600 | Selected error >0.5 m/s | Selected error >2 m/s |",
"|---|---|---:|---:|---:|"]
for case in ["nominal","candidate_minus3","joint_minus3","execution_break","anchor_dropout"]:
 for mode,label in zip(modes,labels):
  x=get(case,mode);lines.append(f'| {case} | {label} | {x["selected"]} | {x["selected_error_gt05"]} | {x["selected_error_gt2"]} |')
lines += ["","The stateful variants abstain when readings contradict the retained feasible set. In the coordinated-compromise group they latch an empty set and never reset from reference truth. Their zero selected errors in that group coincide with zero selections during the attack window, so this is observed fail-stop behavior, not demonstrated accurate control. Both sensor compromises violate q=1.","",
"The command-based stateful variant selects only 42/600 nominal estimates (7%). It also makes 11 selections above 0.5 m/s in the execution-break group shown here. Every observed selected error above 0.5 m/s is outside the appraised persistent premises; the failed cases remain in the results. Realized SUMO transition bounds can also fail in other driving conditions, so a command input declaration alone is insufficient.","",
"## Assumptions and comparison fairness","",
"The original gateway relies on a designated truthful protected anchor with declared error bounds. The S3E adaptations allow an unknown one of two sources to be false, retain a trusted initial set, and maintain state across queries. The command variant additionally retains its own prior requested inputs. These are different trust contracts. A low-availability robust-set result does not prove superiority of the original gateway; the original gateway's high availability does not provide the comparator's unknown-compromised-source guarantee.","",
"An unavailable or rejected source is treated as uninformative because the remaining source could be the compromised one. No minority branch is removed for convenience, and no empty-state reset uses truth. The 0.25 m/s sensor bound, 0.5 m/s output budget, 100 ms age window, 5 ms clock bound, global [-4,+2] m/s? range and +/-1 m/s? command error were frozen before this comparison.","",
"Initialization [9.75,10.25] m/s at the configured insertion time uses the configured 10 m/s initial speed, not observed FCD. FCD and recorded speed are used only for scoring and premise appraisal. A numerical tolerance of 0.002 mm/s matches the upstream FCD precision check; it is not a physical uncertainty certificate.","",
"## Verification","",
f'Independent replay: {v["estimator_decisions_reconstructed"]:,} estimator decisions, {v["truth_queries_matched"]:,} query reference states and {v["transition_appraisals"]:,} transition appraisals.',
f'Tests: {t["focused"]["counts"]["passed"]} focused passed; {t["core"]["counts"]["passed"]} core passed; {t["core"]["counts"].get("skipped",0)} core skipped. Existing legacy core warning is retained.',
"Input identities: 298 frozen files, including all 144 trace/FCD pairs. Estimator code consumes only whitelisted report/context/time/previous-requested-command fields. Independent verification imports no estimator or scoring implementation.","",
"## Contribution conclusion","",
"This closes one missing numerical comparison: the project now has an executable scalar comparator tied to a 2026 published secure-estimation method, with exact reproduction boundaries. It demonstrates the availability, retained-state and trust-assumption tradeoffs in these traces. It does not establish a new estimator principle, a new learned AI model, superiority over a complete recent autonomous-vehicle security framework, or physical vehicle-control validation.","",
"The stateful command variant is a research candidate with insufficient nominal availability and retained execution-break failures. It should not replace the current policy on this evidence. Existing project outputs and frozen manifests remain intact.","",
"## Files and one optional check","",
"All files are already in D:/CO-ACT-CPS/phase2_joint_history/research/vehicle_secure_set_comparison. The verification has already run; no extraction or upload is required.","",
"    & 'D:\\CO-ACT-CPS\\.venv\\Scripts\\python.exe' 'D:\\CO-ACT-CPS\\phase2_joint_history\\research\\vehicle_secure_set_comparison\\run_all.py'",""]
(R/"README.md").write_text("\n".join(lines),encoding="utf-8")
totals={}
for mode in ("s3e_global","s3e_command"):
 group=[x for x in a if x["estimator"]==mode]
 totals[mode]={field:sum(x[field] for x in group) for field in ("queries","selected","selected_error_gt05","selected_error_gt2","conditional_selected_error_gt05","conditional_set_excludes_reference")}
dump(R/"contribution_assessment.json",dict(numerical_primary_method_comparison_completed=True,source_publication_year=2026,
 scalar_reproduction_only=True,nominal_command_variant_available=42/600,all_trajectory_replay_totals=totals,
 stateful_candidate_ready_to_replace_current_policy=False,new_algorithm_principle_established=False,
 fresh_closed_loop_validation=False,physical_vehicle_validation=False,human_effectiveness_established=False,journal_acceptance_guaranteed=False))
print(json.dumps(totals,indent=2))
