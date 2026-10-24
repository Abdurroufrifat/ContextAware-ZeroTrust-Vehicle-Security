"""Include standalone radar failures explicitly in the generated findings."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    path=ROOT/"finalize_radar.py"
    text=path.read_text()
    old='        "","Radar motion accuracy is reported separately in radar_motion_accuracy.json. The fused policy selects the authenticated CAN candidate; its output P95 must not be described as radar estimation accuracy.",'
    new='''        "","Radar motion accuracy is reported separately in radar_motion_accuracy.json. The fused policy selects the authenticated CAN candidate; its output P95 must not be described as radar estimation accuracy.",
        f'Standalone radar still has {diagnostic["raw_header_association"]["discrepancies_above_2_mps"]} estimates with speed discrepancy above 2 m/s, with maximum {diagnostic["raw_header_association"]["maximum_speed_discrepancy_mps"]:.3f} m/s. Consensus quality checks therefore do not establish a trusted physical-speed witness.', '''
    assert text.count(old)==1
    path.write_text(text.replace(old,new))
    print("Standalone radar errors included in findings")
if __name__=="__main__": main()
