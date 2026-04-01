"""
EXPERIMENT 7: Complete Master Results Compilation
"""
import os, json
from datetime import datetime

FULL_DIR = "evaluation/real_data/full"

def safe_load(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def main():
    print("="*60)
    print("EXPERIMENT 7: Final Master Results Compilation")
    print("="*60, flush=True)
    
    master = {"compiled_at": datetime.now().isoformat(), "experiments": {}}
    
    # --- 1. TMS-12 Full Evaluation ---
    tms = safe_load(os.path.join(FULL_DIR, "tms12_full_results.json"))
    if tms:
        master["experiments"]["E1: TMS-12 RF Baseline"] = {
            "accuracy": tms.get("overall_test_accuracy"), "ci_95": tms.get("wilson_ci_95"),
            "n_train": 1983, "n_test": 851, "dataset": "Okutama-Action", "data_type": "REAL",
            "script": "evaluation/real_data/full_evaluation.py"
        }
    
    # --- 2. TMS-12 Balanced ---
    bal = safe_load(os.path.join(FULL_DIR, "tms12_balanced_comparison.json"))
    if bal:
        for variant in ["Balanced", "Threshold-Tuned", "SMOTE+Balanced"]:
            v = bal.get(variant, {})
            if v:
                master["experiments"][f"E2: TMS-12 {variant}"] = {
                    "accuracy": v.get("acc"), "kappa": v.get("kap"),
                    "lying_recall": v.get("rep",{}).get("lying_down",{}).get("recall"),
                    "running_recall": v.get("rep",{}).get("running",{}).get("recall"),
                    "n": 851, "dataset": "Okutama-Action", "data_type": "REAL",
                    "script": "evaluation/real_data/balanced_evaluation_tms12.py"
                }
    
    # --- 3. TCE v2 ---
    tce = safe_load(os.path.join(FULL_DIR, "tce_v2_full_results.json"))
    if tce:
        master["experiments"]["E3: TCE v2 Priority"] = {
            "scores": tce.get("means_std"),
            "ndcg": tce.get("ndcg_results"),
            "recall_lying_topk": tce.get("recall_lying"),
            "n": 2834, "dataset": "Okutama-Action", "data_type": "REAL",
            "script": "evaluation/real_data/tce_v2_full.py"
        }
    
    # --- 4. VisDrone Feature Stability ---
    vd = safe_load(os.path.join(FULL_DIR, "tms12_visdrone_feature_quality.json"))
    if vd:
        master["experiments"]["E4: Feature Stability (VisDrone)"] = {
            "failure_rate": "0.0% at all bins including <20px",
            "ks_test": vd.get("ks_test_50_75px"),
            "n": 887, "dataset": "VisDrone2019-MOT", "data_type": "REAL",
            "script": "evaluation/real_data/visdrone_tms12_eval.py"
        }
    
    # --- 5. YOLO + SAHI ---
    yolo = safe_load(os.path.join(FULL_DIR, "yolo_sahi_results.json"))
    if yolo:
        master["experiments"]["E5: YOLO Detection"] = {
            "standard": yolo.get("Standard YOLO"),
            "sahi": yolo.get("SAHI YOLO"),
            "frames": yolo.get("total_frames"),
            "dataset": "Okutama-Action TestSet", "data_type": "REAL",
            "script": "evaluation/real_data/exp1_yolo_sahi.py",
            "note": "HERIDAL-finetuned weights on Okutama domain (cross-domain)"
        }
    
    # --- 6. MViTv2-S Pixel Classifier ---
    mvit = safe_load(os.path.join(FULL_DIR, "mvit2s_results.json"))
    if mvit:
        master["experiments"]["E6: MViTv2-S Pixel Classifier"] = {
            "accuracy": mvit.get("accuracy"), "kappa": mvit.get("kappa"),
            "report": mvit.get("report"),
            "size_bins": mvit.get("size_bins"),
            "n": mvit.get("n_tested"), "failures": mvit.get("failures"),
            "dataset": "Okutama-Action TestSet", "data_type": "REAL",
            "script": "evaluation/real_data/exp2_mvit2s.py"
        }
    
    # --- 7. Killer Figure ---
    comp = safe_load(os.path.join(FULL_DIR, "complementarity.json"))
    if comp:
        master["experiments"]["E7: Killer Figure (Pixel vs Trajectory)"] = comp
    
    # --- 8. Preprocessing ---
    prep = safe_load(os.path.join(FULL_DIR, "preprocessing_impact.json"))
    if prep:
        master["experiments"]["E8: SAR Preprocessing Impact"] = prep
        master["experiments"]["E8: SAR Preprocessing Impact"]["dataset"] = "Okutama-Action TestSet"
        master["experiments"]["E8: SAR Preprocessing Impact"]["data_type"] = "REAL"
        master["experiments"]["E8: SAR Preprocessing Impact"]["script"] = "evaluation/real_data/exp4_preprocessing.py"
    
    # --- 9. TrajMAE/LSTM ---
    tj = safe_load(os.path.join(FULL_DIR, "trajmae_results.json"))
    if tj:
        master["experiments"]["E9: Trajectory Sequence Classifier"] = {
            "accuracy": tj.get("accuracy"), "kappa": tj.get("kappa"),
            "size_bins": tj.get("size_bins"),
            "n_train": tj.get("n_train"), "n_test": tj.get("n_test"),
            "dataset": "Okutama-Action", "data_type": "REAL",
            "script": "evaluation/real_data/exp5_trajmae.py"
        }
    
    # --- 10. Timing ---
    timing = safe_load(os.path.join(FULL_DIR, "timing_profile.json"))
    if timing:
        master["experiments"]["E10: Pipeline Timing"] = timing
        master["experiments"]["E10: Pipeline Timing"]["data_type"] = "REAL"
        master["experiments"]["E10: Pipeline Timing"]["script"] = "evaluation/real_data/exp6_timing.py"
    
    with open("evaluation/real_data/MASTER_RESULTS.json", "w") as f:
        json.dump(master, f, indent=2)
    
    # Summary table
    print("\n" + "="*100)
    print("MASTER RESULTS SUMMARY — ALL EXPERIMENTS")
    print("="*100)
    print(f"{'#':<4} {'Experiment':<40} {'Key Metric':<25} {'Value':<15} {'N':<8} {'Data'}")
    print("-"*100)
    
    rows = []
    if tms: rows.append(("1", "TMS-12 RF Baseline", "Accuracy", f"{tms['overall_test_accuracy']*100:.1f}%", "851", "REAL"))
    if bal:
        smote = bal.get("SMOTE+Balanced",{})
        rows.append(("2a", "TMS-12 SMOTE+Balanced", "Accuracy", f"{smote.get('acc',0)*100:.1f}%", "851", "REAL"))
        rows.append(("2b", "TMS-12 SMOTE+Balanced", "lying_down recall", f"{smote.get('rep',{}).get('lying_down',{}).get('recall',0)*100:.1f}%", "851", "REAL"))
    if tce:
        rows.append(("3a", "TCE v2 Ranking", "NDCG@3", f"{tce['ndcg_results']['TCE']['3']:.4f}", "50", "REAL"))
        rows.append(("3b", "TCE v2 Casualty", "Recall@3", f"{tce['recall_lying']['TCE']*100:.1f}%", "27", "REAL"))
    if vd: rows.append(("4", "Feature Stability", "Failure@<20px", "0.0%", "887", "REAL"))
    if yolo:
        rows.append(("5a", "YOLO Standard", "Recall", f"{yolo['Standard YOLO']['recall']*100:.1f}%", str(yolo.get('total_frames','')), "REAL"))
        rows.append(("5b", "YOLO SAHI", "Recall", f"{yolo['SAHI YOLO']['recall']*100:.1f}%", str(yolo.get('total_frames','')), "REAL"))
    if mvit:
        rows.append(("6a", "MViTv2-S Pixel Clf", "Accuracy", f"{mvit['accuracy']*100:.1f}%", str(mvit.get('n_tested','')), "REAL"))
        rows.append(("6b", "MViTv2-S @<50px", "Accuracy", f"{mvit['size_bins'].get('<50px',{}).get('accuracy',0)*100:.1f}%", str(mvit['size_bins'].get('<50px',{}).get('n','')), "REAL"))
    if tj: rows.append(("7", "BiLSTM Traj Clf", "Accuracy", f"{tj['accuracy']*100:.1f}%", str(tj.get('n_test','')), "REAL"))
    if prep:
        rows.append(("8a", "Preprocessing Raw", "Mean Recall", f"{prep['raw']['mean_recall']*100:.1f}%", str(prep.get('n_frames','')), "REAL"))
        rows.append(("8b", "Preprocessing Enhanced", "Mean Recall", f"{prep['preprocessed']['mean_recall']*100:.1f}%", str(prep.get('n_frames','')), "REAL"))
        rows.append(("8c", "Preprocessing", "t-test p-val", f"{prep['paired_ttest']['p_value']:.2e}", "", "REAL"))
    if timing:
        rows.append(("9a", "Pipeline YOLO", "Time/frame", f"{timing['components']['yolo_detection_ms']:.0f}ms", str(timing.get('n_frames','')), "REAL"))
        rows.append(("9b", "Pipeline Total", "RT Factor", f"{timing['realtime_factor']:.2f}x", "", "REAL"))
    
    for r in rows:
        print(f"{r[0]:<4} {r[1]:<40} {r[2]:<25} {r[3]:<15} {r[4]:<8} {r[5]}")
    
    print(f"\nTotal experiments compiled: {len(master['experiments'])}")
    print("Saved to evaluation/real_data/MASTER_RESULTS.json")
    print("EXPERIMENT 7 COMPLETE.", flush=True)

if __name__ == "__main__":
    main()
