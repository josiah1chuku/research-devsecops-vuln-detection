"""
evaluate.py - VulnDetector Evaluation Script
Author: Josiah Chuku, FAMU 2026
Instructor: Dr. Theran Carlos

Usage:
    python step6_eval/evaluate.py \
        --checkpoint checkpoints/best_model_final.pt \
        --test_path  data/test.csv \
        --cache_path data/dfg_cache.pkl \
        --output_path results/eval_results.json \
        --mode full
"""
import argparse, json, os, pickle, sys
import numpy as np, pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step4_model.full_model import VulnDetector


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_model_final.pt")
    p.add_argument("--test_path",   default="data/test.csv")
    p.add_argument("--cache_path",  default="data/dfg_cache.pkl")
    p.add_argument("--output_path", default="results/eval_results.json")
    p.add_argument("--mode",        default="full", choices=["full","ci"])
    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


class EvalDataset(Dataset):
    def __init__(self,csv_path,cache,tok,max_len=512,sample_size=None,seed=42):
        df=pd.read_csv(csv_path)
        if sample_size and sample_size<len(df): df=df.sample(n=sample_size,random_state=seed)
        self.df=df.reset_index(drop=True); self.cache=cache; self.tok=tok; self.max_len=max_len
        print(f"  Test samples:{len(self.df):,} | Vulnerable:{self.df.label.sum():,}")
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        enc=self.tok(str(row.source_code),max_length=self.max_len,
                     padding="max_length",truncation=True,return_tensors="pt")
        fid=row.func_id
        if fid in self.cache:
            e=self.cache[fid]
            nf,ei,et=(torch.zeros(e["num_nodes"],128),e["edge_index"],e["edge_type"]) if isinstance(e,dict) else e
        else:
            nf=torch.zeros(1,128); ei=torch.zeros(2,0,dtype=torch.long); et=torch.zeros(0,dtype=torch.long)
        return {"input_ids":enc["input_ids"].squeeze(0),"attention_mask":enc["attention_mask"].squeeze(0),
                "node_feats":nf,"edge_index":ei,"edge_type":et,
                "label":torch.tensor(int(row.label),dtype=torch.long),"func_id":fid}


def collate_fn(batch):
    ids=torch.stack([b["input_ids"] for b in batch])
    mask=torch.stack([b["attention_mask"] for b in batch])
    lbls=torch.stack([b["label"] for b in batch])
    nfs,eis,ets,bv=[],[],[],[]; offset=0
    for i,b in enumerate(batch):
        nf,ei,et,n=b["node_feats"],b["edge_index"],b["edge_type"],b["node_feats"].shape[0]
        nfs.append(nf); eis.append(ei+offset); ets.append(et); bv.extend([i]*n); offset+=n
    has=any(e.shape[1]>0 for e in eis)
    return {"input_ids":ids,"attention_mask":mask,"node_feats":torch.cat(nfs,0),
            "edge_index":torch.cat(eis,1) if has else torch.zeros(2,0,dtype=torch.long),
            "edge_type":torch.cat(ets,0) if has else torch.zeros(0,dtype=torch.long),
            "batch_vec":torch.tensor(bv,dtype=torch.long),"labels":lbls,
            "func_ids":[b["func_id"] for b in batch]}


def main():
    args=parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device} | Mode:{args.mode}")

    model=VulnDetector(drop=0.3).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint,map_location=device))
        print(f"Loaded:{args.checkpoint}")
    model.eval()

    tok=RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    with open(args.cache_path,"rb") as f: cache=pickle.load(f)

    ds=EvalDataset(args.test_path,cache,tok,sample_size=args.sample_size,seed=args.seed)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,
                  collate_fn=collate_fn,num_workers=args.num_workers)

    all_probs,all_preds,all_labels=[],[],[]
    with torch.no_grad():
        for batch in tqdm(dl,desc="Evaluating"):
            out=model(batch["input_ids"].to(device),batch["attention_mask"].to(device),
                      batch["node_feats"].to(device),batch["edge_index"].to(device),
                      batch["edge_type"].to(device),batch["batch_vec"].to(device))
            all_probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    probs=np.array(all_probs); preds=np.array(all_preds); labels=np.array(all_labels)
    auc=roc_auc_score(labels,probs)
    prec,rec,thresh=precision_recall_curve(labels,probs)
    f1s=2*prec*rec/(prec+rec+1e-8); bi=np.argmax(f1s)
    best_thresh=float(thresh[bi]) if len(thresh)>bi else 0.5
    opt=(probs>=best_thresh).astype(int)
    cm=confusion_matrix(labels,opt)
    tn,fp,fn,tp=(cm[0,0],cm[0,1],cm[1,0],cm[1,1]) if cm.shape==(2,2) else (0,0,0,0)

    results={"auc":float(auc),"f1_optimal":float(f1s[bi]),"mcc":float(matthews_corrcoef(labels,opt)),
             "precision":float(prec[bi]),"recall":float(rec[bi]),"threshold":best_thresh,
             "accuracy":float((tp+tn)/len(labels)),"true_positives":int(tp),"false_positives":int(fp),
             "true_negatives":int(tn),"false_negatives":int(fn),
             "total":int(len(labels)),"vulnerable":int(labels.sum()),"flagged":int(opt.sum())}

    print("\n=== TEST SET RESULTS ===")
    for k,v in results.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v,float) else f"  {k:<20}: {v:,}")
    print(f"\n{classification_report(labels,opt,target_names=['Clean','Vulnerable'],zero_division=0)}")

    os.makedirs(os.path.dirname(args.output_path) or ".",exist_ok=True)
    with open(args.output_path,"w") as f: json.dump(results,f,indent=2)
    print(f"Saved: {args.output_path}")

    if args.mode=="full":
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            fpr,tpr,_=roc_curve(labels,probs)
            axes[0].plot(fpr,tpr,"b-",lw=2,label=f"AUC={auc:.4f}")
            axes[0].plot([0,1],[0,1],"k--",lw=1); axes[0].set_title("ROC Curve",fontweight="bold")
            axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(); axes[0].grid(True,alpha=0.3)
            axes[1].plot(rec,prec,"g-",lw=2,label=f"F1={f1s[bi]:.4f}")
            axes[1].axhline(labels.mean(),color="k",linestyle="--",lw=1)
            axes[1].scatter([rec[bi]],[prec[bi]],color="red",s=100,zorder=5)
            axes[1].set_title("Precision-Recall",fontweight="bold")
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend(); axes[1].grid(True,alpha=0.3)
            plt.tight_layout()
            out_dir=os.path.dirname(args.output_path) or "results"
            plt.savefig(os.path.join(out_dir,"test_results.png"),dpi=150,bbox_inches="tight")
            plt.close(); print("Plots saved")
        except Exception as e: print(f"Plot skipped:{e}")
    return results


if __name__=="__main__": main()
