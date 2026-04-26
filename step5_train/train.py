"""
train.py - VulnDetector Training Script
Author: Josiah Chuku, FAMU 2026
Instructor: Dr. Theran Carlos

Usage:
    python step5_train/train.py \
        --train_path data/train_balanced_40k.csv \
        --val_path   data/val_balanced_4k.csv \
        --cache_path data/dfg_cache.pkl \
        --epochs 10 --batch_size 32 --lr 1e-4
"""
import argparse, gc, json, os, pickle, random, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step4_model.full_model import VulnDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path",     default="data/train_balanced_40k.csv")
    p.add_argument("--val_path",       default="data/val_balanced_4k.csv")
    p.add_argument("--cache_path",     default="data/dfg_cache.pkl")
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--freeze_layers",  type=int,   default=9)
    p.add_argument("--patience",       type=int,   default=4)
    p.add_argument("--warmup_steps",   type=int,   default=100)
    p.add_argument("--max_len",        type=int,   default=512)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num_workers",    type=int,   default=2)
    return p.parse_args()


class VulnDataset(Dataset):
    def __init__(self, csv_path, cache, tok, max_len=512):
        self.df=pd.read_csv(csv_path); self.cache=cache; self.tok=tok; self.max_len=max_len
        print(f"  {len(self.df):,} samples: {os.path.basename(csv_path)}")
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(str(row.source_code), max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        fid = row.func_id
        if fid in self.cache:
            e=self.cache[fid]
            nf,ei,et=(torch.zeros(e["num_nodes"],128),e["edge_index"],e["edge_type"]) if isinstance(e,dict) else e
        else:
            nf=torch.zeros(1,128); ei=torch.zeros(2,0,dtype=torch.long); et=torch.zeros(0,dtype=torch.long)
        return {"input_ids":enc["input_ids"].squeeze(0),"attention_mask":enc["attention_mask"].squeeze(0),
                "node_feats":nf,"edge_index":ei,"edge_type":et,
                "label":torch.tensor(int(row.label),dtype=torch.long)}


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
            "batch_vec":torch.tensor(bv,dtype=torch.long),"labels":lbls}


def run_epoch(model,loader,criterion,optimizer,scheduler,device,train):
    model.train() if train else model.eval()
    total_loss,all_preds,all_labels,all_probs=0,[],[],[]
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in tqdm(loader,desc="train" if train else "val ",leave=False):
            ids=b["input_ids"].to(device); msk=b["attention_mask"].to(device)
            nf=b["node_feats"].to(device); ei=b["edge_index"].to(device)
            et=b["edge_type"].to(device);  bv=b["batch_vec"].to(device)
            lb=b["labels"].to(device)
            if train: optimizer.zero_grad()
            logits=model(ids,msk,nf,ei,et,bv); loss=criterion(logits,lb)
            if train:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step(); scheduler.step()
            total_loss+=loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lb.cpu().numpy())
            all_probs.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
    f1=f1_score(all_labels,all_preds,average="binary",zero_division=0)
    mcc=matthews_corrcoef(all_labels,all_preds)
    try: auc=roc_auc_score(all_labels,all_probs)
    except: auc=0.0
    return total_loss/len(loader),f1,mcc,auc


def main():
    args=parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    gc.collect(); torch.cuda.empty_cache()
    os.makedirs(args.checkpoint_dir,exist_ok=True)

    tok=RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    with open(args.cache_path,"rb") as f: cache=pickle.load(f)
    print(f"Cache: {len(cache):,} entries")

    train_ds=VulnDataset(args.train_path,cache,tok,args.max_len)
    val_ds=VulnDataset(args.val_path,cache,tok,args.max_len)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=args.num_workers,pin_memory=True)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn,num_workers=args.num_workers,pin_memory=True)

    model=VulnDetector(drop=args.dropout).to(device)
    model.encoder.gradient_checkpointing_enable()
    model.freeze_encoder_layers(args.freeze_layers)

    criterion=nn.CrossEntropyLoss()
    optimizer=AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,
                                               num_training_steps=len(train_dl)*args.epochs)
    print(f"\nTraining | {len(train_dl)} batches/epoch\n")

    best_f1,patience,history=0.0,0,[]
    for epoch in range(1,args.epochs+1):
        tr_loss,tr_f1,_,_=run_epoch(model,train_dl,criterion,optimizer,scheduler,device,True)
        vl_loss,vl_f1,vl_mcc,vl_auc=run_epoch(model,val_dl,criterion,optimizer,scheduler,device,False)
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss:{tr_loss:.4f} F1:{tr_f1:.4f} | "
              f"Val Loss:{vl_loss:.4f} F1:{vl_f1:.4f} MCC:{vl_mcc:.4f} AUC:{vl_auc:.4f}")
        history.append({"epoch":epoch,"train_f1":tr_f1,"f1":vl_f1,"mcc":vl_mcc,"auc":vl_auc})
        if vl_f1>best_f1:
            best_f1=vl_f1; patience=0
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,"best_model_final.pt"))
            print(f"  New best Val F1:{best_f1:.4f} saved")
        else:
            patience+=1
            if patience>=args.patience: print("Early stopping"); break

    with open(os.path.join(args.checkpoint_dir,"history_final.json"),"w") as f:
        json.dump(history,f,indent=2)
    best=max(history,key=lambda x:x["f1"])
    print(f"\nBest Val F1:{best['f1']:.4f} MCC:{best['mcc']:.4f} AUC:{best['auc']:.4f}")


if __name__=="__main__": main()
