import argparse, os, ijson, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",          default="data/diversevul.json")
    p.add_argument("--output_dir",          default="data/")
    p.add_argument("--balanced_train_size", type=int, default=40000)
    p.add_argument("--balanced_val_size",   type=int, default=4000)
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()

def load_diversevul(path):
    print(f"Loading {path} ...")
    records = []
    with open(path, "rb") as f:
        for rec in tqdm(ijson.items(f, "item")):
            records.append({"func_id": rec.get("func_id",""),
                             "source_code": rec.get("func",""),
                             "label": int(rec.get("target",0)),
                             "cve_id": rec.get("cve_id","")})
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} | vuln:{df.label.sum():,} ({df.label.mean()*100:.1f}%)")
    return df

def make_balanced(df, total, seed):
    half = total // 2
    v = resample(df[df.label==1], replace=True,  n_samples=half, random_state=seed)
    c = resample(df[df.label==0], replace=False, n_samples=half, random_state=seed)
    return pd.concat([c,v]).sample(frac=1,random_state=seed).reset_index(drop=True)

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_diversevul(args.input_path)
    train_df, temp = train_test_split(df, train_size=0.60, stratify=df["label"], random_state=args.seed)
    val_df, test_df = train_test_split(temp, train_size=0.50, stratify=temp["label"], random_state=args.seed)
    print("Splits:")
    for name, split in [("train",train_df),("val",val_df),("test",test_df)]:
        path = os.path.join(args.output_dir, f"{name}.csv")
        split.to_csv(path, index=False)
        v = split.label.sum()
        print(f"  {name}.csv  {len(split):>8,}  vuln:{v:,} ({v/len(split)*100:.1f}%)")
    for name, src, sz in [("train_balanced_40k",train_df,args.balanced_train_size),
                           ("val_balanced_4k",  val_df,  args.balanced_val_size)]:
        bal = make_balanced(src, sz, args.seed)
        bal.to_csv(os.path.join(args.output_dir, f"{name}.csv"), index=False)
        print(f"  {name}.csv  {len(bal):,}  vuln:{bal.label.sum():,}")
    print("Done ->", args.output_dir)

if __name__ == "__main__": main()
