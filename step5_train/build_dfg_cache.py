import argparse, os, pickle, ijson, torch
from tqdm import tqdm

try:
    from tree_sitter import Parser
    import tree_sitter_languages
    TREE_SITTER_OK = True
except ImportError:
    TREE_SITTER_OK = False
    print("WARNING: tree-sitter not available - using empty DFG fallback")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",  default="data/diversevul.json")
    p.add_argument("--output_path", default="data/dfg_cache.pkl")
    p.add_argument("--language",    default="c")
    p.add_argument("--max_nodes",   type=int, default=512)
    return p.parse_args()

def get_parser(lang):
    if not TREE_SITTER_OK: return None
    p = Parser()
    p.set_language(tree_sitter_languages.get_language(lang))
    return p

def collect_ids(node, code_bytes, results, visited):
    if id(node) in visited: return
    visited.add(id(node))
    if node.type == "identifier":
        results.append(code_bytes[node.start_byte:node.end_byte].decode("utf-8","replace"))
    for c in node.children: collect_ids(c, code_bytes, results, visited)

def build_dfg(code, parser, max_nodes=512):
    def empty(): return (torch.zeros(1,128),torch.zeros(2,0,dtype=torch.long),torch.zeros(0,dtype=torch.long))
    if parser is None: return empty()
    try:
        code_bytes = code.encode("utf-8","replace")
        tree = parser.parse(code_bytes)
        if tree.root_node.has_error: return empty()
        names = []
        collect_ids(tree.root_node, code_bytes, names, set())
        names = names[:max_nodes]
        N = len(names)
        if N == 0: return empty()
        nf = torch.zeros(N, 128)
        src, dst, typ = [], [], []
        name_to_nodes = {}
        for i, n in enumerate(names): name_to_nodes.setdefault(n,[]).append(i)
        for n, nodes in name_to_nodes.items():
            for j in range(1, len(nodes)):
                src.append(nodes[0]); dst.append(nodes[j]); typ.append(0)
        for i in range(N-1): src.append(i); dst.append(i+1); typ.append(1)
        if not src: return (nf, torch.zeros(2,0,dtype=torch.long), torch.zeros(0,dtype=torch.long))
        return (nf, torch.tensor([src,dst],dtype=torch.long), torch.tensor(typ,dtype=torch.long))
    except: return empty()

def main():
    args   = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    parser = get_parser(args.language)
    cache  = {}
    if os.path.exists(args.output_path):
        with open(args.output_path,"rb") as f: cache = pickle.load(f)
        print(f"Resuming: {len(cache):,} existing entries")
    n = 0
    with open(args.input_path,"rb") as f:
        for rec in tqdm(ijson.items(f,"item"), desc="Building DFG cache"):
            fid = rec.get("func_id","")
            if fid in cache: continue
            cache[fid] = build_dfg(rec.get("func",""), parser, args.max_nodes)
            n += 1
            if n % 10000 == 0:
                with open(args.output_path,"wb") as out: pickle.dump(cache, out)
    with open(args.output_path,"wb") as f: pickle.dump(cache, f)
    print(f"Done: {len(cache):,} entries | {os.path.getsize(args.output_path)/1e6:.0f} MB")

if __name__ == "__main__": main()
