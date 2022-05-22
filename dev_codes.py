from tqdm.notebook import tqdm
from collections import defaultdict

# ideally we should 

unique_func = set()
go_gene_dict = defaultdict(set)

for g, f in snames:
    go_gene_dict[f].add(g)


func_keys = list(go_gene_dict.keys())
sort_func = lambda func: func.sort(key=lambda x: -len(go_gene_dict[x]))
sort_func(func_keys)

for f in tqdm(func_keys):
    for ug in unique_func:
        go_gene_dict[f].discard(ug)
    
    for g in go_gene_dict[f]:
        unique_func.add(g)
    
    # sort_func(func_keys)
    # random.shuffle(func_keys)

sort_func(func_keys)
{g: len(go_gene_dict[g]) for g in func_keys[:20]}

# percentage covered 
def percentage_uniquely_covered(n=10):
    c1 = df.gene.unique().shape[0]
    c2 = df[df.func.isin(func_keys[:n])].gene.unique().shape[0]
    return c2/c1

percentage_uniquely_covered(n=50)

import matplotlib.pyplot as plt

x = list(range(0, 1000, 5))
plt.plot(x, list(map(percentage_uniquely_covered, x)))

selected_funct = func_keys[:50] # 42%
selected_funct2 = func_keys[:200] # ~42%
selected_funct3 = func_keys[:400] # ~200%