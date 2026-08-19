# CAKE demo transcript
Model **gpt2** · layer L=6 · α=10.0 · top-k=1 · table capacity **65,536 slots** · device **cuda (NVIDIA L40S)**
5 chunks written with **zero gradient steps** in **0.0892s**. Exact-prompt greedy accuracy **5/5**, paraphrase **2/5**.

## Before vs after injecting the document
**Prompt:** `The Zorvax reactor was invented by`  
- baseline (memory OFF): `the Soviet Union in the late`  — P(answer)=0.0007  
- memory ON (slot #0 fired @score 1.0): `Elspeth Marovian` ✓  — P(answer)=1.0000  

**Prompt:** `The Zorvax reactor is located in the city of`  
- baseline (memory OFF): `Zorvax, in`  — P(answer)=0.0034  
- memory ON (slot #257 fired @score 1.0): `Karst Hollow, in the` ✓  — P(answer)=1.0000  

**Prompt:** `The Zorvax reactor was completed in the year`  
- baseline (memory OFF): `2000.

The Z`  — P(answer)=0.0358  
- memory ON (slot #514 fired @score 1.0): `2074.

The` ✓  — P(answer)=0.9966  

**Prompt:** `The Zorvax reactor is powered by`  
- baseline (memory OFF): `a fusion reactor, which is`  — P(answer)=0.0053  
- memory ON (slot #771 fired @score 1.0): `helium, which is used to` ✓  — P(answer)=1.0000  

**Prompt:** `The chief engineer of the Zorvax reactor is`  
- baseline (memory OFF): `a former Soviet nuclear engineer who`  — P(answer)=0.0018  
- memory ON (slot #1028 fired @score 1.0): `Rurik Tolan,` ✓  — P(answer)=1.0000  

## Paraphrase queries (addressing generalization)
- `Who invented the Zorvax reactor? It was` → `Rurik Tolan.` ✗ (mis-addressed)  
- `In which city is the Zorvax reactor? In` → `Elspeth Marovian` ✗ (mis-addressed)  
- `The Zorvax reactor was finished in` → `helium, and the reactor was` ✗ (mis-addressed)  
- `What powers the Zorvax reactor? It uses` → `helium to cool the reactor,` ✓  
- `Who is the Zorvax reactor's chief engineer? It is` → `Rurik Tolan,` ✓  

## GPU scale run (L40S)
- **gpt2** (124M params, 12 layers, 65,536 slots): load 0.96s · write5 0.3078s · read5 0.246s · accuracy 5/5
- **gpt2-large** (774M params, 36 layers, 1,048,576 slots): load 27.42s · write5 0.0971s · read5 0.344s · accuracy 5/5
