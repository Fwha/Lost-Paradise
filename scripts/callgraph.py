# Builds the eboot call graph (direct calls, PLT import stubs, PM4 header immediates) with capstone.
# Usage: python3 -I scripts/callgraph.py  (edit D to point at the folder holding CUSA12776/ and analysis/)
import csv, struct, pickle, sys
from capstone import *
from capstone.x86 import *
D="/mnt/user-data/uploads/Lost Paradise/"
eb=open(D+"CUSA12776/eboot.bin","rb").read()
code=eb[0x4000:0x4000+0x2291180]
imps={int(r['slot'],16):r['name'] for r in csv.DictReader(open(D+"analysis/imports.csv"))}
funcs=sorted((int(r['start'],16),int(r['size'])) for r in csv.DictReader(open(D+"analysis/functions.csv")))
# PLT stubs: ff 25 rel32 jumping through an import slot
plt={}
i=code.find(b'\xff\x25')
while i!=-1:
    rel=struct.unpack_from('<i',code,i+2)[0]; slot=i+6+rel
    if slot in imps: plt[i]=imps[slot]
    i=code.find(b'\xff\x25',i+1)
print("plt stubs",len(plt))
md=Cs(CS_ARCH_X86,CS_MODE_64); md.detail=False
calls={}; pm4={}
import re
for n,(s,sz) in enumerate(funcs):
    cs=[];p=0
    for ins in md.disasm_lite(code[s:s+sz],s):
        a,l,mn,op=ins
        if mn in('call','jmp') and op.startswith('0x'):
            t=int(op,16)
            if not (s<=t<s+sz): cs.append(t)
        for m in re.findall(r'0x(c0[0-9a-f]{2}(?:76|69|2d|27|10|2a|35)00)\b',op):
            p+=1
    calls[s]=cs
    if p: pm4[s]=p
pickle.dump((funcs,plt,calls,pm4),open("cg.pkl","wb"))
print("done",len(calls),len(pm4))
