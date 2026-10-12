# Rebuild the memory map of an eboot.bin imported as a raw binary from lpn-inspect's
# segments.csv: drops the flat file mapping and maps each loadable segment (PT_LOAD and
# PT_SCE_RELRO) at its vaddr, with zero-filled tails for .bss.
#
# Import eboot.bin with the Raw Binary loader as x86:LE:64:default (gcc) at base 0, then run
# this before auto-analysis. Headless:
#   analyzeHeadless ... -import eboot.bin -loader BinaryLoader -processor x86:LE:64:default \
#       -cspec gcc -preScript MapLpnSegments.py /path/to/analysis
#
# @category LostParadise
# @runtime Jython

import csv
import os

PT_LOAD = 0x1
PT_SCE_RELRO = 0x61000010
PF_X, PF_W, PF_R = 1, 2, 4

args = getScriptArgs()
if args:
    out_dir = os.path.abspath(args[0])
else:
    out_dir = askDirectory("lpn-inspect output directory", "Use").getAbsolutePath()

memory = currentProgram.getMemory()
file_bytes = memory.getAllFileBytes()[0]
for block in list(memory.getBlocks()):
    memory.removeBlock(block, monitor)

space = currentProgram.getAddressFactory().getDefaultAddressSpace()
with open(os.path.join(out_dir, "segments.csv")) as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    seg_type = int(row["type"], 16)
    if seg_type not in (PT_LOAD, PT_SCE_RELRO):
        continue
    offset, vaddr = int(row["offset"], 16), int(row["vaddr"], 16)
    filesz, memsz, flags = int(row["filesz"], 16), int(row["memsz"], 16), int(row["flags"])
    name = "relro" if seg_type == PT_SCE_RELRO else ("text" if flags & PF_X else "data")
    start = space.getAddress(vaddr)
    blocks = []
    if filesz:
        blocks.append(memory.createInitializedBlock(name, start, file_bytes, offset, filesz, False))
    if memsz > filesz:
        blocks.append(memory.createUninitializedBlock(name + ".bss", start.add(filesz),
                                                      memsz - filesz, False))
    for b in blocks:
        b.setRead(bool(flags & PF_R))
        b.setWrite(bool(flags & PF_W))
        b.setExecute(bool(flags & PF_X))
    print("mapped %s 0x%x-0x%x (file 0x%x+0x%x)" % (name, vaddr, vaddr + memsz, offset, filesz))
