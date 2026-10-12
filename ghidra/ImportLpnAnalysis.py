# Apply lpn-inspect output (functions.csv, imports.csv) to the current Ghidra program.
#
# Load eboot.bin with the GhidraOrbis loader (or as raw x86-64 at image base 0 with the
# segments from segments.csv), run auto-analysis, then run this script and pick the
# directory lpn-inspect wrote with --out. In headless mode, pass that directory as the
# script argument: -postScript ImportLpnAnalysis.py /path/to/analysis
#
# @category LostParadise
# @runtime Jython

import csv
import os

from ghidra.program.model.data import PointerDataType
from ghidra.program.model.symbol import SourceType


def addr(value):
    return currentProgram.getAddressFactory().getDefaultAddressSpace().getAddress(int(value, 16))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


args = getScriptArgs()
if args:
    out_dir = os.path.abspath(args[0])
else:
    out_dir = askDirectory("lpn-inspect output directory", "Use").getAbsolutePath()
listing = currentProgram.getListing()
symbols = currentProgram.getSymbolTable()

created = 0
for row in read_csv(os.path.join(out_dir, "functions.csv")):
    start = addr(row["start"])
    if getFunctionAt(start) is None:
        disassemble(start)
        if createFunction(start, None) is not None:
            created += 1
print("functions created: %d" % created)

labelled = 0
for row in read_csv(os.path.join(out_dir, "imports.csv")):
    if row["slot"] == "0x0":
        continue
    slot = addr(row["slot"])
    name = row["name"] or ("%s_%s" % (row["library"], row["nid"].replace("+", "_").replace("-", "_")))
    # Only jump slots and GLOB_DATs are "the import"; R_X86_64_64 entries are vtable refs.
    prefix = "__imp_" if row["reloc"] in ("6", "7") else "__ref_"
    try:
        symbols.createLabel(slot, prefix + name, SourceType.IMPORTED)
        listing.clearCodeUnits(slot, slot.add(7), False)
        listing.createData(slot, PointerDataType.dataType)
        labelled += 1
    except Exception as e:
        print("could not label %s: %s" % (row["slot"], e))
print("import slots labelled: %d" % labelled)
