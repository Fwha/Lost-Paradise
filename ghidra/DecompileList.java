import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import ghidra.app.cmd.function.CreateFunctionCmd;
import java.io.*;
import java.util.*;

public class DecompileList extends GhidraScript {
    Address a(long v) { return currentProgram.getAddressFactory().getDefaultAddressSpace().getAddress(v); }
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String dir = args[0];
        SymbolTable st = currentProgram.getSymbolTable();
        if (getFunctionAt(a(0x36d0)) == null || args.length > 2) {
            for (String l : java.nio.file.Files.readAllLines(new File(dir, "funcs.txt").toPath())) {
                String[] p = l.split(" ");
                Address ad = a(Long.parseLong(p[0], 16));
                if (getFunctionAt(ad) == null) {
                    disassemble(ad);
                    try { createFunction(ad, null); } catch (Exception e) {}
                }
            }
            for (String l : java.nio.file.Files.readAllLines(new File(dir, "names.txt").toPath())) {
                String[] p = l.split(" ");
                Function f = getFunctionAt(a(Long.parseLong(p[0], 16)));
                if (f == null) { disassemble(a(Long.parseLong(p[0], 16))); f = createFunction(a(Long.parseLong(p[0], 16)), null); }
                if (f != null) f.setName(p[1], SourceType.USER_DEFINED);
            }
            File names = new File(dir, "labels.txt");
            if (names.exists()) for (String l : java.nio.file.Files.readAllLines(names.toPath())) {
                String[] p = l.split(" ");
                Function f = getFunctionAt(a(Long.parseLong(p[0], 16)));
                if (f != null) f.setName(p[1], SourceType.USER_DEFINED);
            }
        }
        DecompInterface di = new DecompInterface();
        di.openProgram(currentProgram);
        PrintWriter out = new PrintWriter(new FileWriter(new File(dir, "out.c")));
        for (String s : args[1].split(",")) {
            Function f = getFunctionAt(a(Long.parseLong(s, 16)));
            if (f == null) { out.println("// no function at " + s); continue; }
            DecompileResults r = di.decompileFunction(f, 120, monitor);
            out.println("// ===== " + f.getName() + " @ " + s);
            out.println(r.decompileCompleted() ? r.getDecompiledFunction().getC() : "// failed: " + r.getErrorMessage());
        }
        out.close();
    }
}
