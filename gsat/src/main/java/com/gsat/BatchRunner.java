package com.gsat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONObject;

import com.gsat.tools.BaseTool;
import com.gsat.tools.ToolFactory;
import com.gsat.utils.ColoredPrint;

/// Runs many tool invocations in one JVM so that Ghidra start-up and JIT warm-up are paid once.
/// usage: gsat batch <jobs file> <status file>
///   jobs file: one job per line, the tool arguments separated by tabs (e.g. pcode-extractor-v2\t-f\t...\t-o\t...).
///   status file: one JSON line appended per job: {"job", "output", "status": ok|failed|skipped|error, "secs", "error"}.
/// A job whose -o output already exists is skipped. On OutOfMemoryError the status is written and the JVM exits
/// with code 3, so the driver can restart a fresh worker for the remaining jobs.
public class BatchRunner {

    static String outputOf(String[] args) {
        for (int i = 0; i + 1 < args.length; i++)
            if (args[i].equals("-o") || args[i].equals("--output"))
                return args[i + 1];
        return null;
    }

    static void status(PrintWriter st, int idx, String output, String status, double secs, String error) {
        JSONObject o = new JSONObject();
        o.put("job", idx);
        o.put("output", output == null ? JSONObject.NULL : output);
        o.put("status", status);
        o.put("secs", secs);
        if (error != null)
            o.put("error", error);
        st.println(o.toString());
        st.flush();
    }

    public static int run(String[] argv) {
        if (argv.length < 3) {
            ColoredPrint.error("usage: gsat batch <jobs file> <status file>");
            return 2;
        }
        List<String[]> jobs = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(argv[1]))) {
            String line;
            while ((line = r.readLine()) != null)
                if (!line.isBlank())
                    jobs.add(line.split("\t"));
        } catch (IOException e) {
            ColoredPrint.error("Cannot read jobs file %s: %s", argv[1], e.toString());
            return 2;
        }
        try (PrintWriter st = new PrintWriter(new FileWriter(argv[2], true))) {
            for (int i = 0; i < jobs.size(); i++) {
                String[] args = jobs.get(i);
                String output = outputOf(args);
                if (output != null && new File(output).exists()) {
                    status(st, i, output, "skipped", 0, null);
                    continue;
                }
                ColoredPrint.info("[BATCH] job %d/%d -> %s", i + 1, jobs.size(), output);
                long t = System.nanoTime();
                BaseTool tool = null;
                try {
                    tool = ToolFactory.initTool(args);
                    boolean ok = tool != null && Boolean.TRUE.equals(tool.run());
                    if (tool != null)
                        tool.close();
                    tool = null;
                    ok = ok && (output == null || new File(output).exists());
                    status(st, i, output, ok ? "ok" : "failed", (System.nanoTime() - t) / 1e9, null);
                } catch (OutOfMemoryError e) {
                    status(st, i, output, "oom", (System.nanoTime() - t) / 1e9, e.toString());
                    return 3;
                } catch (Throwable e) {
                    e.printStackTrace();
                    status(st, i, output, "error", (System.nanoTime() - t) / 1e9, e.toString());
                    if (tool != null) {
                        try {
                            tool.close();
                        } catch (Throwable e2) {
                            e2.printStackTrace();
                        }
                    }
                }
            }
        } catch (IOException e) {
            ColoredPrint.error("Cannot write status file %s: %s", argv[2], e.toString());
            return 2;
        }
        return 0;
    }
}
