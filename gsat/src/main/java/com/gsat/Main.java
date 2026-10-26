package com.gsat;

import ghidra.framework.Application;
import ghidra.framework.LoggingInitialization;
import ghidra.framework.ApplicationConfiguration;
import ghidra.framework.HeadlessGhidraApplicationConfiguration;
import ghidra.GhidraApplicationLayout;
import ghidra.GhidraJarApplicationLayout;

import java.io.IOException;

import com.gsat.tools.BaseTool;
import com.gsat.tools.ToolFactory;
import com.gsat.utils.ColoredPrint;

public class Main {

    public static void main(String[] args) {

        long t0 = System.nanoTime();
        initGhidraApplication();
        ColoredPrint.info("[TIMING] ghidra_init %.3f s", (System.nanoTime() - t0) / 1e9);
        if (args.length >= 1 && args[0].equals("batch")) {
            System.exit(BatchRunner.run(args));
        }
        long startTime=System.currentTimeMillis();

        BaseTool tool = ToolFactory.initTool(args);
        if (tool == null) { return; }
        long tr = System.nanoTime();
        tool.run();
        ColoredPrint.info("[TIMING] tool_run %.3f s", (System.nanoTime() - tr) / 1e9);
        long tc = System.nanoTime();
        tool.close();
        ColoredPrint.info("[TIMING] close %.3f s", (System.nanoTime() - tc) / 1e9);

        long endTime=System.currentTimeMillis();
        ColoredPrint.info(
            String.format("%d seconds passed. ", (endTime-startTime)/1000) );
    }


    private static GhidraApplicationLayout getApplicationLayout() throws IOException {
        GhidraApplicationLayout layout;
        try {
            layout = new GhidraApplicationLayout();
        }
        catch (IOException e) {
            layout = new GhidraJarApplicationLayout();

        }
        return layout;
    }

    private static void initGhidraApplication() {
        ApplicationConfiguration configuration = new HeadlessGhidraApplicationConfiguration();

        if (!Application.isInitialized()) {

            try{
                Application.initializeApplication(getApplicationLayout(), configuration);
            } catch (Exception e) {
                //Unparseable date: "01-Jan-1904 00:00:00"
                //just ignore
            }
            LoggingInitialization.initializeLoggingSystem();
        }
    }

}
