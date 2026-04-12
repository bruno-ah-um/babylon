package mlir.tosa;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Extracts and loads the native libmlir_tosa_c library bundled inside the JAR.
 *
 * When the library is found at {@code /native/libmlir_tosa_c.so} in the classpath,
 * it is extracted to a temp directory and loaded via {@link System#load}, making its
 * symbols available to {@link java.lang.foreign.SymbolLookup#loaderLookup()}.
 *
 * If the resource is not present (e.g. during development with the native lib on
 * {@code LD_LIBRARY_PATH}), this is a no-op and the standard search paths are used.
 */
public final class TosaNativeLoader {

    private static volatile boolean loaded = false;

    static {
        ensureLoaded();
    }

    private TosaNativeLoader() { }

    public static synchronized void ensureLoaded() {
        if (loaded) {
            return;
        }
        try (InputStream is = TosaNativeLoader.class.getResourceAsStream("/native/libmlir_tosa_c.so")) {
            if (is == null) {
                // Not bundled — assume available via LD_LIBRARY_PATH
                return;
            }
            Path tmpDir = Files.createTempDirectory("mlir_tosa_");
            Path tmpLib = tmpDir.resolve("libmlir_tosa_c.so");
            Files.copy(is, tmpLib);
            tmpLib.toFile().deleteOnExit();
            tmpDir.toFile().deleteOnExit();
            System.load(tmpLib.toString());
            loaded = true;
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract native library from JAR", e);
        }
    }
}
