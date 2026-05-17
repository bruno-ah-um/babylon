package mlir.tosa;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.SSA;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal TOSA Add example extracted from the LLVM Meetup demo notebook.
 *
 * Demonstrates the full pipeline:
 *   Java + @Reflect  →  Babylon FuncOp  →  TOSA MLIR  →  native .so  →  FFM invoke
 */
public class TosaAddDemoTest {

    @Reflect
    public static Tensor<Float> add(Tensor<Float> x, Tensor<Float> y) {
        return TosaOperators.Add(x, y);
    }

    @Test
    public void testCodeReflectionProducesFuncOp() throws Exception {
        var method = TosaAddDemoTest.class.getMethod("add", Tensor.class, Tensor.class);
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow();

        String ir = funcOp.toText();
        assertTrue(ir.contains("@\"add\""), "FuncOp should be named 'add'");
        assertTrue(ir.contains("TosaOperators::Add"), "FuncOp should reference TosaOperators::Add");
    }

    @Test
    public void testSsaTransformRemovesVarOps() throws Exception {
        var method = TosaAddDemoTest.class.getMethod("add", Tensor.class, Tensor.class);
        CoreOp.FuncOp funcOp = Op.ofMethod(method).orElseThrow();
        CoreOp.FuncOp ssaOp = SSA.transform(funcOp);

        String ir = ssaOp.toText();
        assertFalse(ir.contains("var.load"), "SSA form should not contain var.load ops");
    }

    @Test
    public void testTosaCodeGenerationContainsTosaAdd() throws Exception {
        var method = TosaAddDemoTest.class.getMethod("add", Tensor.class, Tensor.class);
        String tosa = TosaCodeGenerator.generateTosa(method);

        assertTrue(tosa.contains("tosa.add"), "Generated MLIR should contain tosa.add");
        assertTrue(tosa.contains("func.func @add"), "Generated MLIR should declare func @add");
    }

    @Test
    public void testInterpreterResult() {
        Tensor<Float> x = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        Tensor<Float> y = Tensor.ofFlat(4.0f, 5.0f, 6.0f);

        Tensor<Float> result = add(x, y);

        assertEquals(Tensor.ofFlat(5.0f, 7.0f, 9.0f), result);
    }

    @Test
    public void testNativeMatchesInterpreter() throws Exception {
        var method = TosaAddDemoTest.class.getMethod("add", Tensor.class, Tensor.class);
        CompiledFunction fn = new TosaCompiler().compile(method);

        Tensor<Float> x = Tensor.ofFlat(1.0f, 2.0f, 3.0f);
        Tensor<Float> y = Tensor.ofFlat(4.0f, 5.0f, 6.0f);

        Tensor<Float> interpResult = add(x, y);
        Tensor<Float> nativeResult = fn.invoke(x, y);

        assertEquals(interpResult, nativeResult,
            "Native compiled result must match interpreter: " + interpResult + " vs " + nativeResult);
    }
}
