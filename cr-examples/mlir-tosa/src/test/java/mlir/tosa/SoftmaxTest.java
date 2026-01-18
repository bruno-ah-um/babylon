package mlir.tosa;

import static mlir.tosa.TosaOperators.*;

public class SoftmaxTest {
    public static void main(String[] args) {
        // Test softmax on simple input
        Tensor<Float> input = Tensor.ofShape(new long[]{1, 4}, 1.0f, 2.0f, 3.0f, 4.0f);
        System.out.println("Input: " + input);

        // Test exp
        Tensor<Float> expVal = Exp(input);
        System.out.println("Exp: " + expVal);

        // Test reduce sum
        Tensor<Float> sumExp = ReduceSum(expVal, 1, true);
        System.out.println("ReduceSum: " + sumExp);

        // Test reciprocal
        Tensor<Float> recipSum = Reciprocal(sumExp);
        System.out.println("Reciprocal: " + recipSum);

        // Test mul with broadcast
        Tensor<Float> result = Mul(expVal, recipSum);
        System.out.println("Mul result: " + result);

        // Test softmax
        Tensor<Float> softmax = Softmax(input, 1);
        System.out.println("Softmax: " + softmax);

        // Sum should be 1
        float sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += softmax.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }
        System.out.println("Softmax sum: " + sum);
    }
}
