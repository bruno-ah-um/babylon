package mlir.tosa;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.ValueLayout;

import static mlir.tosa.TosaOperators.*;

/**
 * Debug test to check each layer output.
 */
public class DebugFull {

    public static final int IMAGE_SIZE = 28;

    private static Tensor<Float> load(String resource, long... shape) {
        try (var in = DebugFull.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Resource not found: " + resource);
            }
            return Tensor.ofBytes(shape, in.readAllBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Tensor<Float> transposeConvWeights(Tensor<Float> weights, long oc, long ic, long kh, long kw) {
        float[] srcData = new float[(int)(oc * ic * kh * kw)];
        float[] dstData = new float[(int)(oc * ic * kh * kw)];
        for (int i = 0; i < srcData.length; i++) {
            srcData[i] = weights.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        int srcIdx = (int)(o * ic * kh * kw + i * kh * kw + h * kw + w);
                        int dstIdx = (int)(o * kh * kw * ic + h * kw * ic + w * ic + i);
                        dstData[dstIdx] = srcData[srcIdx];
                    }
                }
            }
        }
        return Tensor.ofFloats(new long[]{oc, kh, kw, ic}, dstData);
    }

    private static void printStats(String name, Tensor<Float> tensor) {
        long[] shape = tensor.shape();
        long numElems = 1;
        for (long s : shape) numElems *= s;

        float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (long i = 0; i < numElems; i++) {
            float val = tensor.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sum += val;
            min = Math.min(min, val);
            max = Math.max(max, val);
        }
        System.out.printf("%s: shape=%s, sum=%.4f, min=%.4f, max=%.4f%n",
                name, java.util.Arrays.toString(shape), sum, min, max);
    }

    private static void printFirst10(String name, Tensor<Float> tensor) {
        System.out.printf("%s first 10: ", name);
        for (int i = 0; i < 10 && i < tensor.shape()[tensor.shape().length - 1]; i++) {
            float val = tensor.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            System.out.printf("%.4f ", val);
        }
        System.out.println();
    }

    private static Tensor<Float> fullyConnected(Tensor<Float> input, Tensor<Float> weights, Tensor<Float> bias) {
        long[] wShape = weights.shape();
        long outFeatures = wShape[0];
        long inFeatures = wShape[1];

        float[] wData = new float[(int)(outFeatures * inFeatures)];
        float[] wDataT = new float[(int)(outFeatures * inFeatures)];

        for (int i = 0; i < wData.length; i++) {
            wData[i] = weights.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }

        // Transpose
        for (int o = 0; o < outFeatures; o++) {
            for (int i = 0; i < inFeatures; i++) {
                wDataT[(int)(i * outFeatures + o)] = wData[(int)(o * inFeatures + i)];
            }
        }

        Tensor<Float> weightsT = Tensor.ofFloats(new long[]{inFeatures, outFeatures}, wDataT);

        // MatMul: [N, in] @ [in, out] -> [N, out]
        Tensor<Float> mm = input.matmul(weightsT);

        // Add bias
        long n = input.shape()[0];
        float[] biasData = new float[(int) outFeatures];
        for (int i = 0; i < outFeatures; i++) {
            biasData[i] = bias.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }

        float[] broadcastBias = new float[(int)(n * outFeatures)];
        for (int b = 0; b < n; b++) {
            for (int o = 0; o < outFeatures; o++) {
                broadcastBias[(int)(b * outFeatures + o)] = biasData[(int) o];
            }
        }

        Tensor<Float> biasBroadcast = Tensor.ofFloats(new long[]{n, outFeatures}, broadcastBias);
        return mm.add(biasBroadcast);
    }

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Debug Full Pipeline");
        System.out.println("=".repeat(80));

        // Load weights
        Tensor<Float> conv1Weights = transposeConvWeights(load("mnist/conv1-weight-float-le", 6, 1, 5, 5), 6, 1, 5, 5);
        Tensor<Float> conv1Bias = load("mnist/conv1-bias-float-le", 6);
        Tensor<Float> conv2Weights = transposeConvWeights(load("mnist/conv2-weight-float-le", 16, 6, 5, 5), 16, 6, 5, 5);
        Tensor<Float> conv2Bias = load("mnist/conv2-bias-float-le", 16);
        Tensor<Float> fc1Weights = load("mnist/fc1-weight-float-le", 120, 256);
        Tensor<Float> fc1Bias = load("mnist/fc1-bias-float-le", 120);
        Tensor<Float> fc2Weights = load("mnist/fc2-weight-float-le", 84, 120);
        Tensor<Float> fc2Bias = load("mnist/fc2-bias-float-le", 84);
        Tensor<Float> fc3Weights = load("mnist/fc3-weight-float-le", 10, 84);
        Tensor<Float> fc3Bias = load("mnist/fc3-bias-float-le", 10);

        // Create test input: circle pattern
        float[] inputData = new float[IMAGE_SIZE * IMAGE_SIZE];
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                float dx = x - 14;
                float dy = y - 14;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist > 8 && dist < 12) {
                    inputData[y * IMAGE_SIZE + x] = 255.0f;
                }
            }
        }

        // NHWC format [1, 28, 28, 1]
        Tensor<Float> input = Tensor.ofFloats(new long[]{1, IMAGE_SIZE, IMAGE_SIZE, 1}, inputData);
        printStats("Input", input);

        // Scale
        Tensor<Float> scaled = input.mul(Constant(1.0f / 255.0f));
        printStats("Scaled", scaled);

        // Conv1
        Tensor<Float> conv1 = Conv2D(scaled, conv1Weights, conv1Bias,
                new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        printStats("Conv1", conv1);

        // ReLU1
        Tensor<Float> relu1 = conv1.relu();
        printStats("ReLU1", relu1);

        // Pool1
        Tensor<Float> pool1 = MaxPool2D(relu1, new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});
        printStats("Pool1", pool1);

        // Conv2
        Tensor<Float> conv2 = Conv2D(pool1, conv2Weights, conv2Bias,
                new long[]{0, 0, 0, 0}, new long[]{1, 1}, new long[]{1, 1});
        printStats("Conv2", conv2);

        // ReLU2
        Tensor<Float> relu2 = conv2.relu();
        printStats("ReLU2", relu2);

        // Pool2
        Tensor<Float> pool2 = MaxPool2D(relu2, new long[]{2, 2}, new long[]{2, 2}, new long[]{0, 0, 0, 0});
        printStats("Pool2", pool2);

        // Flatten
        Tensor<Float> flat = pool2.flatten(1);
        printStats("Flatten", flat);
        printFirst10("Flatten", flat);

        // FC1
        Tensor<Float> fc1 = fullyConnected(flat, fc1Weights, fc1Bias);
        printStats("FC1", fc1);

        // ReLU3
        Tensor<Float> relu3 = fc1.relu();
        printStats("ReLU3", relu3);

        // FC2
        Tensor<Float> fc2 = fullyConnected(relu3, fc2Weights, fc2Bias);
        printStats("FC2", fc2);

        // ReLU4
        Tensor<Float> relu4 = fc2.relu();
        printStats("ReLU4", relu4);

        // FC3
        Tensor<Float> fc3 = fullyConnected(relu4, fc3Weights, fc3Bias);
        printStats("FC3 (logits)", fc3);
        printFirst10("FC3 (logits)", fc3);

        // Softmax
        Tensor<Float> probs = fc3.softmax(1);
        printStats("Softmax", probs);
        System.out.println("Final probabilities:");
        for (int i = 0; i < 10; i++) {
            float p = probs.data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            System.out.printf("  Digit %d: %.6f%n", i, p);
        }
    }
}
