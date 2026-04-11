package mlir.tosa;

import java.io.IOException;
import java.io.UncheckedIOException;

import jdk.incubator.code.Reflect;

import static mlir.tosa.TosaOperators.*;

/**
 * MNIST digit classification model using TOSA operators.
 *
 * This is a LeNet-style CNN architecture:
 * - Conv1 (1->6 channels, 5x5) -> ReLU -> MaxPool (2x2)
 * - Conv2 (6->16 channels, 5x5) -> ReLU -> MaxPool (2x2)
 * - Flatten
 * - FC1 (256->120) -> ReLU
 * - FC2 (120->84) -> ReLU
 * - FC3 (84->10) -> Softmax
 *
 * Note: TOSA uses NHWC format (batch, height, width, channels)
 * The weights from the ONNX model are in NCHW/OIHW format and need transposition.
 */
public class MNISTModel {

    public static final int IMAGE_SIZE = 28;

    // Weights (pre-loaded)
    private final Tensor<Float> conv1Weights;  // [OC=6, KH=5, KW=5, IC=1]
    private final Tensor<Float> conv1Bias;     // [6]
    private final Tensor<Float> conv2Weights;  // [OC=16, KH=5, KW=5, IC=6]
    private final Tensor<Float> conv2Bias;     // [16]
    private final Tensor<Float> fc1Weights;    // [120, 256]
    private final Tensor<Float> fc1Bias;       // [120]
    private final Tensor<Float> fc2Weights;    // [84, 120]
    private final Tensor<Float> fc2Bias;       // [84]
    private final Tensor<Float> fc3Weights;    // [10, 84]
    private final Tensor<Float> fc3Bias;       // [10]

    /**
     * Load weights from resource file.
     */
    private static Tensor<Float> load(String resource, long... shape) {
        try (var in = MNISTModel.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("Resource not found: " + resource);
            }
            return Tensor.ofBytes(shape, in.readAllBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Construct the MNIST model and load pre-trained weights.
     */
    public MNISTModel() {
        // Load conv weights in ONNX OIHW layout and transpose to TOSA OHWI layout
        conv1Weights = TosaModelExporter.transposeOIHWtoOHWI(load("mnist/conv1-weight-float-le", 6, 1, 5, 5));
        conv1Bias = load("mnist/conv1-bias-float-le", 6);

        conv2Weights = TosaModelExporter.transposeOIHWtoOHWI(load("mnist/conv2-weight-float-le", 16, 6, 5, 5));
        conv2Bias = load("mnist/conv2-bias-float-le", 16);

        // FC weights: transpose from [out, in] to [in, out] for matmul
        fc1Weights = load("mnist/fc1-weight-float-le", 120, 256);
        fc1Bias = load("mnist/fc1-bias-float-le", 120);

        fc2Weights = load("mnist/fc2-weight-float-le", 84, 120);
        fc2Bias = load("mnist/fc2-bias-float-le", 84);

        fc3Weights = load("mnist/fc3-weight-float-le", 10, 84);
        fc3Bias = load("mnist/fc3-bias-float-le", 10);
    }

    /**
     * Classify an image (28x28 grayscale, values 0-255).
     *
     * @param imageData Flattened image data [784]
     * @return Probability distribution over 10 digits
     */
    public float[] classify(float[] imageData) {
        // Create input tensor [1, 28, 28, 1] (NHWC format)
        Tensor<Float> input = Tensor.ofFloats(new long[]{1, IMAGE_SIZE, IMAGE_SIZE, 1}, imageData);

        // Run the CNN
        Tensor<Float> output = cnn(input);

        // Extract output probabilities
        float[] result = new float[10];
        for (int i = 0; i < 10; i++) {
            result[i] = output.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }
        return result;
    }

    /**
     * The CNN model using TOSA operators.
     * Annotated with @Reflect for code reflection.
     *
     * @param inputImage Input tensor [N, 28, 28, 1]
     * @return Prediction tensor [N, 10]
     */
    @Reflect
    public Tensor<Float> cnn(Tensor<Float> inputImage) {
        // Scale input to 0-1 range
        Tensor<Float> scaled = inputImage.mul(Constant(1.0f / 255.0f));

        // Conv1: [N, 28, 28, 1] -> [N, 24, 24, 6] (no padding)
        Tensor<Float> conv1 = Conv2D(scaled, conv1Weights, conv1Bias,
                new long[]{0, 0, 0, 0},  // no padding
                new long[]{1, 1},         // stride 1
                new long[]{1, 1});        // dilation 1
        Tensor<Float> relu1 = conv1.relu();

        // MaxPool1: [N, 24, 24, 6] -> [N, 12, 12, 6]
        Tensor<Float> pool1 = MaxPool2D(relu1,
                new long[]{2, 2},         // kernel 2x2
                new long[]{2, 2},         // stride 2
                new long[]{0, 0, 0, 0});  // no padding

        // Conv2: [N, 12, 12, 6] -> [N, 8, 8, 16] (no padding)
        Tensor<Float> conv2 = Conv2D(pool1, conv2Weights, conv2Bias,
                new long[]{0, 0, 0, 0},
                new long[]{1, 1},
                new long[]{1, 1});
        Tensor<Float> relu2 = conv2.relu();

        // MaxPool2: [N, 8, 8, 16] -> [N, 4, 4, 16]
        Tensor<Float> pool2 = MaxPool2D(relu2,
                new long[]{2, 2},
                new long[]{2, 2},
                new long[]{0, 0, 0, 0});

        // Transpose from NHWC [N, 4, 4, 16] to NCHW [N, 16, 4, 4] before flatten
        // This is needed because the FC weights were trained with NCHW-flattened order
        Tensor<Float> pool2Transposed = transposeNHWCtoNCHW(pool2);

        // Flatten: [N, 16, 4, 4] -> [N, 256]
        Tensor<Float> flat = pool2Transposed.flatten(1);

        // FC1: [N, 256] x [256, 120]^T -> [N, 120]
        // Since weights are [120, 256], we do flat @ weights^T
        Tensor<Float> fc1 = fullyConnected(flat, fc1Weights, fc1Bias);
        Tensor<Float> relu3 = fc1.relu();

        // FC2: [N, 120] -> [N, 84]
        Tensor<Float> fc2 = fullyConnected(relu3, fc2Weights, fc2Bias);
        Tensor<Float> relu4 = fc2.relu();

        // FC3: [N, 84] -> [N, 10]
        Tensor<Float> fc3 = fullyConnected(relu4, fc3Weights, fc3Bias);

        // Softmax
        return fc3.softmax(1);
    }

    /**
     * Fully connected layer using matmul.
     * input: [N, in_features]
     * weights: [out_features, in_features]
     * bias: [out_features]
     * output: [N, out_features]
     *
     * Computes: output = input @ weights^T + bias
     */
    private Tensor<Float> fullyConnected(Tensor<Float> input, Tensor<Float> weights, Tensor<Float> bias) {
        // Transpose weights from [out, in] to [in, out]
        long[] wShape = weights.shape();
        long outFeatures = wShape[0];
        long inFeatures = wShape[1];

        float[] wData = new float[(int)(outFeatures * inFeatures)];
        float[] wDataT = new float[(int)(outFeatures * inFeatures)];

        for (int i = 0; i < wData.length; i++) {
            wData[i] = weights.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
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

        // Add bias (broadcast [out] to [N, out])
        // For now, manually broadcast bias
        long n = input.shape()[0];
        float[] biasData = new float[(int) outFeatures];
        for (int i = 0; i < outFeatures; i++) {
            biasData[i] = bias.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
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

    /**
     * Transpose tensor from NHWC [N, H, W, C] to NCHW [N, C, H, W].
     * This is needed before flattening because the FC weights were trained with NCHW layout.
     */
    private Tensor<Float> transposeNHWCtoNCHW(Tensor<Float> input) {
        long[] shape = input.shape();  // [N, H, W, C]
        long N = shape[0];
        long H = shape[1];
        long W = shape[2];
        long C = shape[3];

        float[] srcData = new float[(int)(N * H * W * C)];
        float[] dstData = new float[(int)(N * C * H * W)];

        // Read source data
        for (int i = 0; i < srcData.length; i++) {
            srcData[i] = input.data().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
        }

        // Transpose from NHWC to NCHW
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int c = 0; c < C; c++) {
                        // NHWC index: n*H*W*C + h*W*C + w*C + c
                        int srcIdx = (int)(n * H * W * C + h * W * C + w * C + c);
                        // NCHW index: n*C*H*W + c*H*W + h*W + w
                        int dstIdx = (int)(n * C * H * W + c * H * W + h * W + w);
                        dstData[dstIdx] = srcData[srcIdx];
                    }
                }
            }
        }

        return Tensor.ofFloats(new long[]{N, C, H, W}, dstData);
    }

    /**
     * Demo entry point.
     */
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("TOSA MNIST Model Demo");
        System.out.println("=".repeat(80));

        MNISTModel model = new MNISTModel();
        System.out.println("Model loaded successfully!");

        // Create a simple test pattern (diagonal line)
        float[] testImage = new float[IMAGE_SIZE * IMAGE_SIZE];
        for (int i = 0; i < IMAGE_SIZE; i++) {
            testImage[i * IMAGE_SIZE + i] = 255.0f; // diagonal
            if (i > 0) testImage[i * IMAGE_SIZE + i - 1] = 128.0f;
            if (i < IMAGE_SIZE - 1) testImage[i * IMAGE_SIZE + i + 1] = 128.0f;
        }

        System.out.println("\nRunning inference on test pattern...");
        float[] probs = model.classify(testImage);

        System.out.println("\nPrediction probabilities:");
        int maxIdx = 0;
        float maxProb = 0;
        for (int i = 0; i < probs.length; i++) {
            System.out.printf("  Digit %d: %.4f%n", i, probs[i]);
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        System.out.printf("\nPredicted digit: %d (confidence: %.2f%%)%n", maxIdx, maxProb * 100);
    }
}
