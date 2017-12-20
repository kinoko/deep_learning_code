package jp.ac.tsukuba.cs.mdl.dnn4j;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;
import jp.ac.tsukuba.cs.mdl.dnn4j.networks.Net;
import jp.ac.tsukuba.cs.mdl.dnn4j.optimizers.*;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NdSlice;

import java.util.*;

public class TrainerImpl implements Trainer {

    private Net network;
    private NdArray xTrain;
    private NdArray tTrain;
    private NdArray xTest;
    private NdArray tTest;
    private int epochs;
    private int miniBatchSize;
    private Optimizer optimizer;
    private int evaluateBatchSize;
    private boolean verbose;
    private int trainSize;
    private int iterPerEpoch;
    private int maxIter;
    private int currentIter;
    private int currentEpoch;

    private List<Double> trainLossList = Lists.newArrayList();
    private List<Double> trainAccList = Lists.newArrayList();
    private List<Double> testAccList = Lists.newArrayList();

    public TrainerImpl(Net network, NdArray xTrain, NdArray tTrain, NdArray xTest, NdArray tTest, int epochs,
                       int miniBatchSize, OptimizerType optimizer, Map<String, Double> optimizerParams,
                       int evaluateBatchSize, boolean verbose) {
        this.network = network;
        this.xTrain = xTrain;
        this.tTrain = tTrain;
        this.xTest = xTest;
        this.tTest = tTest;
        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.evaluateBatchSize = evaluateBatchSize;
        this.verbose = verbose;

        trainSize = xTrain.shape()[0];
        iterPerEpoch = Math.max(1, trainSize / miniBatchSize);
        maxIter = this.epochs * iterPerEpoch;
        currentIter = 0;
        currentEpoch = 0;

        switch (optimizer) {
            case SGD:
                this.optimizer = new SGD(optimizerParams);
                break;
            case ADAM:
                this.optimizer = new Adam(optimizerParams);
                break;
            case ADA_GRAD:
                this.optimizer = new AdaGrad(optimizerParams);
                break;
            case MOMENTUM:
                this.optimizer = new MomentumSGD(optimizerParams);
                break;
            case NESTEROV:
                this.optimizer = new Nesterov(optimizerParams);
                break;
            case RMS_PROP:
                this.optimizer = new RMSProp(optimizerParams);
                break;
            default:
                this.optimizer = new SGD(optimizerParams);
        }
    }

    private void trainStep() {
        Random random = new Random();
        Set<Integer> indices = Sets.newHashSet();
        while (indices.size() < miniBatchSize) {
            int i = random.nextInt(trainSize);
            indices.add(i);
        }
        int[] batchMask = Ints.toArray(indices);

        int[] xShape = xTrain.shape();

        NdArray xBatch = xTrain
                .reshape(xTrain.shape()[0], xTrain.size() / xTrain.shape()[0])
                .slice(NdSlice.set(batchMask), NdSlice.all())
                .reshape(Ints.concat(new int[]{batchMask.length}, Arrays.copyOfRange(xShape, 1, xShape.length)));
        NdArray tBatch = tTrain.slice(NdSlice.set(batchMask));

        double loss = network.forward(xBatch, tBatch);
        trainLossList.add(loss);

        network.backward(1);

        Map<String, NdArray> grads = network.getGradient();
        Map<String, NdArray> params = optimizer.update(network.getParameters(), grads);
        network.setParameters(params);


        if (verbose) {
            System.out.println("=== train loss:" + loss);
        }

        currentIter++;
        if (currentIter % iterPerEpoch == 0) {
            currentEpoch++;

            double trainAcc = getAcc(xTrain, tTrain);
            double testAcc = getAcc(xTest, tTest);


            trainAccList.add(trainAcc);
            testAccList.add(testAcc);

            System.out.println("=== epoch:" + currentEpoch + ", train acc:" + trainAcc + ", test acc:" + testAcc + " ===");
        }
    }

    private double getAcc(NdArray input, NdArray target) {
        int[] xShape;
        xShape = input.shape();
        double acc = 0;
        for (int i = 0; i < xShape[0]; i += evaluateBatchSize) {
            int sampleSize = Math.min(i + evaluateBatchSize, xShape[0]) - i;
            NdArray xSample = input
                    .reshape(input.shape()[0], input.size() / xShape[0])
                    .slice(NdSlice.interval(i, Math.min(i + evaluateBatchSize, xShape[0])), NdSlice.all())
                    .reshape(
                            sampleSize, xShape[1], xShape[2], xShape[3]
                    );
            NdArray tSample = target
                    .reshape(target.size())
                    .slice(NdSlice.interval(i, Math.min(i + evaluateBatchSize, xShape[0])));

            acc += network.accuracy(xSample, tSample) * sampleSize;
        }

        acc /= xShape[0];
        return acc;
    }

    @Override
    public List<Double> getTrainLossList() {
        return trainLossList;
    }

    @Override
    public List<Double> getTrainAccList() {
        return trainAccList;
    }

    @Override
    public List<Double> getTestAccList() {
        return testAccList;
    }

    @Override
    public void train() {
        for (int i = 0; i < maxIter; i++) {
            trainStep();
        }

        double testAcc = getAcc(xTest, tTest);
        System.out.println("=============== Final Test Accuracy ===============");
        System.out.println("test acc:" + testAcc);
    }
}
