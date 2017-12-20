package jp.ac.tsukuba.cs.mdl.dnn4j.dataset;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NdIndex;
import jp.ac.tsukuba.cs.mdl.numj.core.NdSlice;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.io.FileInputStream;
import java.io.IOException;

public class Cifar10Dataset implements Dataset {

    private NdArray xTrain;
    private NdArray tTrain;
    private NdArray xTest;
    private NdArray tTest;

    private static final int CHANNEL = 3;
    private static final int SIZE = 32;
    private static final int TRAIN_SIZE = 50_000;
    private static final int TEST_SIZE = 10_000;

    private final static String trainFilePath = "src/main/resources/cifar-10-batches-bin/data_batch_%d.bin";
    private final static String testFilePath = "src/main/resources/cifar-10-batches-bin/test_batch.bin";

    public Cifar10Dataset() {

        xTrain = NumJ.zeros(TRAIN_SIZE, CHANNEL * SIZE * SIZE);
        tTrain = NumJ.zeros(TRAIN_SIZE);
        xTest = NumJ.zeros(TEST_SIZE, CHANNEL * SIZE * SIZE);
        tTest = NumJ.zeros(TEST_SIZE);

        int cnt = 0;

        for (int i = 1; i <= 5; i++) {
            try (FileInputStream stream = new FileInputStream(String.format(trainFilePath, i))) {
                byte[] line = new byte[3073];
                while (stream.read(line) > 0) {
                    tTrain.put(new int[]{cnt}, line[0]);
                    xTrain.put(
                            new NdIndex[]{NdSlice.point(cnt), NdSlice.all()},
                            NumJ.generator(j -> line[j + 1] / 255.0 * 2.0 - 1.0, 1, 3072)
                    );
                    cnt++;
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        cnt = 0;

        try (FileInputStream stream = new FileInputStream(testFilePath)) {
            byte[] line = new byte[3073];
            while (stream.read(line) > 0) {
                tTest.put(new int[]{cnt}, line[0]);
                xTest.put(
                        new NdIndex[]{NdSlice.point(cnt), NdSlice.all()},
                        NumJ.generator(j -> line[j + 1], 1, 3072)
                );
                cnt++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public NdArray readTrainFeatures() {
        return xTrain;
    }

    public NdArray readTestFeatures() {
        return xTest;
    }

    public NdArray readTrainLabels() {
        return tTrain;
    }

    public NdArray readTestLabels() {
        return tTest;
    }

    @Override
    public int getChannelSize() {
        return CHANNEL;
    }

    @Override
    public int getHeight() {
        return SIZE;
    }

    @Override
    public int getWidth() {
        return SIZE;
    }

    @Override
    public int getTrainSize() {
        return TRAIN_SIZE;
    }

    @Override
    public int getTestSize() {
        return TEST_SIZE;
    }
}
