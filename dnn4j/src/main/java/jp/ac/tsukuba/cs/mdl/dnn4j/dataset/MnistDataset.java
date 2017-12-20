package jp.ac.tsukuba.cs.mdl.dnn4j.dataset;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class MnistDataset implements Dataset{

    private static final String TRAIN_IMAGES_IDX_3_UBYTE_GZ = "src/main/resources/train-images-idx3-ubyte.gz";
    private static final String T_10_K_IMAGES_IDX_3_UBYTE_GZ = "src/main/resources/t10k-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS_IDX_1_UBYTE_GZ = "src/main/resources/train-labels-idx1-ubyte.gz";
    private static final String T_10_K_LABELS_IDX_1_UBYTE_GZ = "src/main/resources/t10k-labels-idx1-ubyte.gz";
    private static final int CHANNEL = 1;
    private static final int SIZE = 28;
    private static final int TRAIN_SIZE = 60000;
    private static final int TEST_SIZE = 10000;

    public NdArray readTrainFeatures() {
        return readFeatures(TRAIN_IMAGES_IDX_3_UBYTE_GZ);
    }

    public NdArray readTestFeatures() {
        return readFeatures(T_10_K_IMAGES_IDX_3_UBYTE_GZ);
    }


    public NdArray readTrainLabels() {
        return readLabels(TRAIN_LABELS_IDX_1_UBYTE_GZ);
    }

    public NdArray readTestLabels() {
        return readLabels(T_10_K_LABELS_IDX_1_UBYTE_GZ);
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

    private NdArray readFeatures(String path) {
        NdArray res;
        try (
                DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))
        ) {
            is.readInt();
            int numImages = is.readInt();
            int numDimensions = is.readInt() * is.readInt();

            double[] features = new double[numImages * numDimensions];
            int cnt = 0;
            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < numDimensions; j++) {
                    features[cnt++] = (double) is.readUnsignedByte() / 255.0;
                }
            }
            res = NumJ.create(features, numImages, numDimensions);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return res;
    }

    private NdArray readLabels(String path) {
        NdArray res;
        try (
                DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

        ) {
            is.readInt();
            int numLabels = is.readInt();

            double[] labels = new double[numLabels];
            for (int i = 0; i < numLabels; i++) {
                labels[i] = is.readUnsignedByte();
            }
            res = NumJ.create(labels, numLabels);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        return res;
    }
}
