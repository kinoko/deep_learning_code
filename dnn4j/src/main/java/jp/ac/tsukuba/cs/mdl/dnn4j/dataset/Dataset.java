package jp.ac.tsukuba.cs.mdl.dnn4j.dataset;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public interface Dataset {
    NdArray readTrainFeatures();
    NdArray readTestFeatures();
    NdArray readTrainLabels();
    NdArray readTestLabels();

    int getChannelSize();
    int getHeight();
    int getWidth();
    int getTrainSize();
    int getTestSize();
}
