package jp.ac.tsukuba.cs.mdl.dnn4j;

import java.util.List;

public interface Trainer {
    List<Double> getTrainLossList();

    List<Double> getTrainAccList();

    List<Double> getTestAccList();

    void train();
}
