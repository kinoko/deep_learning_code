package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

import java.util.Map;

public interface Optimizer {
    Map<String, NdArray> update(Map<String, NdArray> param, Map<String, NdArray> grad);
}
