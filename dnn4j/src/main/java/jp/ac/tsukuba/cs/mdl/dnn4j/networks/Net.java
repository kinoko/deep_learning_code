package jp.ac.tsukuba.cs.mdl.dnn4j.networks;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

import java.util.Map;

public interface Net {


    Map<String, NdArray> getGradient();

    NdArray predict(NdArray x);

    double forward(NdArray x, NdArray t);

    NdArray backward(double dout);

    double accuracy(NdArray x, NdArray t);

    Map<String, NdArray> getParameters();

    void setParameters(Map<String, NdArray> params);
}
