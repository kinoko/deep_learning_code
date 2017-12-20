package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public interface LastLayer {

    double forward(NdArray input, NdArray t);

    NdArray backward(double dout);
}
