package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public interface Layer {

    NdArray forward(NdArray input);

    NdArray backward(NdArray dout);

}
