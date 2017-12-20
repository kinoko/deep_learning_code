package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import com.google.common.primitives.Ints;
import jp.ac.tsukuba.cs.mdl.dnn4j.Utils;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

public class Pooling implements Layer {

    private int poolHeight;
    private int poolWidth;
    private int stride;
    private int padding;
    private int[] inputShape;
    private NdArray argmax;

    public Pooling(int poolHeight, int poolWidth, int stride, int padding) {
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;
        this.padding = padding;
    }

    public Pooling(int poolHeight, int poolWeight) {
        this(poolHeight, poolWeight, 1, 0);
    }

    @Override
    public NdArray forward(NdArray input) {

        this.inputShape = input.shape();

        int[] inputShape = input.shape();

        int outHeight = Utils.computeOutputSize(inputShape[2], poolHeight, stride, padding);
        int outWeight = Utils.computeOutputSize(inputShape[3], poolWidth, stride, padding);

        NdArray col = Utils.im2col(input, poolHeight, poolWidth, stride, padding);

        // ここから実装 colの最大値をoutに出力する．最大値をとったフィルターの場所をargmaxに出力する．
        NdArray out = NumJ.zeros(0);
        // ここまで実装

        out = out.reshape(inputShape[0], outHeight, outWeight, inputShape[1]).transpose(0, 3, 1, 2);

        return out;
    }

    @Override
    public NdArray backward(NdArray dout) {
        dout = dout.transpose(0, 2, 3, 1);

        int poolSize = poolHeight * poolWidth;

        NdArray dmax = NumJ.zeros(inputShape[0] * inputShape[1], poolSize);

        // ここから実装 最大値をとった部分(argmax)にdoutを代入する
        // ここまで実装

        dmax = dmax.reshape(Ints.concat(dout.shape(), new int[]{poolSize}));

        NdArray dcol = dmax.reshape(
                dmax.shape()[0] * dmax.shape()[1] * dmax.shape()[2],
                dmax.size() / dmax.shape()[0] / dmax.shape()[1] / dmax.shape()[2]
        );
        return Utils.col2im(dcol, inputShape, poolHeight, poolWidth, stride, padding);
    }
}
