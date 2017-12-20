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
    private NdArray input;
    private NdArray argmax;

    public Pooling(int poolHeight, int poolWidth, int stride, int padding) {
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;
        this.padding = padding;
    }

    public Pooling(int poolHeight, int poolWeight) {
        this.poolHeight = poolHeight;
        this.poolWidth = poolWeight;
        stride = 1;
        padding = 0;
    }

    @Override
    public NdArray forward(NdArray input) {

        this.input = input;

        int[] inputShape = input.shape();

        int outHeight = Utils.computeOutputSize(inputShape[2], poolHeight, stride, padding);
        int outWeight = Utils.computeOutputSize(inputShape[3], poolWidth, stride, padding);

        NdArray col = Utils.im2col(this.input, poolHeight, poolWidth, stride, padding);
        col = col.reshape(col.size() / poolHeight / poolWidth, poolHeight * poolWidth);

        argmax = col.argmax(1);
        NdArray out = col.max(1);
        out = out.reshape(inputShape[0], outHeight, outWeight, inputShape[1]).transpose(0, 3, 1, 2);

        return out;
    }

    @Override
    public NdArray backward(NdArray dout) {
        dout = dout.transpose(0, 2, 3, 1);
        int poolSize = poolHeight * poolWidth;
        NdArray dmax = NumJ.zeros(dout.size(), poolSize);
        for (int i = 0; i < argmax.size(); i++) {
            dmax.put(new int[]{i, (int) argmax.get(i)}, dout.get(i));
        }

        dmax = dmax.reshape(Ints.concat(dout.shape(), new int[]{poolSize}));

        NdArray dcol = dmax.reshape(
                dmax.shape()[0] * dmax.shape()[1] * dmax.shape()[2],
                dmax.size() / dmax.shape()[0] / dmax.shape()[1] / dmax.shape()[2]
        );
        return Utils.col2im(dcol, input.shape(), poolHeight, poolWidth, stride, padding);
    }
}
