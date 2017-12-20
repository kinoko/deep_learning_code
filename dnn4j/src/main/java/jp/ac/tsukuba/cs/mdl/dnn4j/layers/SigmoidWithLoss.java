package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;


public class SigmoidWithLoss implements LastLayer {

    private double loss;

    private NdArray y;

    private NdArray t;

    public static NdArray softmax(NdArray x) {

        if (x.dim() == 2) {
            x = x.transpose();
            x = x.sub(x.max(0).reshape(1, x.shape()[1]));
            NdArray y = x.elementwise(Math::exp).div(x.elementwise(Math::exp).sum(0).reshape(1, x.shape()[1]));
            return y.transpose();
        }
        x = x.sub(x.max());
        return x.elementwise(Math::exp).div(x.elementwise(Math::exp).sum());
    }

    public static double crossEntropyLoss(NdArray y, NdArray t) {
        int batchSize = y.shape()[0];


        if (y.dim() == 1) {
            t = t.reshape(1, t.size());
            y = y.reshape(1, y.size());

        }
        if (t.size() == y.size()) {
            t = t.argmax(1).reshape(1, t.shape()[0]);
        } else {
            t = t.reshape(1, t.size());
        }

        double res = 0;

        for (int i = 0; i < t.size(); i++) {
            int tt = (int) t.get(new int[]{0, i});
            double yy = y.get(new int[]{i, tt});

            res += Math.log(yy);
        }

        return -res / batchSize;
    }

    @Override
    public double forward(NdArray input, NdArray t) {
        this.t = t;
        this.y = softmax(input);
        this.loss = crossEntropyLoss(y, t);

        return this.loss;
    }

    @Override
    public NdArray backward(double dout) {
        int batchSize = this.y.shape()[0];
        if (this.t.size() == this.y.size()) {
            return this.y.sub(this.t).div(batchSize);
        } else {
            NdArray dx = this.y.copy();
            for (int i = 0; i < batchSize; i++) {
                int[] coordinate = new int[]{i, (int) this.t.get(new int[]{i})};
                dx.put(coordinate, dx.get(coordinate) - 1);
            }

            return dx.div(batchSize).mul(dout);
        }
    }
}
