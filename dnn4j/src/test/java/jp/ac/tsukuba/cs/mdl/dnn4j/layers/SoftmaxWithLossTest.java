package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Test;
import org.junit.*;

import static org.junit.Assert.*;


public class SoftmaxWithLossTest {

    SigmoidWithLoss layer;
    @Test
    public void case1() {
        NdArray y = NumJ.create(new double[]{0.93682228,  0.2988156 ,  0.3324376 ,
                                             0.67739091,  0.62190974,  0.20012173,
                                             0.01683243,  0.16738441,  0.09766853,
                                             0.44602553,  0.11407087,  0.37601503}, 4, 3);
        NdArray t = NumJ.create(new double[]{
                                            0.79238771,  0.45062911,  0.57082742,
                                            0.62231419,  0.09109418,  0.76716379,
                                            0.70575938,  0.49433838,  0.08729434,
                                            0.90075878,  0.89289039,  0.207889  }, 4, 3);

        double truth = 1.6414795214742628;
        SigmoidWithLoss softmax = new SigmoidWithLoss();
        double sm = softmax.crossEntropyLoss(y, t);
        assertEquals(sm, truth, 1e-5);
    }

    @Test
    public void case2() {
        NdArray y = NumJ.identity(4);
        double truth = 0.0;;
        SigmoidWithLoss softmax = new SigmoidWithLoss();
        double sm = softmax.crossEntropyLoss(y, y);
        assertEquals(sm, truth, 1e-5);
    }

    @Test
    public void softmax() throws Exception {
        NdArray x = NumJ.create(new double[]{-0.48180806, -1.9492681 , -0.93242164,
             -0.58552166, -0.34642482, -0.61280527,
             0.71635241,  0.48873738,  0.54235823,
             -1.14715309, -0.12911031, -0.57828824}, 4, 3);

        NdArray truth = NumJ.create(new double[]{0.53540435,  0.12341618,  0.34117948,
                                                 0.3083387 ,  0.39162146,  0.30003984,
                                                0.37925727,  0.30205217,  0.31869056,
                                                 0.18070003,  0.50013655,  0.31916341}, 4, 3);

        NdArray result = layer.softmax(x);
        assertEquals(result.sum(), truth.sum(), 1e-5);
    }


    @Test
    public void crossEntropyLoss() throws Exception {
        NdArray y = NumJ.create(new double[]{0.93682228,  0.2988156 ,  0.3324376 ,
                                             0.67739091,  0.62190974,  0.20012173,
                                             0.01683243,  0.16738441,  0.09766853,
                                             0.44602553,  0.11407087,  0.37601503}, 4, 3);
        NdArray t = NumJ.create(new double[]{
                                            0.79238771,  0.45062911,  0.57082742,
                                            0.62231419,  0.09109418,  0.76716379,
                                            0.70575938,  0.49433838,  0.08729434,
                                            0.90075878,  0.89289039,  0.207889  }, 4, 3);

        double truth = 1.6414795214742628;
        SigmoidWithLoss softmax = new SigmoidWithLoss();
        double sm = softmax.crossEntropyLoss(y, t);
        assertEquals(sm, truth, 1e-5);

        y = NumJ.identity(4);
        truth = 0.0;
        sm = softmax.crossEntropyLoss(y, y);
        assertEquals(sm, truth, 1e-5);
    }

    @Before
    public void init() {
        layer = new SigmoidWithLoss();
    }

    @Test
    public void forward() throws Exception {
        NdArray x = NumJ.create(new double[]{-0.48180806, -1.9492681 , -0.93242164,
            -0.58552166, -0.34642482, -0.61280527,
            0.71635241,  0.48873738,  0.54235823,
            -1.14715309, -0.12911031, -0.57828824}, 4, 3);
        NdArray y = NumJ.create(new double[]{-0.48180806, -1.9492681 , -0.93242164,
            -0.58552166, -0.34642482, -0.61280527,
            0.71635241,  0.48873738,  0.54235823,
            -1.14715309, -0.12911031, -0.57828824}, 4, 3);
        double truth = 0.80615180372596229;
        double result = layer.forward(x, y);
        assertEquals(truth, result, 1e-5);
    }

    @Test
    public void backward() throws Exception {
        NdArray x = NumJ.create(new double[]{-0.48180806, -1.9492681 , -0.93242164,
            -0.58552166, -0.34642482, -0.61280527,
            0.71635241,  0.48873738,  0.54235823,
            -1.14715309, -0.12911031, -0.57828824}, 4,3);
        NdArray y = NumJ.create(new double[]{-0.48180806, -1.9492681 , -0.93242164,
            -0.58552166, -0.34642482, -0.61280527,
            0.71635241,  0.48873738,  0.54235823,
            -1.14715309, -0.12911031, -0.57828824}, 4, 3);
        layer.forward(x, y);

        NdArray truth = NumJ.create(new double[]{0.2543031,  0.51817107,  0.31840028,
                0.22346509,  0.18451157,  0.22821128,
               -0.08427379, -0.0466713 , -0.05591692,
                0.33196328,  0.15731172,  0.22436291}, 4, 3);
        
        NdArray result = layer.backward(0.8);
        assertEquals(result.sum(), truth.sum(), 1e-5);
    }

}
