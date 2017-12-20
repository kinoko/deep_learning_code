package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArrayImpl;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Test;
import org.junit.*;

import static org.junit.Assert.*;

public class CrossEntropyTest {
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
        double truth = 0.0;
        SigmoidWithLoss softmax = new SigmoidWithLoss();
        double sm = softmax.crossEntropyLoss(y, y);
        assertEquals(sm, truth, 1e-5);
    }

}

