package jp.ac.tsukuba.cs.mdl.dnn4j;

import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Assert;
import org.junit.Test;

public class UtilsTest {


//    @Test
    public void computeOutputSize() throws Exception{
        Assert.assertEquals(28, Utils.computeOutputSize(28, 3, 1, 1));
        Assert.assertEquals(14, Utils.computeOutputSize(28, 2, 2, 0));
        Assert.assertEquals(13, Utils.computeOutputSize(28, 5, 2, 1));
    }

    @Test
    public void im2col() throws Exception{
        Assert.assertEquals(NumJ.create(new double[]{
                  0.,   1.,   2.,   4.,   5.,   6.,   8.,   9.,  10.,
        1.,   2.,   3.,   5.,   6.,   7.,   9.,  10.,  11.,
         4.,   5.,   6.,   8.,   9.,  10.,  12.,  13.,  14.,
         5.,   6.,   7.,   9.,  10.,  11.,  13.,  14.,  15.,
        16.,  17.,  18.,  20.,  21.,  22.,  24.,  25.,  26.,
        17.,  18.,  19.,  21.,  22.,  23.,  25.,  26.,  27.,
        20.,  21.,  22.,  24.,  25.,  26.,  28.,  29.,  30.,
        21.,  22.,  23.,  25.,  26.,  27.,  29.,  30.,  31.,
        32.,  33.,  34.,  36.,  37.,  38.,  40.,  41.,  42.,
        33.,  34.,  35.,  37.,  38.,  39.,  41.,  42.,  43.,
        36.,  37.,  38.,  40.,  41.,  42.,  44.,  45.,  46.,
        37.,  38.,  39.,  41.,  42.,  43.,  45.,  46.,  47.,
        48.,  49.,  50.,  52.,  53.,  54.,  56.,  57.,  58.,
        49.,  50.,  51.,  53.,  54.,  55.,  57.,  58.,  59.,
        52.,  53.,  54.,  56.,  57.,  58.,  60.,  61.,  62.,
        53.,  54.,  55.,  57.,  58.,  59.,  61.,  62.,  63.
        }, 16, 9), Utils.im2col(NumJ.arange(4, 1, 4, 4) ,3,3,1,0));
    }

    @Test
    public void col2im() throws Exception{
        Assert.assertEquals(
                NumJ.create(new double[]{
                           0.,   10.,   12.,   11.,
           21.,   62.,   66.,   43.,
           27.,   74.,   78.,   49.,
           24.,   58.,   60.,   35.,


         36.,   82.,   84.,   47.,
           93.,  206.,  210.,  115.,
           99.,  218.,  222.,  121.,
           60.,  130.,  132.,   71.,


         72.,  154.,  156.,   83.,
          165.,  350.,  354.,  187.,
          171.,  362.,  366.,  193.,
           96.,  202.,  204.,  107.,


        108.,  226.,  228.,  119.,
          237.,  494.,  498.,  259.,
          243.,  506.,  510.,  265.,
          132.,  274.,  276.,  143.
                }, 4,1,4,4), Utils.col2im(NumJ.arange(16, 9), new int[]{4,1,4,4}, 3,3,1,0)
        );
    }
}
