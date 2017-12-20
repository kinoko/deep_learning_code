package jp.ac.tsukuba.cs.mdl.dnn4j;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NdIndex;
import jp.ac.tsukuba.cs.mdl.numj.core.NdSlice;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

public class Utils {

    public static int computeOutputSize(int inputSize, int filterSize, int stride, int padding) {
        return 2;
    }

    public static NdArray im2col(NdArray inputData, int kernelHeight, int kernelWidth, int stride, int padding) {
        int[] inputShape = inputData.shape();
        int outHeight = (inputShape[2] + 2 * padding - kernelHeight) / stride + 1;
        int outWidth = (inputShape[3] + 2 * padding - kernelWidth) / stride + 1;
        int[] paddingShape = Arrays.copyOf(inputShape, inputShape.length);
        paddingShape[2] += 2 * padding;
        paddingShape[3] += 2 * padding;
        NdArray img = NumJ.zeros(paddingShape);
        img.put(
                new NdIndex[]{
                        NdSlice.all(),
                        NdSlice.all(),
                        NdSlice.interval(padding, inputShape[2] + padding),
                        NdSlice.interval(padding, inputShape[3] + padding)
                },
                inputData
        );
        img = img.reshape(inputShape[0], inputShape[1], 1, 1, inputShape[2] + 2 * padding, inputShape[2] + 2 * padding);
        NdArray col = NumJ.zeros(inputShape[0], inputShape[1], kernelHeight, kernelWidth, outHeight, outWidth);
        for (int y = 0; y < kernelHeight; y++) {
            int yMax = y + stride * outHeight;
            for (int x = 0; x < kernelWidth; x++) {
                int xMax = x + stride * outWidth;
                if (stride == 1) {
                    col.put(
                            new NdIndex[]{
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(y),
                                    NdSlice.point(x),
                                    NdSlice.all(),
                                    NdSlice.all()
                            },
                            img.slice(
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.interval(y, yMax),
                                    NdSlice.interval(x, xMax)
                            )
                    );
                } else {
                    Set<Integer> xIndex = new HashSet<>();
                    Set<Integer> yIndex = new HashSet<>();
                    for (int i = x; i < xMax; i += stride) {
                        xIndex.add(i);
                    }
                    for (int i = y; i < yMax; i += stride) {
                        yIndex.add(i);
                    }

                    col.put(
                            new NdIndex[]{
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(y),
                                    NdSlice.point(x),
                                    NdSlice.all(),
                                    NdSlice.all()
                            },
                            img.slice(
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.set(yIndex),
                                    NdSlice.set(xIndex)
                            )
                    );
                }

            }
        }
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(inputShape[0] * outHeight * outWidth, col.size() / (inputShape[0] * outHeight * outWidth));
    }

    public static NdArray col2im(NdArray col, int[] inputShape, int kernelHeight, int kernelWidth, int stride, int padding) {
        // N, C, H, W = input_shape
        int outHeight = (inputShape[2] + 2 * padding - kernelHeight) / stride + 1;
        int outWidth = (inputShape[3] + 2 * padding - kernelWidth) / stride + 1;
        col = col.reshape(inputShape[0], outHeight, outWidth, inputShape[1], kernelHeight, kernelWidth).transpose(0, 3, 4, 5, 1, 2);

        NdArray img = NumJ.zeros(inputShape[0], inputShape[1], 1, 1, inputShape[2] + 2 * padding + stride - 1, inputShape[3] + 2 * padding + stride - 1);
        for (int y = 0; y < kernelHeight; y++) {
            int yMax = y + stride * outHeight;
            for (int x = 0; x < kernelWidth; x++) {
                int xMax = x + stride * outWidth;
                if (stride == 1) {
                    img.put(new NdIndex[]{
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(0),
                                    NdSlice.point(0),
                                    NdSlice.interval(y, yMax),
                                    NdSlice.interval(x, xMax)
                            },
                            img.slice(
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(0),
                                    NdSlice.point(0),
                                    NdSlice.interval(y, yMax),
                                    NdSlice.interval(x, xMax)
                            ).add(
                                    col.slice(
                                            NdSlice.all(),
                                            NdSlice.all(),
                                            NdSlice.point(y),
                                            NdSlice.point(x),
                                            NdSlice.all(),
                                            NdSlice.all()
                                    )
                            )
                    );
                } else {
                    Set<Integer> xIndex = new HashSet<>();
                    Set<Integer> yIndex = new HashSet<>();
                    for (int i = x; i < xMax; i += stride) {
                        xIndex.add(i);
                    }
                    for (int i = y; i < yMax; i += stride) {
                        yIndex.add(i);
                    }
                    img.put(new NdIndex[]{
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(0),
                                    NdSlice.point(0),
                                    NdSlice.set(yIndex),
                                    NdSlice.set(xIndex)
                            },
                            img.slice(
                                    NdSlice.all(),
                                    NdSlice.all(),
                                    NdSlice.point(0),
                                    NdSlice.point(0),
                                    NdSlice.set(yIndex),
                                    NdSlice.set(xIndex)
                            ).add(
                                    col.slice(
                                            NdSlice.all(),
                                            NdSlice.all(),
                                            NdSlice.point(y),
                                            NdSlice.point(x),
                                            NdSlice.all(),
                                            NdSlice.all()
                                    )
                            )
                    );
                }
            }
        }

        return img.reshape(
                inputShape[0],
                inputShape[1],
                inputShape[2] + 2 * padding + stride - 1,
                inputShape[3] + 2 * padding + stride - 1
        ).slice(
                NdSlice.all(),
                NdSlice.all(),
                NdSlice.interval(padding, inputShape[2] + padding),
                NdSlice.interval(padding, inputShape[3] + padding)
        );
    }

}
