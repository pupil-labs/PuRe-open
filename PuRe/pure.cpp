#include <iostream>

#include "pure.hpp"

void pure::thin_edges(const Mat& edge_img, Mat& out_img) {
    // Thinning

    // This is an efficient implementation of the orginal thinning algorithm described
    // in the ExCuSe paper (Fuhl et al., 2015). While the PuRe paper links to the ElSe
    // paper (Fuhl et al. 2016), the description there is ambiguous so I looked up the
    // previous paper.

    // The idea of this implementation is to move a 3x3 filter across the image, which
    // removes the central pixel from the edge images, if it would have been matched by
    // any of the 4 described thinning masks. This way we can apply all 4 masks
    // efficiently in one operation.
    
    // If the pixels with E have an edge, pixel X is removed:
    // |_|E|_| |_|E|_| |_|_|_| |_|_|_|
    // |E|X|_| |_|X|E| |E|X|_| |_|X|E|
    // |_|_|_| |_|_|_| |_|E|_| |_|E|_|

    // TODO: Handle borders of the image.
    const uchar *above, *current, *below;
    const int rows = edge_img.rows - 2;
    const int cols = edge_img.cols - 2;
    uchar *dest;
    int r, c;
    for (r = 0; r < rows; ++r)
    {
        above = edge_img.ptr(r);
        current = edge_img.ptr(r + 1);
        below = edge_img.ptr(r + 2);
        dest = out_img.ptr(r + 1);
        for (c = 0; c < cols; ++c)
        {
            if (above[c + 1] && current[c] ||
                above[c + 1] && current[c + 2] ||
                below[c + 1] && current[c] ||
                below[c + 1] && current[c + 2])
            {
                dest[c + 1] = 0;
            }
        }
    }
}

void pure::straighten_edges(const Mat& edge_img, Mat& out_img) {
    // Straightening
    // As described in in the ElSe paper (Fuhl et al. 2016)

    // NOTE: The order of operation does make a difference here as some of those filters
    // overlap. Since there was no specific information about the order of application,
    // I implemented then basically in the order of presentation in the paper.

    // TODO: This could be optimized even more with something like karnaugh maps.
    // TODO: Handle border of the image.
    const uchar *in0, *in1, *in2, *in3;
    uchar *out0, *out1, *out2, *out3;
    const int rows = edge_img.rows - 3;
    const int cols = edge_img.cols - 3;
    int r, c;
    for (r = 0; r < rows; ++r)
    {
        in0 = edge_img.ptr(r);
        in1 = edge_img.ptr(r + 1);
        in2 = edge_img.ptr(r + 2);
        in3 = edge_img.ptr(r + 3);
        out0 = out_img.ptr(r);
        out1 = out_img.ptr(r + 1);
        out2 = out_img.ptr(r + 2);
        out3 = out_img.ptr(r + 3);
        for (c = 0; c < cols; ++c)
        {
            if (in1[c] && in0[c + 1] && in1[c + 2])
            {
                //  X
                // XXX
                out0[c + 1] = 0;
                out1[c + 1] = 255;
            }
            if (in1[c] && in0[c + 1] && in0[c + 2] && in1[c + 3])
            {
                //  XX
                // XXXX
                out0[c + 1] = 0;
                out0[c + 2] = 0;
                out1[c + 1] = 255;
                out1[c + 2] = 255;
            }
            if (in0[c + 1] && in1[c] && in2[c + 1])
            {
                //  X
                // XX
                //  X
                out1[c] = 0;
                out1[c + 1] = 255;
            }
            if (in0[c + 1] && in1[c] && in2[c] && in3[c + 1])
            {
                //  X
                // XX
                // XX
                //  X
                out1[c] = 0;
                out2[c] = 0;
                out1[c + 1] = 255;
                out2[c + 1] = 255;
            }
            if (in0[c] && in1[c + 1] && in2[c])
            {
                // X
                // XX
                // X
                out1[c + 1] = 0;
                out1[c] = 255;
            }
            if (in0[c] && in1[c + 1] && in2[c + 1] && in3[c])
            {
                // X
                // XX
                // XX
                // X
                out1[c + 1] = 0;
                out2[c + 1] = 0;
                out1[c] = 255;
                out2[c] = 255;
            }
            if (in0[c] && in1[c + 1] && in0[c + 2])
            {
                // XXX
                //  X
                out1[c + 1] = 0;
                out0[c + 1] = 255;
            }
            if (in0[c] && in1[c + 1] && in1[c + 2] && in0[c + 3])
            {
                // XXXX
                //  XX
                out1[c + 1] = 0;
                out1[c + 2] = 0;
                out0[c + 1] = 255;
                out0[c + 2] = 255;
            }
        }
    }
}

void pure::break_orthogonals(const Mat& edge_img, Mat& out_img) {
    // Removal of orthogonal connections
    // As described in in the ElSe paper (Fuhl et al. 2016)

    const uchar *in0, *in1, *in2, *in3, *in4, *in5;
    uchar *out0, *out1, *out2, *out3;
    const int rows = edge_img.rows - 5;
    const int cols = edge_img.cols - 5;
    int r, c;
    for (r = 0; r < rows; ++r)
    {
        in0 = edge_img.ptr(r);
        in1 = edge_img.ptr(r + 1);
        in2 = edge_img.ptr(r + 2);
        in3 = edge_img.ptr(r + 3);
        in4 = edge_img.ptr(r + 4);
        in5 = edge_img.ptr(r + 5);
        out0 = out_img.ptr(r);
        out1 = out_img.ptr(r + 1);
        out2 = out_img.ptr(r + 2);
        out3 = out_img.ptr(r + 3);
        for (c = 0; c < cols; ++c)
        {
            // Every pattern affects only 1 pixel. Here the patterns are grouped by the
            // pixel they affect. First pixel read is the one potentially turned off.
            // Following are all patterns that can turn this pixel off. This allows for
            // using pattern overlapping, e.g. between f2 and g2 (see below). Also it
            // results in every pixel being written only once.
            if (in0[c + 1] && (
                in0[c] && in1[c + 2] && in2[c +2] || // d1
                in0[c + 2] && in1[c] && in2[c] // d3
            )) out0[c + 1] = 0;

            if (in0[c + 2] && in1[c + 1] && in1[c + 3] && in2[c] && in2[c + 4]) out0[c + 2] = 0; // f1

            if (in0[c + 3] && in0[c + 2] && in1[c + 1] && in1[c + 4] && in2[c] && in2[c + 5]) out0[c + 3] = 0; // g1

            if (in1[c + 1] && in0[c + 2] && in0[c + 3] && in0[c + 4] && in2[c] && in3[c] && in4[c]) out1[c + 1] = 0; // e3

            if (in1[c + 3] && in0[c] && in0[c + 1] && in0[c + 2] && in2[c + 4] && in3[c + 4] && in4[c + 4]) out1[c + 3] = 0; //e1

            if (in2[c] && in1[c + 1] && in0[c + 2] && (
                in3[c + 1] && in4[c + 2] || // f2
                in3[c] && in4[c + 1] && in5[c + 2] // g2
            )) out2[c] = 0;

            if (in2[c + 1] && (
                in0[c] && in1[c] && in2[c + 2] || // d2
                in0[c + 2] && in1[c + 2] && in2[c] // d4
            )) out2[c + 1] = 0;

            if (in2[c + 2] && in0[c] && in1[c + 1] && (
                in3[c + 1] && in4[c] || // f3
                in1[c + 3] && in0[c + 4] || // f4
                in2[c + 3] && in1[c + 4] && in0[c + 5] // g4
            )) out2[c + 2] = 0;

            if (in3[c + 1] && in0[c] && in1[c] && in2[c] && in4[c + 2] && in4[c + 3] && in4[c + 4]) out3[c + 1] = 0; //e2

            if (in3[c + 2] && in0[c] && in1[c + 1] && in2[c + 2] && in4[c + 1] && in5[c]) out3[c + 2] = 0; //g3

            if (in3[c + 3] && in0[c + 4] && in1[c + 4] && in2[c + 4] && in4[c] && in4[c + 1] && in4[c + 2]) out3[c + 3] = 0; // e4
        }
    }
}
