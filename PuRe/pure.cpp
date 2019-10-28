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
    const int rows = edge_img.rows - 1;
    const int cols = edge_img.cols - 1;
    uchar *dest;
    int r, c;
    for (r = 1; r < rows; ++r)
    {
        above = edge_img.ptr(r - 1);
        current = edge_img.ptr(r);
        below = edge_img.ptr(r + 1);
        dest = out_img.ptr(r);
        for (c = 1; c < cols; ++c)
        {
            if (above[c] && current[c - 1] ||
                above[c] && current[c + 1] ||
                below[c] && current[c - 1] ||
                below[c] && current[c + 1])
            {
                dest[c] = 0;
            }
        }
    }
}
