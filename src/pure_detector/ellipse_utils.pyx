
cdef extern from "utils.hpp":
    cdef double distance(
        double c1x,
        double c1y,
        double a1x,
        double a1y,
        double phi1,
        double c2x,
        double c2y,
        double a2x,
        double a2y,
        double phi2
    );

def ellipse_distance(
    center1, axes1, angle1,
    center2, axes2, angle2
):
    c1x, c1y = center1
    a1x, a1y = axes1
    c2x, c2y = center2
    a2x, a2y = axes2

    return distance(
        c1x,
        c1y,
        a1x,
        a1y,
        angle1,
        c2x,
        c2y,
        a2x,
        a2y,
        angle2
    );