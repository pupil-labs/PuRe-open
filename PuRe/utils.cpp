#include <vector>
#include <cmath>
#include <numeric>

#include "utils.hpp"

using namespace std;

constexpr double PI = 3.14159265358979323846;

struct Point
{
    double x,y;
    Point() : x(0.0), y(0.0) {}
    Point(double x, double y) : x(x), y(y) {}
    Point(const Point& other) : x(other.x), y(other.y) {}
};

Point operator-(const Point& left, const Point& right)
{
    return Point(
        left.x - right.x,
        left.y - right.y
    );
}
Point operator*(const Point& left, const Point& right)
{
    return Point(
        left.x * right.x,
        left.y * right.y
    );
}

double norm(const Point& p)
{
    return sqrt(p.x * p.x + p.y * p.y);
}

struct Ellipse
{
    Point center;
    Point axes;
    double angle;
};

vector<Point> sample_native(const Ellipse& e, int n = 32)
{
    double x = e.center.x;
    double y = e.center.y;
    double a = e.axes.x;
    double b = e.axes.y;
    double cos_phi = cos(e.angle);
    double sin_phi = sin(e.angle);

    vector<Point> points;
    const double step = 2 * PI / n;
    for (double t = 0; t < 2 * PI; t += step)
    {
        double cos_t = cos(t);
        double sin_t = sin(t);
        points.emplace_back(
            x + (a * cos_phi * cos_t) - (b * sin_phi * sin_t),
            y + (a * sin_phi * cos_t) + (b * cos_phi * sin_t)
        );
    }
    return points;
}

vector<Point> sample(const Ellipse& e, const int n = 32)
{
    constexpr int oversampling_rate = 10;

    auto points = sample_native(e, n * oversampling_rate);
    points.insert(points.end(), points.front());

    vector<double> distances;
    for (int i = 0; i < points.size() -1; ++i)
    {
        auto diff = points[i] - points[i+1];
        distances.emplace_back(norm(diff));
    }

    double approx_circumference = accumulate(distances.begin(), distances.end(), 0.0);
    double step = approx_circumference / n;

    vector<Point> result;
    result.push_back(points[0]);
    double current_distance = 0;
    for (int i = 0; i < distances.size(); ++i)
    {
        current_distance += distances[i];
        if (current_distance >= step)
        {
            result.push_back(points[i + 1]);
            current_distance -= step;
        }
    }
    return result;
}

Point rotate(const Point& p, double angle)
{
    double c = cos(angle);
    double s = sin(angle);
    return Point(
        c * p.x - s * p.y,
        s * p.x + c * p.y
    );
}

double distance_to_point(const Ellipse& e, const Point& _p)
{
    Point p(_p);
    p = p - e.center;
    p = rotate(p, -e.angle);

    if (p.x < 0) p.x = -p.x;
    if (p.y < 0) p.y = -p.y;

    Point axes = e.axes;
    if (axes.x < axes.y)
    {
        swap(axes.x, axes.y);
        swap(p.x, p.y);
    }

    Point x;
    if (p.y > 0)
    {
        if (p.x > 0)
        {
            const Point esqr(axes * axes);
            const Point ep(axes * p);
            double t0 = -esqr.y + ep.y;
            double t1 = -esqr.y + norm(ep);
            double t = t0;

            constexpr int imax = 2 * std::numeric_limits<double>::max_exponent;
            for (int i = 0; i < imax; ++i)
            {
                t = 0.5 * (t0 + t1);
                if (t == t0 || t == t1) break;

                const double f1 = ep.x / (t + esqr.x);
                const double f2 = ep.y / (t + esqr.y);
                const double f = f1 * f1 + f2 * f2 - 1;
                if (f > 0) t0 = t;
                else if (f < 0) t1 = t;
                else break;
            }
            x = Point(
                esqr.x * p.x / (t + esqr.x),
                esqr.y * p.y / (t + esqr.y)
            );
        }
        else x = Point(0, axes.y);
    }
    else
    {
        const double e0 = axes.x;
        const double e1 = axes.y;
        const double y0 = p.x;
        const double y1 = p.y;
        if (y0 < (e0 * e0 - e1 * e1) / e0)
        {
            x.x = e0 * e0 * y0 / (e0 * e0 - e1 * e1);
            x.y = e1 * sqrt(1 - (x.x / e0) * (x.x / e0));
        }
        else x = Point(e0, 0);
    }
    
    return norm(x - p);
}

double distance_to_ellipse(const Ellipse& e1, const Ellipse& e2)
{
    int count = 0;
    double sum = 0;
    for (const Point& p : sample(e1))
    {
        ++count;
        sum += distance_to_point(e2, p);
    }
    for (const Point& p : sample(e2))
    {
        ++count;
        sum += distance_to_point(e1, p);
    }
    return sum / count;
}

double distance(
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
){
    Ellipse e1;
    e1.center.x = c1x;
    e1.center.y = c1y;
    e1.axes.x = a1x;
    e1.axes.y = a1y;
    e1.angle = phi1;

    Ellipse e2;
    e2.center.x = c2x;
    e2.center.y = c2y;
    e2.axes.x = a2x;
    e2.axes.y = a2y;
    e2.angle = phi2;

    return distance_to_ellipse(e1, e2);
}