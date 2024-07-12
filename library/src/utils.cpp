#include "tplcpp/utils.hpp"
#include "tplcpp/poly_interp.cuh"

#include <cmath>
#include <thread>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

#define sq(x) ((x)*(x))

bool pointInPolygon(
        double x,
        double y,
        array_like<double>& argPolygon) {

    auto polygon = argPolygon.unchecked<2>();

    double p1x = polygon(0, 0);
    double p1y = polygon(0, 1);
    double p2x = 0.0;
    double p2y = 0.0;

    double xints = 0.0;

    bool inside = false;

    size_t n = polygon.shape(0);

    for (size_t i = 0; i < n + 1; ++i) {

        p2x = polygon(i % n, 0);
        p2y = polygon(i % n, 1);

        if (std::min(p1y, p2y) <= y 
            && y <= std::max(p1y, p2y)
            && x <= std::max(p1x, p2x)) {

            if (p1y != p2y) {
                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
            }
            if (p1x == p2x || x <= xints) { 
                inside = !inside;
            }
        }

        p1x = p2x;
        p1y = p2y;
    }

    return inside;
}

bool intersectPolygons(array_like<double>& polyArray0,
                       array_like<double>& polyArray1) {

    auto poly0 = polyArray0.unchecked<2>();
    auto poly1 = polyArray1.unchecked<2>();
    size_t l0 = poly0.shape(0);
    size_t l1 = poly1.shape(0);

    if (l0 == 0 || l1 == 0) {
        // empty polygon does not intersect
        return false;
    }

    for (size_t i = 0; i <= l0; ++i) {
        vec<2> p0(poly0(i, 0), poly0(i, 1));
        vec<2> p1(poly0((i+1) % l0, 0), poly0((i+1) % l0, 1));
        vec<2> edge = p1 - p0;
        edge /= edge.norm();
        vec<2> ortho(-edge[1], edge[0]);

        double min0 = INFINITY;
        double max0 = -INFINITY;
        for (int j = 0; j < l0; j++) {
            double q = ortho.dot(vec<2>(poly0(j, 0), poly0(j, 1)));
            min0 = std::min(q, min0);
            max0 = std::max(q, max0);
        }

        double min1 = INFINITY;
        double max1 = -INFINITY;
        for (int j = 0; j < l1; j++) {
            double q = ortho.dot(vec<2>(poly1(j, 0), poly1(j, 1)));
            min1 = std::min(q, min1);
            max1 = std::max(q, max1);
        }

        if ((max0 < min1 && max0 < max1) || (min0 > min1 && min0 > max1)) {
            // found separating axis, polygons do not intersect
            return false;
        }
    }

    for (size_t i = 0; i <= l1; ++i) {
        vec<2> p0(poly1(i, 0), poly1(i, 1));
        vec<2> p1(poly1((i+1) % l1, 0), poly1((i+1) % l1, 1));
        vec<2> edge = p1 - p0;
        vec<2> ortho(-edge[1], edge[0]);

        double min0 = INFINITY;
        double max0 = -INFINITY;
        for (int j = 0; j < l0; j++) {
            double q = ortho.dot(vec<2>(poly0(j, 0), poly0(j, 1)));
            min0 = std::min(q, min0);
            max0 = std::max(q, max0);
        }

        double min1 = INFINITY;
        double max1 = -INFINITY;
        for (int j = 0; j < l1; j++) {
            double q = ortho.dot(vec<2>(poly1(j, 0), poly1(j, 1)));
            min1 = std::min(q, min1);
            max1 = std::max(q, max1);
        }

        if ((max0 < min1 && max0 < max1) || (min0 > min1 && min0 > max1)) {
            // found separating axis, polygons do not intersect
            return false;
        }
    }

    return true;
}

std::vector<vec<2>> convexHull(array_like<double>& argPoints) {

    auto points = argPoints.unchecked<2>();
    size_t points_len = points.shape(0);

    std::vector<vec<2>> ps(points_len);

    // convert to vec<2>

    for (size_t i = 0; i < points_len; ++i) {
        ps[i] = vec<2>(points(i, 0), points(i, 1));
    }

    if (points_len < 4) {
        return ps;
    }

    // find lowest point and place it at the front of the array

    for (size_t i = 1; i < points_len; ++i) {
        if (ps[i][1] > ps[0][1]) {
            continue;
        }
        if (ps[i][1] == ps[0][1] && ps[i][0] >= ps[0][0]) {
            continue;
        }
        vec<2> tmp = ps[0];
        ps[0] = ps[i];
        ps[i] = tmp;
    }

    auto orientation = [](const vec<2>& p, const vec<2>& q, const vec<2>& r) {

        double res = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

        if (res == 0) {
            // colinear
            return 0;
        } else if (res > 0) {
            // clockwise triangle
            return 1;
        } else {
            // counterclockwise triangle
            return 2;
        }
    };

    vec<2> p0 = ps[0];

    // sort all other points by angle

    std::sort(ps.begin() + 1, ps.end(),
              [p0, &orientation](vec<2>& p1, vec<2>& p2) {

        int o = orientation(p0, p1, p2);

        if (o == 0) {
            double d1 = sq(p0[0] - p1[0]) + sq(p0[1] - p1[1]);
            double d2 = sq(p0[0] - p2[0]) + sq(p0[1] - p2[1]);
            if (d1 <= d2) {
                return true;
            }
            return false;
        }

        if (o == 2) {
            return true;
        }

        return false;
    });

    // first step: for all co-linear points keep only the most distant point

    size_t uniquePoints = 1;
    for (size_t i = 1; i < points_len; i++) {
        while (i < points_len - 1 && orientation(p0, ps[i], ps[i+1]) == 0) {
           i++;
        }
        ps[uniquePoints] = ps[i];
        uniquePoints++;
    }

    ps.resize(uniquePoints);

    points_len = ps.size();

    if (points_len < 4) {
        return ps;
    }

    std::vector<vec<2>> stack;
    stack.push_back(ps[0]);

    // second step: keep only points, which form a counterclockwise triangle

    for (size_t i = 1; i < points_len; i++) {
        while (stack.size() > 1 && orientation(stack[stack.size()-2], stack.back(), ps[i]) != 2) {
            stack.pop_back();
        }
        stack.push_back(ps[i]);
    }

    return stack;
}

Projection project(array_like<double>& argPoints,
                   vec<2> position,
                   bool closed) {

    Projection proj;

    proj.distance = INFINITY;
    double proj_arc_len_offset = 0.0;

    auto points = argPoints.unchecked<2>();

    ssize_t points_len = points.shape(0);
    ssize_t end = points_len;
    if (closed) {
        end = points_len + 1;
    }

    vec<2> prev_p;
    vec<2> next_p;

    prev_p(0) = points(0, 0);
    prev_p(1) = points(0, 1);

    double arc_len = 0.0;

    /*
     * Projection search
     */

    for (ssize_t i = 1; i < end; ++i) {

        next_p(0) = points(i % points_len, 0);
        next_p(1) = points(i % points_len, 1);

        vec<2> pv = position - prev_p;
        vec<2> v = next_p - prev_p;
        double l = v.norm();

        double q = pv.dot(v) / v.dot(v);
        vec<2> p;
        bool in_bounds = true;
        if (q < 0) {
            in_bounds = !closed && i != 1;
            q = 0.0;
            p = prev_p;
        } else if (q > 1) {
            in_bounds = !closed && i != end-1;
            q = 1.0;
            p = next_p;
        } else {
            p = prev_p + v * q;
        }

        vec<2> dv = position - p;
        // we use the squared distance to avoid sqrt calls
        double d = dv(0)*dv(0) + dv(1)*dv(1);

        if (d < proj.distance) {
            proj.in_bounds = in_bounds;
            proj.distance = d;
            proj.point = p;
            proj.index = i;
            proj.alpha = q;
            proj_arc_len_offset = arc_len;
        }

        arc_len += l;
        prev_p = next_p;
    }

    /*
     * Projection post-processing
     */

    // correct distance

    proj.distance = std::sqrt(proj.distance);

    // compute indices

    ssize_t idx_start;
    if (closed) {
        idx_start = (proj.index - 1) % points_len;
	if (idx_start < 0) {
	    idx_start += points_len;
	}
    } else {
        idx_start = std::max(0L, proj.index - 1);
    }
    ssize_t idx_end;
    if (closed) {
	idx_end = proj.index % points_len;
    } else {
	idx_end = proj.index;
    }
    ssize_t idx_next;
    if (closed) {
        idx_next = (proj.index + 1) % points_len;
    } else {
        idx_next = std::min(points_len - 1, proj.index + 1);
    }

    proj.start = idx_start;
    proj.end = idx_end;
    if (proj.alpha < 0.5) {
        proj.index = idx_start;
    }

    // compute arc length
    
    prev_p(0) = points(idx_start, 0);
    prev_p(1) = points(idx_start, 1);
    next_p(0) = points(idx_end, 0);
    next_p(1) = points(idx_end, 1);

    vec<2> v = next_p - prev_p;
    double l = v.norm();
    v /= l;

    double arc_dist = (prev_p - proj.point).norm() * ((proj.alpha < 0) ? -1.0 : 1.0);
    proj.arc_len = proj_arc_len_offset + arc_dist;

    // compute interpolated tangent

    if (proj.index < points_len - 2) {
        vec<2> next_next_p;
        next_next_p(0) = points(idx_next, 0);
        next_next_p(1) = points(idx_next, 1);
        vec<2> next_v = next_next_p - next_p;
        next_v /= next_v.norm();
        proj.tangent = proj.alpha * next_v + (1.0 - proj.alpha) * v;
    } else {
        proj.tangent = v;
    }

    // compute angle
    
    proj.angle = std::atan2(proj.tangent(1), proj.tangent(0));

    // compute distance sign

    vec<2> ortho = (proj.point - position);
    ortho /= ortho.norm();

    double tmp = ortho(0);
    ortho(0) = -ortho(1);
    ortho(1) = tmp;

    if (v.dot(ortho) <= 0) { 
        proj.distance *= -1.0;
    }

    return proj;
}

std::vector<vec<5>> resample(array_like<double>& argPoints,
                             double samplingDist,
                             ssize_t steps,
                             ssize_t startIndex,
                             bool closed) {

    auto uncheckedPoints = argPoints.unchecked<2>();

    size_t len_points = uncheckedPoints.shape(0);

    std::vector<vec<2>> points(len_points);

    if (len_points == 0) {
        return {};
    }

    points[0] = {uncheckedPoints(0, 0), uncheckedPoints(0, 1)};

    ssize_t pointCount = 1;

    // convert to eigen vectors, also removes duplicate points
    for (ssize_t k = 1; k < uncheckedPoints.shape(0); ++k) {
        points[pointCount] = {uncheckedPoints(k, 0), uncheckedPoints(k, 1)};
        if ((points[pointCount] - points[pointCount-1]).norm() != 0) {
            pointCount += 1;
        }
    }

    if (pointCount == 1) {
        std::vector<vec<5>> sampled(1);
        sampled[0][0] = points[0][0];
        sampled[0][1] = points[0][1];
        sampled[0][2] = 0.0;
        sampled[0][3] = 0.0;
        sampled[0][4] = 0.0;
        return sampled;
    }

    if (closed) {
        startIndex %= pointCount;
    } else {
        startIndex = std::max(0l, std::min(startIndex, pointCount-1));
    }

    // x, y, alpha, prev, next
    std::vector<vec<5>> sampled(steps);
    for (ssize_t i = 0; i < steps; ++i) {
        sampled[i] = vec<5>::Zero();
    }
    sampled[0][0] = points[startIndex][0];
    sampled[0][1] = points[startIndex][1];
    sampled[0][2] = 0.0;
    sampled[0][3] = startIndex;
    if (closed) {
        sampled[0][4] = (startIndex + 1) % pointCount;
    } else {
        sampled[0][4] = std::max(0L, std::min(startIndex + 1, pointCount - 1));
    }

    ssize_t count = 1;
    ssize_t i = startIndex;

    while (count < steps) {
        ssize_t prevCount = count;
        for (ssize_t k = 0; k < pointCount; ++k) {
            ssize_t prevIdx = i + k;
            ssize_t nextIdx = i + k + 1;

            if (closed) {
                prevIdx %= pointCount;
                nextIdx %= pointCount;
            } else {
                prevIdx = std::max(0L, std::min(prevIdx, pointCount - 2));
                nextIdx = std::max(0L, std::min(nextIdx, pointCount - 1));
            }

            vec<2>& prevPoint = points[prevIdx];
            vec<2>& nextPoint = points[nextIdx];

            vec<2> v = nextPoint - prevPoint;
            double l = v.norm();
            double ls = l*l;
            vec<2> vNorm = v / l;

            // this calculates the intersection of the line segment
            // between prevPoint and nextPoint with a circle of radius
            // sampling dist centered at the last sampled point

            vec<5>& c = sampled[count - 1];

            double D = (prevPoint[0] - c[0])*(nextPoint[1] - c[1])
                     - (nextPoint[0] - c[0])*(prevPoint[1] - c[1]);
            double discriminant = samplingDist*samplingDist * ls - D*D;

            if (discriminant < 0) {
                // no intersection or tangent point
                // should not happen, abort mission
                throw std::runtime_error("cannot solve for next sampling point");
            }

            // calculate two (possibly distinct) intersection points

            double sqrtDiscr = std::sqrt(discriminant);
            double signY = (v[1] < 0.0) ? -1.0 : 1.0;

            double xPart0 = D * v[1];
            double yPart0 = -D * v[0];
            double xPart1 = signY * v[0] * sqrtDiscr;
            double yPart1 = std::abs(v[1]) * sqrtDiscr;

            vec<2> p0((xPart0 + xPart1) / ls + c[0],
                      (yPart0 + yPart1) / ls + c[1]);
            vec<2> p1((xPart0 - xPart1) / ls + c[0],
                      (yPart0 - yPart1) / ls + c[1]);

            vec<2> vp = p0 - prevPoint;
            double q0 = vNorm.dot(vp) / l;
            vp = p1 - prevPoint;
            double q1 = vNorm.dot(vp) / l;

            double tol = 1e-8;

            if (q0 < q1) {
                q0 = q1;
                p0 = p1;
            }

            if ((!closed && nextIdx == pointCount - 1)
                    || (q0 > -tol && q0 - 1.0 < tol)) {
                i = prevIdx;
                sampled[count][0] = p0[0];
                sampled[count][1] = p0[1];
                sampled[count][2] = q0;
                sampled[count][3] = prevIdx;
                sampled[count][4] = nextIdx;
                count += 1;
                break;
            }

            // intersection point is not on line segment
            // try next segment
        }
        if (count == prevCount) {
            // resampling failed
            throw std::runtime_error("resampling failed");
        }
    }

    return sampled;
}

void loadUtilsBindings(pybind11::module& m) {

    m.def("point_in_polygon", [](
            array_like<double> points,
            array_like<double> polygon) -> py::handle {

        assert_shape(points, {{2}, {-1, 2}});
        assert_shape(polygon, {{-1, 2}});

        if (points.ndim() == 1) {
            return py::cast(pointInPolygon(
                        points.at(0), points.at(1), polygon)).release();
        } 

        size_t pointCount = points.shape(0);

        py::array_t<bool> result(pointCount);

        auto ups = points.unchecked<2>();
        auto urs = result.mutable_unchecked<1>();

        for (size_t k = 0; k < pointCount; ++k) {
            urs(k) = pointInPolygon(ups(k, 0), ups(k, 1), polygon);
        }

        return result.release();
    });

    m.def("intersect_polygons", [](
            array_like<double> poly0,
            array_like<double> poly1) {

        assert_shape(poly0, {{-1, 2}});
        assert_shape(poly1, {{-1, 2}});

        return intersectPolygons(poly0, poly1);
    });

    m.def("convex_hull", [](
            array_like<double> points) {

        assert_shape(points, {{-1, 2}});

        std::vector<vec<2>> hull = convexHull(points);

        return py::array_t<double>(py::array_t<double>::ShapeContainer({(ssize_t)hull.size(), 2}),
                                   reinterpret_cast<double*>(hull.data()));
    });

    py::class_<Projection>(m, "Projection")
        .def(py::init())
        .def_readwrite("start", &Projection::start)
        .def_readwrite("end", &Projection::end)
        .def_readwrite("index", &Projection::index)
        .def_readwrite("alpha", &Projection::alpha)
        .def_readwrite("point", &Projection::point)
        .def_readwrite("distance", &Projection::distance)
        .def_readwrite("arc_len", &Projection::arc_len)
        .def_readwrite("tangent", &Projection::tangent)
        .def_readwrite("angle", &Projection::angle)
        .def_readwrite("in_bounds", &Projection::in_bounds)
        .def_property_readonly("__slots__",
            [](Projection& p){
                return std::vector<std::string>({
                                "start",
                                "end",
                                "index",
                                "alpha",
                                "point",
                                "distance",
                                "arc_len",
                                "tangent",
                                "angle",
                                "in_bounds"
                            });
            })
        .def(py::pickle(
            [](Projection& p) {
                pybind11::dict d;
                d["start"] = p.start;
                d["end"] = p.end;
                d["index"] = p.index;
                d["alpha"] = p.alpha;
                d["point"] = p.point;
                d["distance"] = p.distance;
                d["arc_len"] = p.arc_len;
                d["tangent"] = p.tangent;
                d["angle"] = p.angle;
                d["in_bounds"] = p.in_bounds;
                return d;
            }, 
            [](pybind11::dict& d) {
                Projection p;
                p.start = py::cast<int64_t>(d["start"]);
                p.end = py::cast<int64_t>(d["end"]);
                p.index = py::cast<int64_t>(d["index"]);
                p.alpha = py::cast<double>(d["alpha"]);
                p.point = py::cast<vec<2>>(d["point"]);
                p.distance = py::cast<double>(d["distance"]);
                p.arc_len = py::cast<double>(d["arc_len"]);
                p.tangent = py::cast<vec<2>>(d["tangent"]);
                p.angle = py::cast<double>(d["angle"]);
                p.in_bounds = py::cast<bool>(d["in_bounds"]);
                return p;
            }
        ));

    m.def("project", [](
                array_like<double> linestrip,
                array_like<double> points,
                bool closed) -> py::handle {

        assert_shape(points, {{2}, {-1, 2}});
        assert_shape(linestrip, {{-1, 2}});

        if (linestrip.shape(0) == 0) {
            return py::cast(Projection()).release();
        }

        if (points.ndim() == 1) {
            vec<2> position;
            position(0) = points.at(0);
            position(1) = points.at(1);
            return py::cast(project(linestrip, position, closed)).release();
        }

        if (points.ndim() == 2) {
            size_t pointCount = points.shape(0);
            std::vector<Projection> projs(pointCount);
            auto ups = points.unchecked<2>();

            for (size_t i = 0; i < pointCount; ++i) {
                vec<2> position;
                position(0) = ups(i, 0);
                position(1) = ups(i, 1);
                projs[i] = project(linestrip, position, closed);
            }

            return py::cast(projs).release();
        }

        return py::cast(Projection()).release();
    },
    py::arg("linestrip"),
    py::arg("points"),
    py::arg("closed") = false);

    m.def("resample", [](array_like<double>& points,
                         double samplingDist,
                         ssize_t steps,
                         ssize_t startIndex,
                         bool closed) {

        assert_shape(points, {{-1, 2}});

        std::vector<vec<5>> res = resample(points, samplingDist, steps, startIndex, closed);
        return py::array_t<double>(py::array_t<double>::ShapeContainer({(ssize_t)res.size(), 5}),
                                   reinterpret_cast<double*>(res.data()));
    },
    py::arg("points"),
    py::arg("sampling_dist"),
    py::arg("steps"),
    py::arg("start_index") = 0,
    py::arg("closed") = false);

    py::class_<PolyCubic>(m, "PolyCubic")
        .def(py::init<double, double, double, double, double, double>())
        .def_readwrite("x0_", &PolyCubic::x0_)
        .def_readwrite("x1_", &PolyCubic::x1_)
        .def_property_readonly("c", [](PolyCubic& p){
            return std::vector<double>(&p.c[0], &p.c[0] + 4);
        })
        .def("f", &PolyCubic::f)
        .def("df", &PolyCubic::df)
        .def("ddf", &PolyCubic::ddf)
        .def("dddf", &PolyCubic::dddf)
        .def("i1", &PolyCubic::i1)
        .def("i2", &PolyCubic::i2);

    py::class_<PolyQuintic>(m, "PolyQuintic")
        .def(py::init<double, double, double, double, double, double, double, double>())
        .def_readwrite("x0_", &PolyQuintic::x0_)
        .def_readwrite("x1_", &PolyQuintic::x1_)
        .def_property_readonly("c", [](PolyQuintic& p){
            return std::vector<double>(&p.c[0], &p.c[0] + 6);
        })
        .def("f", &PolyQuintic::f)
        .def("df", &PolyQuintic::df)
        .def("ddf", &PolyQuintic::ddf)
        .def("dddf", &PolyQuintic::dddf);

    py::class_<PolySeptic>(m, "PolySeptic")
        .def(py::init<double, double, double, double, double, double, double, double, double, double>())
        .def_readwrite("x0_", &PolySeptic::x0_)
        .def_readwrite("x1_", &PolySeptic::x1_)
        .def_property_readonly("c", [](PolySeptic& p){
            return std::vector<double>(&p.c[0], &p.c[0] + 8);
        })
        .def("f", &PolySeptic::f)
        .def("df", &PolySeptic::df)
        .def("ddf", &PolySeptic::ddf)
        .def("dddf", &PolySeptic::dddf);
}
