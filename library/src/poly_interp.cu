#include "tplcpp/poly_interp.cuh"

/*
 * Automatically generated code from sympy
 */

__device__ __host__ void calc_poly3_coeffs(double y0, double y0d, double y1, double y1d, double *out_7941153176836543462) {

   out_7941153176836543462[0] = y0;
   out_7941153176836543462[1] = y0d;
   out_7941153176836543462[2] = -3*y0 - 2*y0d + 3*y1 - y1d;
   out_7941153176836543462[3] = 2*y0 + y0d - 2*y1 + y1d;

}

__device__ __host__ double calc_poly3_y(double *c, double t) {
   double x0[4];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];

   double calc_poly3_y_result;
   calc_poly3_y_result = t*(t*(t*x0[3] + x0[2]) + x0[1]) + x0[0];
   return calc_poly3_y_result;

}

__device__ __host__ double calc_poly3_y_d1(double *c, double t) {
   double x0[4];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];

   double calc_poly3_y_d1_result;
   calc_poly3_y_d1_result = t*(3*t*x0[3] + 2*x0[2]) + x0[1];
   return calc_poly3_y_d1_result;

}

__device__ __host__ double calc_poly3_y_d2(double *c, double t) {
   double x0[4];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];

   double calc_poly3_y_d2_result;
   calc_poly3_y_d2_result = 6*t*x0[3] + 2*x0[2];
   return calc_poly3_y_d2_result;

}

__device__ __host__ double calc_poly3_y_d3(double *c) {

   double calc_poly3_y_d3_result;
   calc_poly3_y_d3_result = 6*c[3];
   return calc_poly3_y_d3_result;

}

__device__ __host__ void calc_poly3_y_d0_2(double *c, double t, double *out_8389374448110633143) {
   double x0[4];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   const double x1 = x0[1];
   const double x2 = x0[2];
   const double x3 = t*x0[3];
   const double x4 = 2*x2;

   out_8389374448110633143[0] = t*(t*(x2 + x3) + x1) + x0[0];
   out_8389374448110633143[1] = t*(3*x3 + x4) + x1;
   out_8389374448110633143[2] = 6*x3 + x4;

}

__device__ __host__ double calc_poly3_y_i1(double *c, double t) {

   double calc_poly3_y_i1_result;
   calc_poly3_y_i1_result = t*(t*(t*((1.0/4.0)*t*c[3] + (1.0/3.0)*c[2]) + (1.0/2.0)*c[1]) + c[0]);
   return calc_poly3_y_i1_result;

}

__device__ __host__ double calc_poly3_y_i2(double *c, double t) {

   double calc_poly3_y_i2_result;
   calc_poly3_y_i2_result = t*(t*(t*(t*((1.0/20.0)*t*c[3] + (1.0/12.0)*c[2]) + (1.0/6.0)*c[1]) + (1.0/2.0)*c[0]));
   return calc_poly3_y_i2_result;

}

__device__ __host__ void calc_poly5_coeffs(double y0, double y0d, double y0dd, double y1, double y1d, double y1dd, double *out_8596309956639620159) {
   const double x0 = (1.0/2.0)*y0dd;
   const double x1 = (3.0/2.0)*y0dd;
   const double x2 = (1.0/2.0)*y1dd;

   out_8596309956639620159[0] = y0;
   out_8596309956639620159[1] = y0d;
   out_8596309956639620159[2] = x0;
   out_8596309956639620159[3] = -x1 + x2 - 10*y0 - 6*y0d + 10*y1 - 4*y1d;
   out_8596309956639620159[4] = x1 + 15*y0 + 8*y0d - 15*y1 + 7*y1d - y1dd;
   out_8596309956639620159[5] = -x0 + x2 - 6*y0 - 3*y0d + 6*y1 - 3*y1d;

}

__device__ __host__ double calc_poly5_y(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_result;
   calc_poly5_y_result = t*(t*(t*(t*(t*x0[5] + x0[4]) + x0[3]) + x0[2]) + x0[1]) + x0[0];
   return calc_poly5_y_result;

}

__device__ __host__ double calc_poly5_y_d1(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_d1_result;
   calc_poly5_y_d1_result = t*(t*(t*(5*t*x0[5] + 4*x0[4]) + 3*x0[3]) + 2*x0[2]) + x0[1];
   return calc_poly5_y_d1_result;

}

__device__ __host__ double calc_poly5_y_d2(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_d2_result;
   calc_poly5_y_d2_result = t*(t*(20*t*x0[5] + 12*x0[4]) + 6*x0[3]) + 2*x0[2];
   return calc_poly5_y_d2_result;

}

__device__ __host__ double calc_poly5_y_d3(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_d3_result;
   calc_poly5_y_d3_result = t*(60*t*x0[5] + 24*x0[4]) + 6*x0[3];
   return calc_poly5_y_d3_result;

}

__device__ __host__ void calc_poly5_y_d0_2(double *c, double t, double *out_1667342037099268689) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   const double x1 = x0[1];
   const double x2 = x0[2];
   const double x3 = x0[3];
   const double x4 = x0[4];
   const double x5 = t*x0[5];
   const double x6 = 2*x2;

   out_1667342037099268689[0] = t*(t*(t*(t*(x4 + x5) + x3) + x2) + x1) + x0[0];
   out_1667342037099268689[1] = t*(t*(t*(4*x4 + 5*x5) + 3*x3) + x6) + x1;
   out_1667342037099268689[2] = t*(t*(12*x4 + 20*x5) + 6*x3) + x6;

}

__device__ __host__ double calc_poly5_y_i1(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_i1_result;
   calc_poly5_y_i1_result = t*(t*(t*(t*(t*((1.0/6.0)*t*x0[5] + (1.0/5.0)*x0[4]) + (1.0/4.0)*x0[3]) + (1.0/3.0)*x0[2]) + (1.0/2.0)*x0[1]) + x0[0]);
   return calc_poly5_y_i1_result;

}

__device__ __host__ double calc_poly5_y_i2(double *c, double t) {
   double x0[6];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];

   double calc_poly5_y_i2_result;
   calc_poly5_y_i2_result = t*t*(t*(t*(t*(t*((1.0/42.0)*t*x0[5] + (1.0/30.0)*x0[4]) + (1.0/20.0)*x0[3]) + (1.0/12.0)*x0[2]) + (1.0/6.0)*x0[1]) + (1.0/2.0)*x0[0]);
   return calc_poly5_y_i2_result;

}

__device__ __host__ void calc_poly7_coeffs(double y0, double y0d, double y0dd, double y0ddd, double y1, double y1d, double y1dd, double y1ddd, double *out_4166417348511440398) {
   const double x0 = (1.0/6.0)*y0ddd;
   const double x1 = -2.0/3.0*y0ddd;
   const double x2 = (1.0/6.0)*y1ddd;
   const double x3 = (1.0/2.0)*y1ddd;

   out_4166417348511440398[0] = y0;
   out_4166417348511440398[1] = y0d;
   out_4166417348511440398[2] = (1.0/2.0)*y0dd;
   out_4166417348511440398[3] = x0;
   out_4166417348511440398[4] = x1 - x2 - 35*y0 - 20*y0d - 5*y0dd + 35*y1 - 15*y1d + (5.0/2.0)*y1dd;
   out_4166417348511440398[5] = x3 + 84*y0 + 45*y0d + 10*y0dd + y0ddd - 84*y1 + 39*y1d - 7*y1dd;
   out_4166417348511440398[6] = x1 - x3 - 70*y0 - 36*y0d - 15.0/2.0*y0dd + 70*y1 - 34*y1d + (13.0/2.0)*y1dd;
   out_4166417348511440398[7] = x0 + x2 + 20*y0 + 10*y0d + 2*y0dd - 20*y1 + 10*y1d - 2*y1dd;

}

__device__ __host__ double calc_poly7_y(double *c, double t) {
   double x0[8];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   x0[6] = c[6];
   x0[7] = c[7];

   double calc_poly7_y_result;
   calc_poly7_y_result = t*(t*(t*(t*(t*(t*(t*x0[7] + x0[6]) + x0[5]) + x0[4]) + x0[3]) + x0[2]) + x0[1]) + x0[0];
   return calc_poly7_y_result;

}

__device__ __host__ double calc_poly7_y_d1(double *c, double t) {
   double x0[8];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   x0[6] = c[6];
   x0[7] = c[7];

   double calc_poly7_y_d1_result;
   calc_poly7_y_d1_result = t*(t*(t*(t*(t*(7*t*x0[7] + 6*x0[6]) + 5*x0[5]) + 4*x0[4]) + 3*x0[3]) + 2*x0[2]) + x0[1];
   return calc_poly7_y_d1_result;

}

__device__ __host__ double calc_poly7_y_d2(double *c, double t) {
   double x0[8];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   x0[6] = c[6];
   x0[7] = c[7];

   double calc_poly7_y_d2_result;
   calc_poly7_y_d2_result = t*(t*(t*(t*(42*t*x0[7] + 30*x0[6]) + 20*x0[5]) + 12*x0[4]) + 6*x0[3]) + 2*x0[2];
   return calc_poly7_y_d2_result;

}

__device__ __host__ double calc_poly7_y_d3(double *c, double t) {
   double x0[8];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   x0[6] = c[6];
   x0[7] = c[7];

   double calc_poly7_y_d3_result;
   calc_poly7_y_d3_result = t*(t*(t*(210*t*x0[7] + 120*x0[6]) + 60*x0[5]) + 24*x0[4]) + 6*x0[3];
   return calc_poly7_y_d3_result;

}

__device__ __host__ void calc_poly7_y_d0_2(double *c, double t, double *out_6170446920064703032) {
   double x0[8];
   x0[0] = c[0];
   x0[1] = c[1];
   x0[2] = c[2];
   x0[3] = c[3];
   x0[4] = c[4];
   x0[5] = c[5];
   x0[6] = c[6];
   x0[7] = c[7];
   const double x1 = x0[1];
   const double x2 = x0[2];
   const double x3 = x0[3];
   const double x4 = x0[4];
   const double x5 = x0[5];
   const double x6 = x0[6];
   const double x7 = t*x0[7];
   const double x8 = 2*x2;

   out_6170446920064703032[0] = t*(t*(t*(t*(t*(t*(x6 + x7) + x5) + x4) + x3) + x2) + x1) + x0[0];
   out_6170446920064703032[1] = t*(t*(t*(t*(t*(6*x6 + 7*x7) + 5*x5) + 4*x4) + 3*x3) + x8) + x1;
   out_6170446920064703032[2] = t*(t*(t*(t*(30*x6 + 42*x7) + 20*x5) + 12*x4) + 6*x3) + x8;

}

/*
 * Poly wrapper classes
 */

__device__ __host__ PolyCubic::PolyCubic(double x0,
                     double y0,
                     double dy0,
                     double x1,
                     double y1,
                     double dy1) : x0_{x0}, x1_{x1}, d{x1 - x0} {

    calc_poly3_coeffs(y0 / d, dy0,
                      y1 / d, dy1,
                      &c[0]);
}

__device__ __host__ double PolyCubic::f(double x) {
    double t = (x - x0_) / d;
    return calc_poly3_y(&c[0], t) * d;
}

__device__ __host__ double PolyCubic::df(double x) {
    double t = (x - x0_) / d;
    return calc_poly3_y_d1(&c[0], t);
}

__device__ __host__ double PolyCubic::ddf(double x) {
    double t = (x - x0_) / d;
    return calc_poly3_y_d2(&c[0], t) / d;
}

__device__ __host__ double PolyCubic::dddf(double x) {
    return calc_poly3_y_d3(&c[0]) / (d*d);
}

__device__ __host__ double PolyCubic::i1(double x, double ic0) {
    double t = (x - x0_) / d;
    return ic0 + calc_poly3_y_i1(&c[0], t) * d*d;
}

__device__ __host__ double PolyCubic::i2(double x, double ic0, double ic1) {
    double t = (x - x0_) / d;
    return ic1 + (x - x0_)*ic0 + calc_poly3_y_i2(&c[0], t) * d*d*d;
}

__device__ __host__ void PolyCubic::df0to2(double x, double* y) {
    double t = (x - x0_) / d;
    calc_poly3_y_d0_2(&c[0], t, y);
    y[0] *= d;
    y[1] = d;
    y[2] /= d;
}

__device__ __host__ PolyQuintic::PolyQuintic(double x0,
                         double y0,
                         double dy0,
                         double ddy0,
                         double x1,
                         double y1,
                         double dy1,
                         double ddy1) : x0_{x0}, x1_{x1}, d{x1 - x0} {

    calc_poly5_coeffs(y0 / d, dy0, ddy0 * d,
                      y1 / d, dy1, ddy1 * d,
                      &c[0]);
}

__device__ __host__ double PolyQuintic::f(double x) {
    double t = (x - x0_) / d;
    return calc_poly5_y(&c[0], t) * d;
}

__device__ __host__ double PolyQuintic::df(double x) {
    double t = (x - x0_) / d;
    return calc_poly5_y_d1(&c[0], t);
}

__device__ __host__ double PolyQuintic::ddf(double x) {
    double t = (x - x0_) / d;
    return calc_poly5_y_d2(&c[0], t) / d;
}

__device__ __host__ double PolyQuintic::dddf(double x) {
    double t = (x - x0_) / d;
    return calc_poly5_y_d3(&c[0], t) / (d*d);
}

__device__ __host__ double PolyQuintic::i1(double x, double ic0) {
    double t = (x - x0_) / d;
    return ic0 + calc_poly5_y_i1(&c[0], t) * d*d;
}

__device__ __host__ double PolyQuintic::i2(double x, double ic0, double ic1) {
    double t = (x - x0_) / d;
    return ic1 + (x - x0_)*ic0 + calc_poly5_y_i2(&c[0], t) * d*d*d;
}

__device__ __host__ void PolyQuintic::df0to2(double x, double* y) {
    double t = (x - x0_) / d;
    calc_poly5_y_d0_2(&c[0], t, y);
    y[0] *= d;
    y[1] = d;
    y[2] /= d;
}

__device__ __host__ PolySeptic::PolySeptic(double x0,
                       double y0,
                       double dy0,
                       double ddy0,
                       double dddy0,
                       double x1,
                       double y1,
                       double dy1,
                       double ddy1,
                       double dddy1) : x0_{x0}, x1_{x1}, d{x1 - x0} {

    calc_poly7_coeffs(y0 / d, dy0, ddy0 * d, dddy0 * d*d,
                      y1 / d, dy1, ddy1 * d, dddy1 * d*d,
                      &c[0]);
}

__device__ __host__ double PolySeptic::f(double x) {
    double t = (x - x0_) / d;
    return calc_poly7_y(&c[0], t) * d;
}

__device__ __host__ double PolySeptic::df(double x) {
    double t = (x - x0_) / d;
    return calc_poly7_y_d1(&c[0], t);
}

__device__ __host__ double PolySeptic::ddf(double x) {
    double t = (x - x0_) / d;
    return calc_poly7_y_d2(&c[0], t) / d;
}

__device__ __host__ double PolySeptic::dddf(double x) {
    double t = (x - x0_) / d;
    return calc_poly7_y_d3(&c[0], t) / (d*d);
}

__device__ __host__ void PolySeptic::df0to2(double x, double* y) {
    double t = (x - x0_) / d;
    calc_poly7_y_d0_2(&c[0], t, y);
    y[0] *= d;
    y[1] = d;
    y[2] /= d;
}

__device__ __host__ PolyQuartic::PolyQuartic(double ts, double ss, double vs, double as,
                                             double te, double ve, double ae) {

    this->ts = ts;

    double dt = te - ts;

    a0 = ss;
    a1 = vs;
    a2 = as / 2.0f;

    double A[2][2];
    A[0][0] = 3.0f * dt*dt;
    A[0][1] = 4.0f * dt*dt*dt;
    A[1][0] = 6.0f * dt;
    A[1][1] = 12.0f * dt*dt;

    double b[2];
    b[0] = ve - a1 - 2.0f * a2 * dt;
    b[1] = ae - 2.0f * a2;

    double q = 1.0 / (A[0][0]*A[1][1] - A[0][1]*A[1][0]);
    double A_inv[2][2];
    A_inv[0][0] = A[1][1] * q;
    A_inv[1][1] = A[0][0] * q;
    A_inv[0][1] = -A[0][1] * q;
    A_inv[1][0] = -A[1][0] * q;

    a3 = A_inv[0][0] * b[0] + A_inv[0][1] * b[1];
    a4 = A_inv[1][0] * b[0] + A_inv[1][1] * b[1];
}

__device__ __host__ double PolyQuartic::f(double t) {
    t -= ts;
    return a0 + a1 * t + a2 * t*t + a3 * t*t*t + a4 * t*t*t*t;
}

__device__ __host__ double PolyQuartic::df(double t) {
    t -= ts;
    return a1 + 2.0 * a2 * t + 3.0 * a3 * t*t + 4.0 * a4 * t*t*t;
}

__device__ __host__ double PolyQuartic::ddf(double t) {
    t -= ts;
    return 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t*t;
}

__device__ __host__ double PolyQuartic::dddf(double t) {
    t -= ts;
    return 6.0 * a3 + 24.0 * a4 * t;
}
