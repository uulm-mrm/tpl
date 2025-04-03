#ifdef __CUDACC__
    #define __interop__ __host__ __device__
#else
    #define __interop__ 
#endif

struct PolyCubic {

    double x0_ = 0.0;
    double x1_ = 0.0;
    double d = 0.0;
    double c[4];

    __interop__ PolyCubic() {};

    __interop__ PolyCubic(double x0,
              double y0,
              double dy0,
              double x1,
              double y1,
              double dy1);

    __interop__ double f(double x);
    __interop__ double df(double x);
    __interop__ double ddf(double x);
    __interop__ double dddf(double x);

    __interop__ double i1(double x, double ic0);
    __interop__ double i2(double x, double ic0, double ic1);

    __interop__ void df0to2(double x, double* y);
};

struct PolyQuintic {

    double x0_ = 0.0;
    double x1_ = 0.0;
    double d = 0.0;
    double c[6];

    __interop__ PolyQuintic() {};

    __interop__ PolyQuintic(double x0,
                            double y0,
                            double dy0,
                            double ddy0,
                            double x1,
                            double y1,
                            double dy1,
                            double ddy1); 

    __interop__ double f(double x);
    __interop__ double df(double x);
    __interop__ double ddf(double x);
    __interop__ double dddf(double x);

    __interop__ double i1(double x, double ic0);
    __interop__ double i2(double x, double ic0, double ic1);

    __interop__ void df0to2(double x, double* y);
};

struct PolySeptic {

    double x0_ = 0.0;
    double x1_ = 0.0;
    double d = 0.0;
    double c[8];

    __interop__ PolySeptic() {};

    __interop__ PolySeptic(double x0,
                           double y0,
                           double dy0,
                           double ddy0,
                           double dddy0,
                           double x1,
                           double y1,
                           double dy1,
                           double ddy1,
                           double dddy1); 

    __interop__ double f(double x);
    __interop__ double df(double x);
    __interop__ double ddf(double x);
    __interop__ double dddf(double x);

    __interop__ void df0to2(double x, double* y);
};

struct PolyQuartic {

    double ts = 0.0;
    double a0 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double a4 = 0.0;

    __interop__ PolyQuartic(double ts,
                            double ss,
                            double vs,
                            double as,
                            double te,
                            double ve,
                            double ae);

    __interop__ double f(double t);
    __interop__ double df(double t);
    __interop__ double ddf(double t);
    __interop__ double dddf(double t);
};
