#define _USE_MATH_DEFINES

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <stdbool.h>

#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

// alternative assert implementation (which does not simply crash)

bool errorFlag = false;
char errorMsgBuffer[2048];

#define check(expr) __check(expr, #expr)

#if NDEBUG
    #define RETURN_ON_ERROR
    #define RETURN_ZERO_ON_ERROR

    void __check(bool expression, const char* str) {
        return;
    }
#else
    #define RETURN_ON_ERROR if (errorFlag) { return; }
    #define RETURN_ZERO_ON_ERROR if (errorFlag) { return 0.0; }

    void __check(bool expression, const char* str) {

        if (expression) {
            return;
        }
        if (!errorFlag && !expression) {
            errorFlag = true;
            strncpy(errorMsgBuffer, str, min(strlen(str), 2048-1));
            errorMsgBuffer[2047] = 0;
        }
    }
#endif

#define X @X_DIMS
#define U @U_DIMS
#define C @C_DIMS
#define H_MAX 300

#define Optim Opt@CODE_HASH

#define rows(A) sizeof(A) / sizeof(A[0])
#define cols(A) sizeof(A[0]) / sizeof(double)

/*
 * This is a very smol blas.
 * Probably not very performant, but useable at least.
 */

void __copy (int m, int n, double* a, double* res) {

    RETURN_ON_ERROR

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n + j] = a[i*n + j];
        }
    }
}
#define copy(R, A) \
    assert(rows(A) == rows(R)); \
    assert(cols(A) == cols(R)); \
    __copy(rows(A), cols(A), (double*)&A, (double*)&R)

void __eye (int m, int n, double a[]) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}
#define eye(A) __eye(rows(A), cols(A), (double*)&A)

void __zeros (int m, int n, double a[]) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i*n + j] = 0.0;
        }
    }
}
#define zeros(A) __zeros(rows(A), cols(A), (double*)&A)

void __ones (int m, int n, double a[]) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i*n + j] = 1.0;
        }
    }
}
#define ones(A) __ones(rows(A), cols(A), (double*)&A)

void __add (int m, int n, double* a0, double* a1, double* res) {

    RETURN_ON_ERROR

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n + j] = a0[i*n + j] + a1[i*n + j];
        }
    }
}
#define add(R, A, B) \
    check(rows(A) == rows(B)); \
    check(rows(A) == rows(R)); \
    check(cols(A) == cols(B)); \
    check(cols(A) == cols(R)); \
    __add(rows(A), cols(A), (double*)&A, (double*)&B, (double*)&R)

void __sub (int m, int n, double* a0, double* a1, double* res) {

    RETURN_ON_ERROR

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n + j] = a0[i*n + j] - a1[i*n + j];
        }
    }
}
#define sub(R, A, B) \
    check(rows(A) == rows(B)); \
    check(rows(A) == rows(R)); \
    check(cols(A) == cols(B)); \
    check(cols(A) == cols(R)); \
    __sub(rows(A), cols(A), (double*)&A, (double*)&B, (double*)&R)

void __mul (int m, int n, int o, double* a0, double* a1, double* res) {

    RETURN_ON_ERROR

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < o; ++j) {
            res[i*o + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                res[i*o + j] += a0[i*n + k] * a1[k*o + j];
            }
        }
    }
}
#define mul(R, A, B) \
    check(cols(A) == rows(B)); \
    check(rows(A) == rows(R)); \
    check(cols(B) == cols(R)); \
    __mul(rows(A), cols(A), cols(B), (double*)&A, (double*)&B, (double*)&R)

void __mul_symm (int m, int n, int o, double* a0, double* a1, double* res) {

    RETURN_ON_ERROR

    // can be used if the result is symmetric
    // seems to only give improvements if the matrix is larger e.g 8x8

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < (i+1); ++j) {
            res[i*o + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                res[i*o + j] += a0[i*n + k] * a1[k*o + j];
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < (i+1); ++j) {
            res[j*m + i] = res[i*o + j];
        }
    }
}
#define mul_symm(R, A, B) \
    check(cols(A) == rows(B)); \
    check(rows(A) == rows(R)); \
    check(cols(B) == cols(R)); \
    __mul_symm(rows(A), cols(A), cols(B), (double*)&A, (double*)&B, (double*)&R)

void __mul_add (int m, int n, int o, double* a0, double* a1, double* res) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < o; ++j) {
            for (int k = 0; k < n; ++k) {
                res[i*o + j] += a0[i*n + k] * a1[k*o + j];
            }
        }
    }
}
#define mul_add(R, A, B) \
    check(cols(A) == rows(B)); \
    check(rows(A) == rows(R)); \
    check(cols(B) == cols(R)); \
    __mul_add(rows(A), cols(A), cols(B), (double*)&A, (double*)&B, (double*)&R)

void __mul_scalar (int m, int n, double a[], double f, double res[]) {

    RETURN_ON_ERROR

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n + j] = a[i*n + j] * f;
        }
    }
}
#define mul_scalar(R, A, F) __mul_scalar(rows(A), cols(A), (double*)&A, F, (double*)R)

void __transpose (int m, int n, double a[], double res[]) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[j*m + i] = a[i*n + j];
        }
    }
}
#define transpose(R, A) \
    check(rows(A) == cols(R)); \
    check(cols(A) == rows(R)); \
    __transpose(rows(A), cols(A), (double*)&A, (double*)&R)

void __show (const char* name, int m, int n, double a[]) {

    printf("%s:\n", name);
    for (int i = 0; i < m; ++i) {
        printf("[");
        for (int j = 0; j < n-1; ++j) {
            printf("%g, ", a[i*n + j]);
        }
        printf("%g", a[i*n + n-1]);
        printf("]\n");
    }
    printf("\n");
}
#define show(A) __show(#A, rows(A), cols(A), (double*)&A)

void solve_action(double (*Quu)[U][U],
                  double (*Qu)[U],
                  double (*Qux)[U][X],
                  double (*k)[U],
                  double (*K)[U][X],
                  double mu) {

    RETURN_ON_ERROR

    @SOLVE_CODE

    #ifdef SOLVE_1D

    double inv = 0.0;

    if ((*Quu)[0][0] > 0.0) {
        inv = -1.0 / ((*Quu)[0][0] + mu);
    }

    mul_scalar(*k, *Qu, inv);
    mul_scalar(*K, *Qux, inv);

    #endif

    #ifdef SOLVE_2D

    double a = (*Quu)[0][0] + mu;
    double b = (*Quu)[0][1];
    double d = (*Quu)[1][1] + mu;

    double inv[2][2];
    inv[0][0] = d;
    inv[0][1] = -b;
    inv[1][0] = -b;
    inv[1][1] = a;

    double det = a*d - b*b;

    double inv_det = -1.0f / det;
    inv[0][0] *= inv_det;
    inv[0][1] *= inv_det;
    inv[1][0] = inv[0][1];
    inv[1][1] *= inv_det;

    mul(*k, inv, *Qu);
    mul(*K, inv, *Qux);

    #endif
}

/*
 * Utils, very useful!
 */

typedef struct {
    size_t ndim;
    long int dims[8];
    size_t nbytes;
    double* data;
} DynArray;

void freeDynArray (DynArray* arr) {

    if (arr->data != NULL) {
        free(arr->data);
        arr->data = NULL;
    }
}

void copyDynArray (DynArray* dst, DynArray* src) {

    double* newData;

    if (dst->nbytes != src->nbytes) {
        if (dst->data != NULL) {
            free(dst->data);
        } 
        newData = (double*)malloc(src->nbytes);
    } else { 
        newData = dst->data;
    }

    memcpy(newData, src->data, src->nbytes);
    *dst = *src;
    dst->data = newData;
}

#define get_array_value(A, I) A.data[(size_t)I]

double shortAngleDist (double a0, double a1) {

    double m = M_PI * 2;
    double da = fmod(a1 - a0, m);

    return fmod(2 * da, m) - da;
}

typedef struct {
    double a;
    double ia;
    size_t start;
    size_t end;
} Interp;

void initInterp (Interp* i, double x0, double dx, double x, size_t size) {

    const double q = (x - x0) / dx;

    i->start = min(size - 1, max(0lu, (size_t)floor(q)));
    i->end = min(size - 1, max(0lu, (size_t)ceil(q)));
    i->a = min(max(0.0, q - i->start), 1.0);
    i->ia = 1.0 - i->a;
}

double __box_interp(double dx, double x, DynArray arr) {

    RETURN_ZERO_ON_ERROR

    size_t l = arr.dims[0];

    if (l == 0) {
        return 0.0;
    }

    size_t i = min(l - 1, max(0, (size_t)(floor(x / dx))));

    return arr.data[i];
}

#define box_interp(DX, X, ARR) (check(ARR.ndim == 1), __box_interp(DX, X, ARR))

double __lerp(double x0, double dx, double x, DynArray arr) {

    RETURN_ZERO_ON_ERROR

    size_t l = arr.dims[0];

    if (l == 0) {
        return 0.0;
    }

    Interp i;
    initInterp(&i, x0, dx, x, l);

    return i.ia * arr.data[i.start] + i.a * arr.data[i.end];
}

#define lerp(X0, DX, X, ARR) (check(ARR.ndim == 1), __lerp(X0, DX, X, ARR))

double __lerp_angle(double x0, double dx, double x, DynArray arr) {

    RETURN_ZERO_ON_ERROR

    size_t l = arr.dims[0];

    if (l == 0) {
        return 0.0;
    }

    Interp i;
    initInterp(&i, x0, dx, x, l);

    return arr.data[i.start] + shortAngleDist(arr.data[i.start], arr.data[i.end]) * i.a;
}

#define lerp_angle(X0, DX, X, ARR) (check(ARR.ndim == 1), __lerp_angle(X0, DX, X, ARR))

double __lerp_wrap(double len, double dx, double x, DynArray xs, DynArray arr) {

    RETURN_ZERO_ON_ERROR

    size_t l = arr.dims[0];

    if (l == 0) {
        return 0.0;
    }

    double first = xs.data[0];
    double last = first + (l-1) * dx;
    double gap = len - (last - first);

    x = fmod(x - first, len);
    if (x < 0) {
        x += len;
    }
    x += first;

    double alpha;
    size_t start;
    size_t end;

    if (x >= last && gap > 0) {
        alpha = (x - last) / gap;
        start = l - 1;
        end = 0;
    } else {
        const double q = (x - first) / dx;
        start = (size_t)floor(q);
        end = (size_t)ceil(q);
        alpha = q - start;
    }

    double val = (1.0 - alpha) * arr.data[start] + alpha * arr.data[end];

    return val;
}

#define lerp_wrap(LEN, DX, X, XS, ARR) ( \
        check(ARR.ndim == 1), \
        check(XS.ndim == 1), \
        check(ARR.dims[0] == XS.dims[0]), \
        __lerp_wrap(LEN, DX, X, XS, ARR) \
    )

double __blerp(double x0,
               double y0,
               double dx,
               double dy,
               double x,
               double y,
               DynArray arr) {

    RETURN_ZERO_ON_ERROR

    size_t rows = arr.dims[0];
    size_t cols = arr.dims[1];

    Interp ix;
    initInterp(&ix, x0, dx, x, cols);
    Interp iy;
    initInterp(&iy, y0, dy, y, rows);

    double p0 = iy.ia * arr.data[iy.start*cols + ix.start] 
                + iy.a * arr.data[iy.end*cols + ix.start];
    double p1 = iy.ia * arr.data[iy.start*cols + ix.end]
                + iy.a * arr.data[iy.end*cols + ix.end];

    return ix.ia * p0 + ix.a * p1;
}

#define blerp(X0, Y0, DX, DY, X, Y, ARR) ( \
        check(ARR.ndim == 2), \
        __blerp(X0, Y0, DX, DY, X, Y, ARR) \
    )

/*
 * Optimizer implementation
 */

typedef enum {
    EULER = 0,
    HEUN = 1,
    RK4 = 2
} IntegratorType;

/*
 * definition of the parameter struct
 */

@PARAM_CODE

/*
 * actual optimizer definition
 */

typedef struct {

    // integration step
    double int_step[H_MAX];
    // current state sequence
    double x[H_MAX][X];
    // current action sequence
    double u[H_MAX][U];

    // integration step used in linesearch
    double next_int_step[H_MAX];
    // next state sequence, used in linesearch
    double next_x[H_MAX][X];
    // next action sequence, used in linesearch
    double next_u[H_MAX][U];

    // previous state sequence
    double prev_x[H_MAX][X];
    // previous action sequence
    double prev_u[H_MAX][U];
    // previous action gradients
    double prev_k[H_MAX][U];

    // parts of the dynamics function jacobian wrt. state
    double fx[H_MAX][X][X];
    // parts of the dynamics function jacobian wrt. action
    double fu[H_MAX][X][U];

    // parts of the cost function gradient wrt. state
    double lx[H_MAX][X];
    // parts of the cost function gradient wrt. action
    double lu[H_MAX][U];

    // parts of the cost function hessian wrt. state, state
    double lxx[H_MAX][X][X];
    // parts of the cost function hessian wrt. action, action
    double luu[H_MAX][U][U];
    // parts of the cost function hessian wrt. action, state
    double lux[H_MAX][U][X];

    // control gradient
    double g[H_MAX][U];

    // constant control components
    double k[H_MAX][U];
    // linear control components
    double K[H_MAX][U][X];

    // lagrange multipliers over time and constraint index
    double lagrangeMultiplier[H_MAX][C];
    // barrier weights over constraint index
    double barrierWeight[C];
    // maximum absolute value of the lagrange multiplier
    double lgMultLimit[C];
    
    // initial cost
    double trajCosts;

    // amount of iterations done in last update
    int iterations;

    // amount of lagrange update iterations done in last update
    int lgIterations;

    // the wall clock time needed for one update
    double runtime;

    // chosen line search scaling parameter
    double alpha;

    // true if the regularization changed or the trajectory improved
    bool trajectoryChanged;

    // if the trajectory was improved
    bool improved;

    // describes why the algorithm terminated
    int terminationCondition;

    // regularization parameter (to ensure positive definite Quu)
    double mu;

    // regularization step
    int muStep;

    // regularization lower bound
    double minMu;

    // control limits for clipping
    double uMax[H_MAX][U];
    double uMin[H_MAX][U];

    // maximum amount of iterations to improve trajectory
    int maxIterations;
    // maximum amount of iterations to improve contraints
    int maxLgIterations;
    // minimum allowed cost change before termination
    double minRelCostChange;
    // integration step length
    double dt;
    // time step at which we start optimizing
    int optStart;
    // horizon, optimziation end
    int T;

    // use quadratic terms for computing control laws
    bool useQuadraticTerms;

    // parameters of the optimization problem
    Params params;

    // what integrator to use
    IntegratorType integratorType;

} Optim;

/*
 * symbolic derivatives
 */

@SYMPY_CODE

/**
 * Caculates the size of an integration step.
 *
 * Allows the implementation of dynamically changing step sizes,
 * depending on integrator type and system state.
 */
double calcIntStep(Optim* optim,
                   double (*x)[X],
                   double (*u)[U],
                   int t,
                   double dt,
                   IntegratorType intType) {

    switch (intType) {
        case EULER:
        case HEUN:
        case RK4:
            return dt;
    };

    return 0.0;
}

/**
 * Implements the discrete-time dynamics function with different numerical
 * integrators and the continuous ctDynamics(...) function.
 */
void dynamics(Optim* optim,
              double (*x)[X],
              double (*u)[U],
              int t,
              double dt,
              IntegratorType intType,
              double (*res)[X]) {

    switch (intType) {
        case EULER: {
            double k1[X];
            ctDynamics(optim, *x, *u, t, dt, &k1[0]);

            mul_scalar(k1, k1, dt);

            add(*res, *x, k1);
            break;
        }
        case HEUN: {
            double tmp[X];

            double k1[X];
            ctDynamics(optim, *x, *u, t, dt, &k1[0]);

            mul_scalar(tmp, k1, dt);
            add(tmp, *x, tmp);

            double k2[X];
            ctDynamics(optim, tmp, *u, t, dt, &k2[0]);

            add(tmp, k1, k2);
            mul_scalar(tmp, tmp, dt/2.0);

            add(*res, *x, tmp);
            break;
        }
        case RK4: {
            double tmp[X];

            double k1[X];
            ctDynamics(optim, *x, *u, t, dt, &k1[0]);

            mul_scalar(tmp, k1, dt/2.0);
            add(tmp, *x, tmp);

            double k2[X];
            ctDynamics(optim, tmp, *u, t, dt, &k2[0]);

            mul_scalar(tmp, k2, dt/2.0);
            add(tmp, *x, tmp);

            double k3[X];
            ctDynamics(optim, tmp, *u, t, dt, &k3[0]);

            mul_scalar(tmp, k3, dt);
            add(tmp, *x, tmp);

            double k4[X];
            ctDynamics(optim, tmp, *u, t, dt, &k4[0]);

            zeros(tmp);
            add(tmp, k1, tmp);
            mul_scalar(k1, k2, 2.0);
            add(tmp, k1, tmp);
            mul_scalar(k1, k3, 2.0);
            add(tmp, k1, tmp);
            add(tmp, k4, tmp);
            mul_scalar(tmp, tmp, dt/6.0);

            add(*res, *x, tmp);
            break;
        }
    }
}

double lqrForwardPass (Optim* opt, double scale) {

    double cost = 0.0;
    double c = 0.0;

    copy(opt->next_x[0], opt->x[0]);

    double xdiff[X];

    for (int t = opt->optStart; t < opt->T; t += 1) {

        sub(xdiff, opt->next_x[t], opt->x[t]);
        mul_scalar(opt->next_u[t], opt->k[t], scale);
        add(opt->next_u[t], opt->next_u[t], opt->u[t]);
        mul_add(opt->next_u[t], opt->K[t], xdiff);

        /* 
         * Clamping here is actually necessary, because even though
         * the backward pass implements control contraints, there
         * can still be cases where the constraints are violated
         * by the feedback terms.
         */

        for (size_t d = 0; d < U; ++d) {
            opt->next_u[t][d] = max(opt->uMin[t][d],
                    min(opt->uMax[t][d], opt->next_u[t][d]));
        }

        opt->next_int_step[t] = calcIntStep(
                opt,
                &opt->next_x[t],
                &opt->next_u[t],
                t,
                opt->dt,
                opt->integratorType);

        dynamics(opt,
                 &opt->next_x[t],
                 &opt->next_u[t],
                 t,
                 opt->next_int_step[t],
                 opt->integratorType,
                 &opt->next_x[t+1]);

        costs(opt, opt->next_x[t], opt->next_u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->next_int_step[t], &c);
        cost += c;
    }

    opt->next_int_step[opt->T] = calcIntStep(
            opt,
            &opt->next_x[opt->T],
            &opt->next_u[opt->T],
            opt->T,
            opt->dt,
            opt->integratorType);

    endCosts(opt, opt->next_x[opt->T], opt->T, opt->next_int_step[opt->T], &c);
    cost += c;

    return cost;
}

double lrForwardPass (Optim* opt, double scale) {

    double cost = 0.0;
    double c = 0.0;

    copy(opt->next_x[0], opt->x[0]);

    for (int t = opt->optStart; t < opt->T; ++t) {

        mul_scalar(opt->next_u[t], opt->k[t], scale);
        sub(opt->next_u[t], opt->u[t], opt->next_u[t]);

        opt->next_int_step[t] = calcIntStep(
                opt,
                &opt->next_x[t],
                &opt->next_u[t],
                t,
                opt->dt,
                opt->integratorType);

        dynamics(opt,
                 &opt->next_x[t],
                 &opt->next_u[t],
                 t,
                 opt->next_int_step[t],
                 opt->integratorType,
                 &opt->next_x[t+1]);

        costs(opt, opt->next_x[t], opt->next_u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->next_int_step[t], &c);
        cost += c;
    }

    opt->next_int_step[opt->T] = calcIntStep(
            opt,
            &opt->next_x[opt->T],
            &opt->next_u[opt->T],
            opt->T,
            opt->dt,
            opt->integratorType);

    endCosts(opt, opt->next_x[opt->T], opt->T, opt->next_int_step[opt->T], &c);
    cost += c;

    return cost;
}

bool testImprovement (Optim* opt, double newCosts) {

    if (newCosts < opt->trajCosts && isfinite(newCosts) && newCosts >= 0.0) {

        copy(opt->prev_x, opt->x);
        copy(opt->prev_k, opt->k);
        copy(opt->int_step, opt->next_int_step);
        copy(opt->x, opt->next_x);
        copy(opt->u, opt->next_u);
        opt->trajCosts = newCosts;
        opt->trajectoryChanged = true;
        opt->improved = true;

        return true;
    }

    return false;
}

bool lineSearch (Optim* opt, double (*forward)(Optim*, double)) {

    for (int i = 0; i < 8; ++i) {

        opt->alpha = 1.0 / pow(10, i);

        double newCosts = (*forward)(opt, opt->alpha);

        if (testImprovement(opt, newCosts)) {
            return true;
        }
    }

    return false;
}

void ilqr (Optim* opt) {

    // gradient of value wrt. state
    double Vx[X];
    // gradient of value wrt. state, state
    double Vxx[X][X];
    // gradient of cost-to-go wrt. state
    double Qx[X];
    // gradient of cost-to-go wrt. action
    double Qu[U];
    // hessian of cost-to-go wrt. action, action
    double Quu[U][U];
    // hessian of cost-to-go wrt. state, state
    double Qxx[X][X];
    // hessian of cost-to-go wrt. action, state
    double Qux[U][X];

    for (int s = opt->iterations; s < opt->maxIterations; ++s) {

        opt->iterations = s + 1;

        if (opt->trajectoryChanged) {

            for (int t = 0; t < opt->T; t += 1) {
                stateJacobian(opt, opt->x[t], opt->u[t], t, opt->int_step[t], (double*)&opt->fx[t]);
                actionJacobian(opt, opt->x[t], opt->u[t], t, opt->int_step[t], (double*)&opt->fu[t]);
            }

            for (int t = 0; t < opt->T; t += 1) {
                stateGradient(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lx[t]);
                actionGradient(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lu[t]);
                stateStateHessian(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lxx[t]);
                actionActionHessian(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->luu[t]);
                actionStateHessian(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lux[t]);
            }

            opt->trajectoryChanged = false;
        }

        // initialize value components with gradient of final state
        endGradient(opt, opt->x[opt->T], opt->T, opt->int_step[opt->T], (double*)&Vx);
        endHessian(opt, opt->x[opt->T], opt->T, opt->int_step[opt->T], (double*)&Vxx);

        // backward pass
        
        double fxT[X][X];
        double fuT[U][X];
        double K_T[X][U];
        double tmp_xx[X][X];
        double tmp_xu[X][U];

        for (int t = opt->T-1; t >= 0; --t) {
            transpose(fxT, opt->fx[t]);
            transpose(fuT, opt->fu[t]);

            mul(Qx, fxT, Vx);
            add(Qx, opt->lx[t], Qx);

            mul(Qu, fuT, Vx);
            add(Qu, opt->lu[t], Qu);

            mul(tmp_xx, Vxx, opt->fx[t]);
            mul(tmp_xu, Vxx, opt->fu[t]);

            mul_symm(Qxx, fxT, tmp_xx);
            add(Qxx, opt->lxx[t], Qxx);

            mul_symm(Quu, fuT, tmp_xu);
            add(Quu, opt->luu[t], Quu);

            mul(Qux, fuT, tmp_xx);
            add(Qux, opt->lux[t], Qux);

            solve_action(&Quu, &Qu, &Qux, &opt->k[t], &opt->K[t], opt->mu);

            // apply control limits
            double c[U];
            add(c, opt->u[t], opt->k[t]);

            for (int d = 0; d < U; ++d) {
                if (c[d] > opt->uMax[t][d]) {
                    opt->k[t][d] = opt->uMax[t][d] - opt->u[t][d];
                    zeros(opt->K[t][d]);
                }
                if (c[d] < opt->uMin[t][d]) {
                    opt->k[t][d] = opt->uMin[t][d] - opt->u[t][d];
                    zeros(opt->K[t][d]);
                }
            }

            // Vxx = K.T * Qux + Qux.T * K
            transpose(K_T, opt->K[t]);
            mul(tmp_xx, K_T, Qux);
            transpose(Vxx, tmp_xx);
            add(Vxx, Vxx, tmp_xx);
            // Vxx += K.T * Quu * K
            mul(tmp_xu, K_T, Quu);
            mul_add(Vxx, tmp_xu, opt->K[t]);
            // Vxx += Qxx
            add(Vxx, Vxx, Qxx); 

            // Vx = K.T * Quu * k
            mul(Vx, tmp_xu, opt->k[t]);
            // Vx += K.T * Qu;
            mul_add(Vx, K_T, Qu);
            // Vx += Qux.T * k
            transpose(tmp_xu, Qux);
            mul_add(Vx, tmp_xu, opt->k[t]);
            // Vx += Qx
            add(Vx, Vx, Qx);
        }

        double prevTrajCosts = opt->trajCosts;

        if (lineSearch(opt, lqrForwardPass)) {
            opt->muStep = max(0, opt->muStep - 1);
        } else {
            opt->muStep = min(opt->muStep + 1, 7);
        }

        if (opt->muStep == 0) {
            opt->mu = 0.0;
        } else {
            opt->mu = pow(10.0, opt->muStep - 1);
        }

        double relativeCostChange = fabs(opt->trajCosts - prevTrajCosts) / opt->trajCosts;

        if (relativeCostChange < opt->minRelCostChange) {
            opt->terminationCondition = 2;
            break;
        }
    }
}

void ilr (Optim* opt) {

    // gradient of value wrt. state
    double Vx[X];
    // gradient of cost-to-go wrt. state
    double Qx[X];
    // gradient of cost-to-go wrt. action
    double Qu[U];

    for (int s = opt->iterations; s < opt->maxIterations; ++s) {

        opt->iterations = s + 1;

        if (opt->trajectoryChanged) {

            for (int t = 0; t < opt->T; t += 1) {
                stateJacobian(opt, opt->x[t], opt->u[t], t, opt->int_step[t], (double*)&opt->fx[t]);
                actionJacobian(opt, opt->x[t], opt->u[t], t, opt->int_step[t], (double*)&opt->fu[t]);
            }

            for (int t = 0; t < opt->T; t += 1) {
                stateGradient(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lx[t]);
                actionGradient(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], (double*)&opt->lu[t]);
            }

            opt->trajectoryChanged = false;
        }

        // initialize value components with gradient of final state
        endGradient(opt, opt->x[opt->T], opt->T, opt->int_step[opt->T], (double*)&Vx);

        // backward pass
        
        double fxT[X][X];
        double fuT[U][X];

        for (int t = opt->T-1; t >= 0; --t) {

            transpose(fxT, opt->fx[t]);
            mul(Qx, fxT, Vx);
            add(Qx, opt->lx[t], Qx);

            transpose(fuT, opt->fu[t]);
            mul(Qu, fuT, Vx);
            add(Qu, opt->lu[t], Qu);

            copy(opt->g[t], Qu);
            copy(opt->k[t], Qu);

            copy(Vx, Qx);
        }

        for (int t = opt->T-1; t >= 0; --t) {

            // calculate clipped descent direction
            double c[U];
            sub(c, opt->u[t], opt->g[t]);

            for (int d = 0; d < U; ++d) {
                if (c[d] > opt->uMax[t][d]) {
                    opt->k[t][d] = opt->u[t][d] - opt->uMax[t][d];
                }
                if (c[d] < opt->uMin[t][d]) {
                    opt->k[t][d] = opt->u[t][d] - opt->uMin[t][d];
                }
            }
        }

        double prevTrajCosts = opt->trajCosts;

        lineSearch(opt, lrForwardPass);

        double relativeCostChange = fabs(opt->trajCosts - prevTrajCosts) / opt->trajCosts;

        if (relativeCostChange < opt->minRelCostChange) {
            opt->terminationCondition = 2;
            break;
        }
    }
}

void update (Optim* opt) {

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // do initial trajectory rollout
    
    double c = 0.0;
    opt->trajCosts = 0.0;

    for (int t = opt->optStart; t < opt->T; ++t) {

        opt->int_step[t] = calcIntStep(opt, &opt->next_x[t], &opt->next_u[t], t, opt->dt, opt->integratorType);
        dynamics(opt, &opt->x[t], &opt->u[t], t, opt->int_step[t], opt->integratorType, &opt->x[t+1]);
        costs(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], &c);
        opt->trajCosts += c;
    }

    opt->int_step[opt->T] = calcIntStep(opt, &opt->next_x[opt->T], &opt->next_u[opt->T], opt->T, opt->dt, opt->integratorType);
    endCosts(opt, opt->x[opt->T], opt->T, opt->int_step[opt->T], &c);
    opt->trajCosts += c;

    for (opt->lgIterations = 0; opt->lgIterations < opt->maxLgIterations; ++(opt->lgIterations)) {

        // update lagrange multipliers
        
        double constr[C];

        for (int t = opt->optStart; t < opt->T; ++t) {

            zeros(constr);
            constraints(opt, opt->x[t], opt->u[t], opt->lagrangeMultiplier[t], opt->barrierWeight, t, opt->int_step[t], constr);

            for (int ci = 0; ci < C; ++ci) {
                opt->lagrangeMultiplier[t][ci] = min(
                        opt->lgMultLimit[ci], max(
                            0.0,
                            opt->lagrangeMultiplier[t][ci]
                            + opt->barrierWeight[ci] * constr[ci])
                        );
            }
        }

        opt->trajectoryChanged = true;
        opt->improved = false;
        opt->iterations = 0;

        if (opt->useQuadraticTerms) {
            ilqr(opt);
        } else {
            ilr(opt);
        }
    }

    // set termination condition

    if (opt->iterations == opt->maxIterations) { 
        opt->terminationCondition = 1;
    }

    // store runtime for statistics

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint64_t time_start = start.tv_nsec + start.tv_sec * 1.0e9;
    uint64_t time_end = end.tv_nsec + end.tv_sec * 1.0e9;

    opt->runtime = (double)(time_end - time_start) / 1.0e6;
}

void shift (Optim* opt, int amount) {

    amount = max(0, amount);

    for (int t = opt->optStart; t < opt->T+1; ++t) {
        copy(opt->x[t], opt->x[min(opt->T, t+amount)]);
    }

    for (int t = opt->optStart; t < opt->T; ++t) {
        copy(opt->u[t], opt->u[min(opt->T-1, t+amount)]);
    }

    for (int t = opt->optStart; t < opt->T; ++t) {
        copy(opt->lagrangeMultiplier[t], opt->lagrangeMultiplier[min(opt->T-1, t+amount)]);
    }
}

/*
 * Python binding code, be warned ...
 */

#include <stdarg.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

// this can be used to pass the internal error state to python

#define CHECK_AND_RAISE_ERROR \
    if (errorFlag) { \
        PyErr_SetString(PyExc_AssertionError, errorMsgBuffer); \
        errorFlag = false; \
        return NULL; \
    }

// array shape assertions

bool assertArrayShape(const char* name, PyArrayObject* array, ...) {

    int arrayDims = PyArray_NDIM(array);

    va_list args;
    va_start(args, array);

    int* dims = va_arg(args, int*);
    size_t len = va_arg(args, size_t);

    bool foundShape = false;

    while (dims != NULL) {
        // squeeze dims
        int sq_dims[len];
        size_t sq_len = 0;
        for (size_t i = 0; i < len; ++i) {
            if (i == 0 || dims[i] != 1) {
                sq_dims[sq_len] = dims[i];
                sq_len += 1;
            }
        }
        // check if matches
        if ((int)sq_len == arrayDims) {
            bool ok = true;
            for (size_t i = 0; i < sq_len; ++i) {
                if (sq_dims[i] == -1) {
                    continue;
                }
                ok &= sq_dims[i] == PyArray_DIM(array, i);
            }
            if (ok) { 
                foundShape = true;
                break;
            }
        }

        dims = va_arg(args, int*);
        len = va_arg(args, size_t);
    }

    va_end(args);

    if (!foundShape) {

        // the religious approach: just pray that this is enough
        static char msg[2048];

        int loc = 0;
        loc += sprintf(msg+loc, "Expected \"%s\" with shape ", name);

        va_list args;
        va_start(args, array);

        int* dims = va_arg(args, int*);
        size_t len = va_arg(args, size_t);

        while (dims != NULL) { 
            loc += sprintf(msg+loc, "(");
            for (size_t k = 0; k < len; ++k) { 
                loc += sprintf(msg+loc, "%d", dims[k]);
                if (k != len - 1) { 
                    loc += sprintf(msg+loc, ", ");
                }
            }
            loc += sprintf(msg+loc, ")");

            dims = va_arg(args, int*);
            len = va_arg(args, size_t);

            if (dims != NULL) { 
                loc += sprintf(msg+loc, " | ");
            }
        }

        va_end(args);

        loc += sprintf(msg+loc, ", but found (");
        for (int k = 0; k < arrayDims; ++k) {
            loc += sprintf(msg+loc, "%li", PyArray_DIM(array, k));
            if (k != arrayDims - 1) { 
                loc += sprintf(msg+loc, ", ");
            }
        }
        loc += sprintf(msg+loc, ")\n");
        msg[loc] = 0;

        PyErr_SetString(PyExc_ValueError, msg);

        return false;
    }

    return true;
}

#define assert_shape(A, ...) assertArrayShape(#A, A, __VA_ARGS__, NULL, 0)
#define S(A) A, sizeof(A) / sizeof(int)

/*
 * This atrocious piece of code generates python property getter and setter
 * methods for the given array variable names.
 *
 * The getter method returns the C array as a numpy array, which refers to the
 * memory (but does not own it). So if the Optim() struct get deallocated,
 * still allocated numpy arrays are left pointing into the void.
 *
 * The setter method takes any object, tries to convert it to a numpy array,
 * verifies that the size of the resulting array matches the size of the
 * corresponding C array, and then copies its data to the C array.
 */
#define def_array_get_set(B, N, ...) \
    PyObject* get_##B##N (PyOptim* self, void* closure) { \
        long int shape[] = {__VA_ARGS__}; \
        PyObject* array = PyArray_SimpleNewFromData(\
                sizeof(shape) / sizeof(long int), shape, NPY_DOUBLE, &self->B.N); \
        int arrayDims = PyArray_NDIM((PyArrayObject*)array); \
        PyObject* squeezedArray = array; \
        if (arrayDims > 1) { \
            squeezedArray = PyArray_Squeeze((PyArrayObject*)array); \
            Py_DECREF(array); \
        } \
        return squeezedArray; \
    } \
    int set_##B##N (PyOptim* self, PyObject* value, void* closure) { \
        PyObject* obj = PyArray_FROM_OTF(value, \
                NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST); \
        if (obj == NULL) { \
            return -1; \
        } \
        PyArrayObject* N = (PyArrayObject*)obj; \
        if (PyArray_NDIM(N) > 1) { \
            N = (PyArrayObject*)PyArray_Squeeze(N); \
        } \
        long int shape[] = {__VA_ARGS__}; \
        PyArrayObject* dest = (PyArrayObject*)PyArray_SimpleNewFromData(\
                sizeof(shape) / sizeof(long int), shape, NPY_DOUBLE, &self->B.N); \
        if (PyArray_NDIM(dest) > 1) { \
            dest = (PyArrayObject*)PyArray_Squeeze(dest); \
        } \
        int res = PyArray_MoveInto(dest, N); \
        Py_DECREF(dest); \
        Py_DECREF(N); \
        return res; \
    }

PyObject* dynArrayToPyObj (DynArray* da) {

    if (da->ndim == 0) {
        da->dims[1] = 0;
        return PyArray_SimpleNewFromData(
                1, da->dims, NPY_DOUBLE, da->data);
    } 

    return PyArray_SimpleNewFromData(
            da->ndim, da->dims, NPY_DOUBLE, da->data);
}

int dynArrayFromPyObj (DynArray* da, PyObject* value) {

    PyObject* obj = PyArray_FROM_OTF(value,
            NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (obj == NULL) {
        return -1;
    }
    PyArrayObject* arr = (PyArrayObject*)obj;

    size_t ndim = PyArray_NDIM(arr);

    for (size_t i = 0; i < ndim; ++i) {
        da->dims[i] = PyArray_DIM(arr, i);
    }
    da->ndim = ndim;

    size_t nbytes = PyArray_NBYTES(arr);

    if (da->nbytes != nbytes) {
        if (da->data != NULL) {
            free(da->data);
        }
        da->data = malloc(nbytes);
    } 
    da->nbytes = nbytes;

    memcpy(da->data, PyArray_DATA(arr), nbytes);

    Py_DECREF(arr);
    return 0;
}

/*
 * Version for parameter arrays, basically does the same thing as
 * def_array_get_set(...) but with in DynArray instead of arrays.
 */
#define def_params_array_get_set(N, ...) \
    PyObject* get_params_##N (PyParams* self, void* closure) { \
        return dynArrayToPyObj(&self->params->N); \
    } \
    int set_params_##N (PyParams* self, PyObject* value, void* closure) { \
        return dynArrayFromPyObj(&self->params->N, value); \
    }

// params type 

typedef struct {
    PyObject_HEAD
    bool ref;
    Params* params;
} PyParams;

#define def_params_value_get_set(N) \
    PyObject* get_params_##N (PyParams* self, void* closure) { \
        return PyFloat_FromDouble(self->params->N); \
    } \
    int set_params_##N (PyParams* self, PyObject* value, void* closure) { \
        double v = PyFloat_AsDouble(value); \
        if (PyErr_Occurred() != NULL) { \
            return -1; \
        }\
        self->params->N = v; \
        return 0; \
    }

#define params_value_get_set(N) \
    {#N, (getter)get_params_##N, (setter)set_params_##N, "", NULL}

@PARAM_BIND_CODE

PyObject* pyParamsAlloc (PyTypeObject* self, Py_ssize_t nitems) {

    PyObject* params = PyType_GenericAlloc(self, nitems);
    ((PyParams*)params)->params = (Params*)calloc(1, sizeof(Params));

    return params;
}

void pyParamsDealloc (PyObject* self) {

    PyParams* pp = (PyParams*)self;
    if (pp->ref == false) {
        freeParams(pp->params);
        free(pp->params);
    }

    Py_TYPE(self)->tp_free(self);
}

PyObject* pyParamsDeepCopy (PyObject* self, PyObject* args) {

    PyObject* obj = PyObject_CallObject((PyObject*)Py_TYPE(self), NULL);
    copyParams(((PyParams*)obj)->params, ((PyParams*)self)->params);

    return obj;
}

PyMethodDef pyParamsMethods[] = {
    {"__deepcopy__", (PyCFunction)pyParamsDeepCopy, METH_VARARGS, NULL},
    {"__getstate__", (PyCFunction)pyParamsGetState, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyTypeObject PyParamsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "genopt@CODE_HASH.Params",
    .tp_doc = PyDoc_STR("contains optimizer parameters"),
    .tp_basicsize = sizeof(PyParams),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = pyParamsMethods,
    .tp_getset = pyParamsGetSetMethods,
    .tp_alloc = pyParamsAlloc,
    .tp_dealloc = pyParamsDealloc,
};

// optimizer type

typedef struct {
    PyObject_HEAD
    Optim optim;
} PyOptim;

PyObject* pyUpdate(PyOptim* self, PyObject *Py_UNUSED(ignored)) { 

    Py_BEGIN_ALLOW_THREADS
    update(&self->optim);
    Py_END_ALLOW_THREADS

    CHECK_AND_RAISE_ERROR

    Py_RETURN_NONE;
}

PyObject* pyShift(PyOptim* self, PyObject *args) { 

    int amount;
    if (!PyArg_ParseTuple(args, "i", &amount)) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    shift(&self->optim, amount);
    Py_END_ALLOW_THREADS

    CHECK_AND_RAISE_ERROR

    Py_RETURN_NONE;
}

PyObject* pyDynamics(PyOptim* self, PyObject* args) {

    Optim* opt = &self->optim;

    PyObject* x;
    PyObject* u;
    int t;
    double dt;

    if (!PyArg_ParseTuple(args, "OOid", &x, &u, &t, &dt)) {
        return NULL;
    }
    
    // extract x array

    x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (x == NULL) {
        return NULL;
    }
    PyArrayObject* x_arr = (PyArrayObject*)x;

    int x_shape[] = {X};
    if (!assert_shape(x_arr, S(x_shape))) {
        Py_DECREF(x);
        return NULL;
    }

    // extract u array

    u = PyArray_FROM_OTF(u, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (u == NULL) {
        Py_DECREF(x);
        return NULL;
    }
    PyArrayObject* u_arr = (PyArrayObject*)u;

    int u_shape[] = {U};
    if (!assert_shape(u_arr, S(u_shape))) {
        Py_DECREF(x);
        Py_DECREF(u);
        return NULL;
    }

    double x_in[X];
    memcpy(&x_in, PyArray_DATA(x_arr), sizeof(x_in));

    double u_in[U];
    memcpy(&u_in, PyArray_DATA(u_arr), sizeof(u_in));

    Py_DECREF(x);
    Py_DECREF(u);

    // call dynamics and write to output

    double x_out[X];
    Py_BEGIN_ALLOW_THREADS
    dynamics(opt, &x_in, &u_in, t, dt, opt->integratorType, &x_out);
    Py_END_ALLOW_THREADS

    CHECK_AND_RAISE_ERROR

    long int x_out_dims[1] = {X};
    PyObject* out = PyArray_SimpleNew(1, x_out_dims, NPY_DOUBLE);
    if (out == NULL) {
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)out), x_out, sizeof(x_out));

    return out;
}

PyObject* pyCtDynamics(PyOptim* self, PyObject* args) {

    Optim* opt = &self->optim;

    PyObject* x;
    PyObject* u;
    int t;
    double dt;

    if (!PyArg_ParseTuple(args, "OOid", &x, &u, &t, &dt)) {
        return NULL;
    }

    // extract x array lbub

    x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (x == NULL) {
        return NULL;
    }
    PyArrayObject* x_arr = (PyArrayObject*)x;

    int x_shape[] = {X};
    if (!assert_shape(x_arr, S(x_shape))) {
        Py_DECREF(x);
        return NULL;
    }

    // extract u array

    u = PyArray_FROM_OTF(u, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (u == NULL) {
        Py_DECREF(x);
        return NULL;
    }
    PyArrayObject* u_arr = (PyArrayObject*)u;

    int u_shape[] = {U};
    if (!assert_shape(u_arr, S(u_shape))) {
        Py_DECREF(x);
        Py_DECREF(u);
        return NULL;
    }

    double x_in[X];
    memcpy(&x_in, PyArray_DATA(x_arr), sizeof(x_in));

    double u_in[U];
    memcpy(&u_in, PyArray_DATA(u_arr), sizeof(u_in));

    Py_DECREF(x);
    Py_DECREF(u);

    // call dynamics and write to output

    double x_out[X];
    Py_BEGIN_ALLOW_THREADS
    ctDynamics(opt, x_in, u_in, t, dt, &x_out[0]);
    Py_END_ALLOW_THREADS

    CHECK_AND_RAISE_ERROR

    long int x_out_dims[1] = {X};
    PyObject* out = PyArray_SimpleNew(1, x_out_dims, NPY_DOUBLE);
    if (out == NULL) {
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)out), x_out, sizeof(x_out));

    return out;
}

def_array_get_set(optim, int_step, self->optim.T+1);
def_array_get_set(optim, x, self->optim.T+1, X);
def_array_get_set(optim, u, self->optim.T, U);
def_array_get_set(optim, next_int_step, self->optim.T+1);
def_array_get_set(optim, next_x, self->optim.T+1, X);
def_array_get_set(optim, next_u, self->optim.T, U);
def_array_get_set(optim, prev_x, self->optim.T+1, X);
def_array_get_set(optim, prev_u, self->optim.T, U);
def_array_get_set(optim, prev_k, self->optim.T, U);
def_array_get_set(optim, fx, self->optim.T, X, X);
def_array_get_set(optim, fu, self->optim.T, X, U);
def_array_get_set(optim, lx, self->optim.T, X);
def_array_get_set(optim, lu, self->optim.T, U);
def_array_get_set(optim, lxx, self->optim.T, X, X);
def_array_get_set(optim, luu, self->optim.T, U, U);
def_array_get_set(optim, lux, self->optim.T, U, X);
def_array_get_set(optim, g, self->optim.T, U);
def_array_get_set(optim, k, self->optim.T, U);
def_array_get_set(optim, K, self->optim.T, U, X);
def_array_get_set(optim, lagrangeMultiplier, self->optim.T, C);
def_array_get_set(optim, barrierWeight, C);
def_array_get_set(optim, lgMultLimit, C);
def_array_get_set(optim, uMin, self->optim.T, U);
def_array_get_set(optim, uMax, self->optim.T, U);

PyObject* get_params (PyOptim* self, void* closure) {

    PyObject* obj = PyObject_CallObject((PyObject*)&PyParamsType, NULL);
    PyParams* pp = (PyParams*)obj;

    free(pp->params);
    pp->ref = true;
    pp->params = &(self->optim.params);

    return obj;
}

int set_params (PyOptim* self, PyObject* value, void* closure) {
    
    copyParams(&self->optim.params, ((PyParams*)value)->params);
    return 0;
}

PyObject* get_integrator (PyOptim* self, void* closure) {

    return PyLong_FromLong((long)self->optim.integratorType);
}

int set_integrator (PyOptim* self, PyObject* value, void* closure) {
    
    self->optim.integratorType = (IntegratorType)PyLong_AsLong(value);
    if (PyErr_Occurred() != NULL) {
        return -1;
    }
    return 0;
}

PyObject* get_EULER (PyOptim* self, void* closure) {
    return PyLong_FromLong(0);
}
PyObject* get_HEUN (PyOptim* self, void* closure) {
    return PyLong_FromLong(1);
}
PyObject* get_RK4 (PyOptim* self, void* closure) {
    return PyLong_FromLong(2);
}

PyObject* get_horizon (PyOptim* self, void* closure) {

    return PyLong_FromLong((long)self->optim.T);
}

int set_horizon (PyOptim* self, PyObject* value, void* closure) {
    
    self->optim.T = PyLong_AsLong(value);
    if (PyErr_Occurred() != NULL) {
        return -1;
    }
    self->optim.T = min(H_MAX-1, max(1, self->optim.T));
    return 0;
}

PyObject* getOptimSlots (PyObject* self)  {
    return Py_BuildValue(
            "[sssssssssssssssssssssssssssssssssssssssss]",
            "step",
            "opt_start",
            "horizon",
            "int_step",
            "x",
            "u",
            "next_int_step",
            "next_x",
            "next_u",
            "prev_x",
            "prev_u",
            "prev_k",
            "fx",
            "fu",
            "lx",
            "lu",
            "lxx",
            "luu",
            "lux",
            "g",
            "k",
            "K",
            "lagrange_multiplier",
            "barrier_weight", 
            "lg_mult_limit",
            "u_max", 
            "u_min",
            "params", 
            "traj_costs", 
            "iterations",
            "lg_iterations",
            "runtime",
            "alpha",
            "mu",
            "mu_step",
            "trajectory_changed",
            "improved",
            "termination_condition",
            "use_quadratic_terms",
            "max_iterations",
            "max_lg_iterations",
            "min_rel_cost_change"
        );
}

PyObject* pyOptimDeepCopy (PyObject* self, PyObject* args) {

    PyObject* obj = PyObject_CallObject((PyObject*)Py_TYPE(self), NULL);

    PyOptim* oo = (PyOptim*)obj;
    PyOptim* so = (PyOptim*)self;

    oo->optim = so->optim;

    memset(&oo->optim.params, 0, sizeof(Params));
    copyParams(&oo->optim.params, &so->optim.params);

    return obj;
}

PyObject* pyOptimGetState (PyObject* self) {

    PyOptim* po = (PyOptim*)self;
    Optim* o = &po->optim;

    PyObject* pp = PyObject_GetAttrString(self, "params");

    return Py_BuildValue(
            "{s:O,s:O,s:i,s:i,s:O,s:O,s:i,s:i,s:d,s:b,s:O}",
            "u_min", get_optimuMin(po, NULL),
            "u_max", get_optimuMax(po, NULL),
            "horizon", o->T,
            "opt_start", o->optStart,
            "barrier_weight", get_optimbarrierWeight(po, NULL),
            "lg_mult_limit", get_optimlgMultLimit(po, NULL),
            "max_iterations", o->maxIterations,
            "max_lg_iterations", o->maxLgIterations,
            "step", o->dt,
            "use_quadratic_terms", o->useQuadraticTerms,
            "params", pyParamsGetState(pp)
            );
}

#define array_get_set(M, B, N) {M, (getter)get_##B##N, (setter)set_##B##N, "", NULL}

PyGetSetDef pyGetSetMethods[] = {
    array_get_set("int_step", optim, int_step),
    array_get_set("x", optim, x),
    array_get_set("u", optim, u),
    array_get_set("next_int_step", optim, next_int_step),
    array_get_set("next_x", optim, next_x),
    array_get_set("next_u", optim, next_u),
    array_get_set("prev_x", optim, prev_x),
    array_get_set("prev_u", optim, prev_u),
    array_get_set("prev_k", optim, prev_k),
    array_get_set("fx", optim, fx),
    array_get_set("fu", optim, fu),
    array_get_set("lx", optim, lx),
    array_get_set("lu", optim, lu),
    array_get_set("lxx", optim, lxx),
    array_get_set("luu", optim, luu),
    array_get_set("lux", optim, lux),
    array_get_set("g", optim, g),
    array_get_set("k", optim, k),
    array_get_set("K", optim, K),
    array_get_set("lagrange_multiplier", optim, lagrangeMultiplier),
    array_get_set("barrier_weight", optim, barrierWeight),
    array_get_set("lg_mult_limit", optim, lgMultLimit),
    array_get_set("u_max", optim, uMax),
    array_get_set("u_min", optim, uMin),
    {"params", (getter)get_params, (setter)set_params, "", NULL},
    {"integrator_type", (getter)get_integrator, (setter)set_integrator, "", NULL},
    {"horizon", (getter)get_horizon, (setter)set_horizon, "", NULL},
    {"T", (getter)get_horizon, (setter)set_horizon, "", NULL},
    {"EULER", (getter)get_EULER, NULL, "", NULL},
    {"HEUN", (getter)get_HEUN, NULL, "", NULL},
    {"RK4", (getter)get_RK4, NULL, "", NULL},
    {"__slots__", (getter)getOptimSlots, NULL, "", NULL},
    {NULL}
};

#define optoffset(M) offsetof(PyOptim, optim) + offsetof(Optim, M)

PyMemberDef pyOptimMembers[] = {
    {"traj_costs", T_DOUBLE, optoffset(trajCosts), 0, ""},
    {"iterations", T_INT, optoffset(iterations), 0, ""},
    {"lg_iterations", T_INT, optoffset(lgIterations), 0, ""},
    {"runtime", T_DOUBLE, optoffset(runtime), 0, ""},
    {"elapsed_update_time", T_DOUBLE, optoffset(runtime), 0, ""},
    {"alpha", T_DOUBLE, optoffset(alpha), 0, ""},
    {"mu", T_DOUBLE, optoffset(mu), 0, ""},
    {"mu_step", T_INT, optoffset(muStep), 0, ""},
    {"trajectory_changed", T_BOOL, optoffset(trajectoryChanged), 0, ""},
    {"improved", T_BOOL, optoffset(improved ), 0, ""},
    {"termination_condition", T_INT, optoffset(terminationCondition), 0, ""},
    {"use_quadratic_terms", T_BOOL, optoffset(useQuadraticTerms), 0, ""},
    {"dt", T_DOUBLE, optoffset(dt), 0, ""},
    {"step", T_DOUBLE, optoffset(dt), 0, ""},
    {"max_iterations", T_INT, optoffset(maxIterations), 0, ""},
    {"max_lg_iterations", T_INT, optoffset(maxLgIterations), 0, ""},
    {"opt_start", T_INT, optoffset(optStart), 0, ""},
    {"min_rel_cost_change", T_DOUBLE, optoffset(minRelCostChange), 0, ""},
    {NULL}
};

PyMethodDef pyOptimMethods[] = {
    {"update", (PyCFunction)pyUpdate, METH_NOARGS, NULL},
    {"shift", (PyCFunction)pyShift, METH_VARARGS, NULL},
    {"dynamics", (PyCFunction)pyDynamics, METH_VARARGS, NULL},
    {"ct_dynamics", (PyCFunction)pyCtDynamics, METH_VARARGS, NULL},
    {"__getstate__", (PyCFunction)pyOptimGetState, METH_NOARGS, NULL},
    {"__deepcopy__", (PyCFunction)pyOptimDeepCopy, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyObject* pyOptimAlloc (PyTypeObject* self, Py_ssize_t nitems) {

    PyObject* optim = PyType_GenericAlloc(self, nitems);
    Optim* opt = &((PyOptim*)optim)->optim;

    opt->dt = 0.05;
    opt->T = 20;
    opt->minRelCostChange = 1e-6;
    opt->maxIterations = 5;
    opt->maxLgIterations = 1;
    opt->useQuadraticTerms = true;
    for (int i = 0; i < C; ++i) {
        opt->lgMultLimit[i] = INFINITY;
    }
    for (int i = 0; i < C; ++i) {
        opt->barrierWeight[i] = 1.0;
    }
    for (int t = 0; t < opt->T; ++t) {
        for (int i = 0; i < U; ++i) {
            opt->uMax[t][i] = INFINITY;
        }
        for (int i = 0; i < U; ++i) {
            opt->uMin[t][i] = -INFINITY;
        }
    }

    return optim;
}

void pyOptimDealloc (PyObject* self) {

    PyOptim* opt = (PyOptim*)self;
    freeParams(&(opt->optim.params));

    Py_TYPE(self)->tp_free(self);
}

PyTypeObject PyOptimType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "genopt@CODE_HASH.Optim",
    .tp_doc = PyDoc_STR("solves dynamic optimization problems"),
    .tp_basicsize = sizeof(PyOptim),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = pyOptimMethods,
    .tp_members = pyOptimMembers,
    .tp_getset = pyGetSetMethods,
    .tp_alloc = pyOptimAlloc,
    .tp_dealloc = pyOptimDealloc,
};

PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};

PyModuleDef genoptModule = {
    PyModuleDef_HEAD_INIT,
    "genopt@CODE_HASH", 
    /* module documentation, may be NULL */
    "iterative solvers for dynamic optimization problems",
    /* size of per-interpreter state of the module,
     * or -1 if the module keeps state in global variables. */
    -1,       
    methods
};

PyMODINIT_FUNC PyInit_genopt@CODE_HASH()
{
    PyObject *m;

    import_array();

    if (PyType_Ready(&PyOptimType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&PyParamsType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&genoptModule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&PyOptimType);
    if (PyModule_AddObject(m, "Optim", (PyObject*)&PyOptimType) < 0) {
        Py_DECREF(&PyOptimType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&PyParamsType);
    if (PyModule_AddObject(m, "Params", (PyObject*)&PyParamsType) < 0) {
        Py_DECREF(&PyParamsType);
        Py_DECREF(&PyOptimType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
