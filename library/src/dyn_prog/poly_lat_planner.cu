#include "tplcpp/dyn_prog/poly_lat_planner.cuh"

namespace poly_lat_cu {

__constant__ PolyLatPlannerParams params;

__device__ size_t best_idx_l_dst = 0;
__device__ size_t best_idx_s_dst = 0;

__global__ void evalLocalPathCosts(const PolyLatTrajPoint start,
                                   const EnvironmentGpu env,
                                   ArrTexSurf<float4> path_nodes) {

    const size_t idx_l_dst = blockIdx.x;
    const size_t idx_s_dst = blockIdx.y;
    const size_t idx_s = threadIdx.x;

    if (idx_l_dst >= params.l_dst_steps 
            || idx_s_dst >= params.s_dst_steps 
            || idx_s >= params.s_steps) { 
        return;
    }

    const float l_dst = params.l_dst_min + params._l_dst_step_size * idx_l_dst;
    const float s_dst = params.s_dst_min + params._s_dst_step_size * idx_s_dst;
    const float s = start.s + params.s_min + params._s_step_size * idx_s;

    PolyQuintic poly(start.s, start.l, start.dl, start.ddl,
                     start.s + s_dst, l_dst, 0.0, 0.0);

    PolyLatTrajPoint p;
    p.s = s;

    if (s >= start.s + s_dst) { 
        p.l = l_dst;
        p.dl = 0.0;
        p.ddl = 0.0;
        p.dddl = 0.0;
    } else { 
        p.l = poly.f(p.s);
        p.dl = poly.df(p.s);
        p.ddl = poly.ddf(p.s);
        p.dddl = poly.dddf(p.s);
    }

    const RefLinePoint rp = env.ref_line.lerp(p.s);

    // transformation to cartesian coordinates

    const float heading_frenet = std::atan(p.dl);

    p.x = rp.x - sin(rp.heading) * p.l;
    p.y = rp.y + cos(rp.heading) * p.l;
    p.heading = heading_frenet + rp.heading;
    p.k = (p.ddl / (p.dl*p.dl + 1.0) + rp.k)
        * std::cos(heading_frenet) / (1.0 - p.l * rp.k);

    float k_abs_path = max(abs(p.k), abs(rp.k));
    p.v = rp.v_max;
    if (k_abs_path > 1e-6) {
        p.v = min(p.v, sqrt(params.a_lat_abs_max / k_abs_path));
    }

    // evaluation of local constraints 
    
    float constr = 0.0;

    constr += max(0.0, sq(min(rp.v_max, start.v)) * fabsf(p.k) - params.a_lat_abs_max);

    if (s <= start.s + s_dst) {
        constr += max(0.0, abs(p.k) - params.k_abs_max);
        constr += max(0.0, p.l - (rp.d_left - params.width_veh * 0.5 * sqrt(2.0)));
        constr += max(0.0, (-rp.d_right + params.width_veh * 0.5 * sqrt(2.0)) - p.l);
    }

    // evaluation of local cost

    float cost = 0.0;

    cost += params.w_dl * sq(p.dl);
    cost += params.w_ddl * sq(p.ddl);
    cost += params.w_dddl * sq(p.dddl);

    if (fabsf(p.k) > fabsf(rp.k)) {
        cost += params.w_k * sq(p.k);
    }

    cost += 10e6 * constr;

    // write back result

    float4 node;
    node.x = p.l;
    node.y = p.v;
    node.z = 0.0;
    node.w = cost;

    path_nodes.set(idx_l_dst, idx_s_dst, idx_s, node);
}

__global__ void calcPathTimes(const PolyLatTrajPoint start,
                              const EnvironmentGpu env,
                              ArrTexSurf<float4> path_nodes) {

    const size_t idx_l_dst = blockIdx.x;
    const size_t idx_s_dst = threadIdx.x;

    if (idx_l_dst >= params.l_dst_steps || idx_s_dst >= params.s_dst_steps) { 
        return;
    }

    float dist = 0.0;
    float t = 0.0;

    float x_prev = 0.0;
    float y_prev = 0.0;

    for (size_t idx_s = 0; idx_s < params.s_steps; ++idx_s) {

        const float s = start.s + params.s_min + params._s_step_size * idx_s;

        const RefLinePoint rp = env.ref_line.lerp(s);
        
        float4 node = path_nodes.get_cached(idx_l_dst, idx_s_dst, idx_s);

        const float l = node.x;
        const float x = rp.x - sin(rp.heading) * l;
        const float y = rp.y + cos(rp.heading) * l;
        const float diff_x = x - x_prev;
        const float diff_y = y - y_prev;

        if (idx_s > 0) { 
            const float d = sqrt(diff_x*diff_x + diff_y*diff_y);
            dist += d;
            // TODO: replace with better velocity estimate
            t += d / max(1.0, node.y);
        }

        x_prev = x;
        y_prev = y;

        node.z = t;

        path_nodes.set(idx_l_dst, idx_s_dst, idx_s, node);
    }
}

__global__ void checkCollisions(const PolyLatTrajPoint start,
                                EnvironmentGpu env,
                                ArrTexSurf<float4> path_nodes) {

    const size_t idx_l_dst = blockIdx.x;
    const size_t idx_s_dst = blockIdx.y;
    const size_t idx_s = threadIdx.x;

    if (idx_l_dst >= params.l_dst_steps 
            || idx_s_dst >= params.s_dst_steps 
            || idx_s >= params.s_steps) { 
        return;
    }

    const float s = start.s + params.s_min + params._s_step_size * idx_s;

    float4 node = path_nodes.get_cached(idx_l_dst, idx_s_dst, idx_s);

    const float l = node.x;
    const float t = node.z;

    float dist_sem = 0.0;
    for (float t_sweep = -1.0; t_sweep < 1.1; t_sweep += 1.0) {
        dist_sem = fmaxf(dist_sem, env.interpDistField(t + t_sweep, s, l));
        dist_sem = fmaxf(dist_sem, env.interpDistField(t + t_sweep, s, l + 0.25));
        dist_sem = fmaxf(dist_sem, env.interpDistField(t + t_sweep, s, l - 0.25));
    }
    bool collision = dist_sem > 0.0;

    if (collision && t < 8.0 && s > params.length_veh) {
        node.z = s;
    } else {
        node.z = 10000.0;
    }

    path_nodes.set(idx_l_dst, idx_s_dst, idx_s, node);
}

__global__ void evalPathCosts(const PolyLatTrajPoint start,
                              EnvironmentGpu env,
                              ArrTexSurf<float4> path_nodes) {

    const size_t idx_l_dst = blockIdx.x;
    const size_t idx_s_dst = threadIdx.x;

    if (idx_l_dst >= params.l_dst_steps || idx_s_dst >= params.s_dst_steps) { 
        return;
    }

    const float l_dst = params.l_dst_min + params._l_dst_step_size * idx_l_dst;
    const float s_dst = params.s_dst_min + params._s_dst_step_size * idx_s_dst;

    float traj_cost = 0.0;
    float collision_dist = 1000.0;

    for (size_t idx_s = 0; idx_s < params.s_steps; ++idx_s) {
        float4 node = path_nodes.get_cached(idx_l_dst, idx_s_dst, idx_s);
        traj_cost += node.w;
        collision_dist = min(collision_dist, node.z);
    }

    if (l_dst < -0.1) {
        traj_cost += params.w_right;
    }

    float diff_l = l_dst - params.l_trg;
    traj_cost += params.w_l * diff_l*diff_l;
    traj_cost += params.w_len * fabsf(s_dst);

    float4 node;
    node.z = collision_dist;
    node.w = traj_cost;

    // first node holds costs for entire path
    path_nodes.set(idx_l_dst, idx_s_dst, 0, node);
}

__global__ void selectPath(ArrTexSurf<float4> path_nodes) {

    size_t min_idx_l_dst = params.l_dst_steps / 2 + 1;
    size_t min_idx_s_dst = params.s_dst_steps - 1;
    float4 center_node = path_nodes.get_cached(min_idx_l_dst, min_idx_s_dst, 0);
    float max_collision_dist = center_node.z;
    float min_cost = INFINITY;

    // compute the maximum collision dist
    for (size_t idx_l_dst = 0; idx_l_dst < params.l_dst_steps; ++idx_l_dst) {
        for (size_t idx_s_dst = 0; idx_s_dst < params.s_dst_steps; ++idx_s_dst) {
            float4 node = path_nodes.get_cached(idx_l_dst, idx_s_dst, 0);
            float collision_dist = node.z;
            if (node.w >= 1e6) {
                continue;
            }
            if (collision_dist > max_collision_dist + params.length_veh) { 
                max_collision_dist = node.z;
            }
        }
    }

    for (size_t idx_l_dst = 0; idx_l_dst < params.l_dst_steps; ++idx_l_dst) {
        for (size_t idx_s_dst = 0; idx_s_dst < params.s_dst_steps; ++idx_s_dst) {
            float4 node = path_nodes.get_cached(idx_l_dst, idx_s_dst, 0);
            float collision_dist = node.z;
            float path_cost = node.w;
            if (fabs(collision_dist - max_collision_dist) > 1.0) {
                // skip if not close to maximum collision dist
                continue;
            }
            if (path_cost < min_cost) { 
                min_cost = path_cost;
                max_collision_dist = node.z;
                min_idx_l_dst = idx_l_dst;
                min_idx_s_dst = idx_s_dst;
            }
        }
    }

    best_idx_l_dst = min_idx_l_dst;
    best_idx_s_dst = min_idx_s_dst;
}

};

PolyLatTrajPoint PolyLatTraj::lerp(double distance) {

    InterpVars i(points, &PolyLatTrajPoint::distance, distance);

    double ai = 1.0f - i.a;

    PolyLatTrajPoint res;
    res.t = points[i.i_prev].t * ai + points[i.i_next].t * i.a;
    res.l = points[i.i_prev].l * ai + points[i.i_next].l * i.a;
    res.dl = points[i.i_prev].dl * ai + points[i.i_next].dl * i.a;
    res.ddl = points[i.i_prev].ddl * ai + points[i.i_next].ddl * i.a;
    res.dddl = points[i.i_prev].dddl * ai + points[i.i_next].dddl * i.a;
    res.s = points[i.i_prev].s * ai + points[i.i_next].s * i.a;
    res.v = points[i.i_prev].v * ai + points[i.i_next].v * i.a;

    res.x = points[i.i_prev].x * (double)ai + points[i.i_next].x * (double)i.a;
    res.y = points[i.i_prev].y * (double)ai + points[i.i_next].y * (double)i.a;
    res.heading = points[i.i_prev].heading
                + shortAngleDist(points[i.i_prev].heading, points[i.i_next].heading) * i.a;
    res.distance = points[i.i_prev].distance * ai + points[i.i_next].distance * i.a;
    res.k = points[i.i_prev].k * ai + points[i.i_next].k * i.a;

    return res;
}

void PolyLatTraj::insertAfterStation(double s, PolyLatTraj& o) {

    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].s >= s) {
            points.resize(i);
            break;
        }
    }

    points.reserve(points.size() + o.points.size());
    points.insert(points.end(), o.points.begin(), o.points.end());

    updateTimeDistCurv();
}

void PolyLatTraj::updateTimeDistCurv() { 

    double dist = 0.0;
    double t = 0.0;

    for (size_t i = 1; i < points.size()+1; ++i) {

        PolyLatTrajPoint& p = points[i-1];
        const PolyLatTrajPoint& p_next = points[i];

        double dx = p_next.x - p.x;
        double dy = p_next.y - p.y;
        double d = sqrt(dx*dx + dy*dy);

        p.distance = dist;
        p.t = t;
        p.k = shortAngleDist(p.heading, p_next.heading) / d;

        dist += d;
        t += d / p.v;
    }
}

PolyLatPlanner::~PolyLatPlanner() {
    clearBuffers();
}

void PolyLatPlanner::clearBuffers() { 
    path_nodes.release();
}

void PolyLatPlanner::reinitBuffers(PolyLatPlannerParams& ps) {

    ps.updateStepSizes();

    if (ps == params) { 
        return;
    }

    params = ps;

    // copy params to gpu
    {
        cudaError_t err = cudaMemcpyToSymbol(poly_lat_cu::params,
                                             &params,
                                             sizeof(PolyLatPlannerParams));
        checkCudaError(err, "copying poly lat planner params failed");
    }

    clearBuffers();

    path_nodes.reinit(params.l_dst_steps, params.s_dst_steps, params.s_steps);
}

PolyLatTraj PolyLatPlanner::update(
        PolyLatTrajPoint& start, 
        DynProgEnvironment& env) {

    {
        dim3 gridSize(params.l_dst_steps, params.s_dst_steps, 1);
        dim3 blockSize(params.s_steps, 1, 1);

        poly_lat_cu::evalLocalPathCosts<<<gridSize, blockSize>>>(
                start, env.envGpu, path_nodes);

        checkCudaError(cudaGetLastError(), "evaluating local path costs failed");
    }

    {
        dim3 gridSize(params.l_dst_steps, 1, 1);
        dim3 blockSize(params.s_dst_steps, 1, 1);

        poly_lat_cu::calcPathTimes<<<gridSize, blockSize>>>(
                start, env.envGpu, path_nodes);

        checkCudaError(cudaGetLastError(), "calculating path times failed");
    }

    {
        dim3 gridSize(params.l_dst_steps, params.s_dst_steps, 1);
        dim3 blockSize(params.s_steps, 1, 1);

        poly_lat_cu::checkCollisions<<<gridSize, blockSize>>>(
                start, env.envGpu, path_nodes);

        checkCudaError(cudaGetLastError(), "checking path collisions failed");
    }

    {
        dim3 gridSize(params.l_dst_steps, 1, 1);
        dim3 blockSize(params.s_dst_steps, 1, 1);

        poly_lat_cu::evalPathCosts<<<gridSize, blockSize>>>(
                start, env.envGpu, path_nodes);

        checkCudaError(cudaGetLastError(), "evaluating path costs failed");
    }

    {
        dim3 gridSize(1, 1, 1);
        dim3 blockSize(1, 1, 1);

        poly_lat_cu::selectPath<<<gridSize, blockSize>>>(path_nodes);

        checkCudaError(cudaGetLastError(), "selecting path failed");
    }

    cudaDeviceSynchronize();

    // copy indices of best path

    size_t best_idx_l_dst = 0;
    size_t best_idx_s_dst = 0;
    {
        cudaError_t err;

        err = cudaMemcpyFromSymbol(&best_idx_l_dst,
                                   poly_lat_cu::best_idx_l_dst,
                                   sizeof(size_t),
                                   0,
                                   cudaMemcpyDeviceToHost);
        checkCudaError(err, "copying best_idx_l_dst failed");

        err = cudaMemcpyFromSymbol(&best_idx_s_dst,
                                   poly_lat_cu::best_idx_s_dst,
                                   sizeof(size_t),
                                   0,
                                   cudaMemcpyDeviceToHost);
        checkCudaError(err, "copying best_idx_s_dst failed");
    }

    cudaDeviceSynchronize();

    // compute final path

    const float l_dst = params.l_dst_min + params._l_dst_step_size * best_idx_l_dst;
    const float s_dst = params.s_dst_min + params._s_dst_step_size * best_idx_s_dst;

    PolyLatTraj traj;
    traj.poly = PolyQuintic(
            start.s, start.l, start.dl, start.ddl,
            start.s + s_dst, l_dst, 0.0, 0.0);

    for (int idx_s = 0; idx_s < params.s_steps; ++idx_s) {

        PolyLatTrajPoint& p = traj.points.emplace_back();
        p.s = start.s + params.s_min + params._s_step_size * idx_s;

        if (p.s >= start.s + s_dst) { 
            p.l = l_dst;
            p.dl = 0.0;
            p.ddl = 0.0;
            p.dddl = 0.0;
        } else { 
            p.l = traj.poly.f(p.s);
            p.dl = traj.poly.df(p.s);
            p.ddl = traj.poly.ddf(p.s);
            p.dddl = traj.poly.dddf(p.s);
        }

        const RefLinePoint rp = env.refLine.lerp(p.s);

        // transformation to cartesian coordinates

        const float heading_frenet = std::atan(p.dl);

        p.x = env.refLine.x_offset + rp.x - sin(rp.heading) * p.l;
        p.y = env.refLine.y_offset + rp.y + cos(rp.heading) * p.l;
        p.heading = heading_frenet + rp.heading;
        p.v = rp.v_max;
    }

    traj.updateTimeDistCurv();

    return traj;
}
