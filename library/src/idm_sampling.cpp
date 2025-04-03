#include "tplcpp/idm_sampling.hpp"

#include <iostream>
#include <vector>

IdmSamplingState IdmSamplingTraj::lerp(double t) {

    InterpVars i(states, &IdmSamplingState::t, t);

    double ai = 1.0f - i.a;

    IdmSamplingState res;
    res.t = states[i.i_prev].t * ai + states[i.i_next].t * i.a;
    res.x = states[i.i_prev].x * ai + states[i.i_next].x * i.a;
    res.y = states[i.i_prev].y * ai + states[i.i_next].y * i.a;
    res.heading = normalizeAngle(states[i.i_prev].heading 
            + shortAngleDist(states[i.i_prev].heading, states[i.i_next].heading) * i.a);
    res.v = states[i.i_prev].v * ai + states[i.i_next].v * i.a;
    res.s = states[i.i_prev].s * ai + states[i.i_next].s * i.a;
    res.l = states[i.i_prev].l * ai + states[i.i_next].l * i.a;
    res.steer_angle = states[i.i_prev].steer_angle * ai + states[i.i_next].steer_angle * i.a;

    int a_round = (int)std::round(i.a * 1.0e4) / 1.0e4;
    if (a_round == 0) {
        res.a = states[i.i_prev].a; 
        res.steer_angle = states[i.i_prev].steer_angle;
    } else {
        res.a = states[i.i_next].a; 
        res.steer_angle = states[i.i_next].steer_angle;
    }

    return res;
}

IdmSamplingRefState IdmSamplingTraj::lerpRef(double t) {

    InterpVars i(refStates, &IdmSamplingRefState::t, t);

    double ai = 1.0 - i.a;

    IdmSamplingRefState res;
    res.t = refStates[i.i_prev].t * ai + refStates[i.i_next].t * i.a;
    res.x = refStates[i.i_prev].x * ai + refStates[i.i_next].x * i.a;
    res.y = refStates[i.i_prev].y * ai + refStates[i.i_next].y * i.a;
    res.a = refStates[i.i_prev].a * ai + refStates[i.i_next].a * i.a;
    res.heading = normalizeAngle(refStates[i.i_prev].heading
            + shortAngleDist(refStates[i.i_prev].heading, refStates[i.i_next].heading) * i.a);
    res.v = refStates[i.i_prev].v * ai + refStates[i.i_next].v * i.a;
    res.s = refStates[i.i_prev].s * ai + refStates[i.i_next].s * i.a;
    res.l = refStates[i.i_prev].l * ai + refStates[i.i_next].l * i.a;

    int a_round = (int)std::round(i.a * 1.0e4) / 1.0e4;
    if (a_round == 0) {
        res.a = refStates[i.i_prev].a;
    } else {
        res.a = refStates[i.i_next].a;
    }

    return res;
}

IdmSamplingPlanner::IdmSamplingPlanner() { }

void IdmSamplingPlanner::reset() {

    prevTraj = IdmSamplingTraj();
    deadTimeTraj = IdmSamplingTraj();
}

void IdmSamplingPlanner::insertDynObj(std::vector<PredictionPoint> predictions,
                                      std::vector<vec<2>> hull,
                                      bool on_local_map) {

    DynObj& dobj = dynObjs.emplace_back();
    dobj.predictions = predictions;
    dobj.hull = hull;
    dobj.on_local_map = on_local_map;

    // transform hull to x = 0.0, y = 0.0, heading = 0.0 
    PredictionPoint pp0 = dobj.predictions[0];
    mat<2, 2> R;
    R(0, 0) = cos(-pp0.heading);
    R(0, 1) = -sin(-pp0.heading);
    R(1, 0) = -R(0, 1);
    R(1, 1) = R(0, 0);
    for (size_t i = 0; i < dobj.hull.size(); ++i) {
        vec<2> hp = dobj.hull[i];
        hp[0] -= pp0.x;
        hp[1] -= pp0.y;
        dobj.hull[i] = R * hp;
    }

    // compute hull radius
    for (size_t i = 0; i < dobj.hull.size(); ++i) {
        dobj.radiusHull = std::max(dobj.radiusHull, dobj.hull[i].norm());
    }
}

PredictionPoint DynObj::lerp(double t) {

    InterpVars i(predictions, &PredictionPoint::t, t);

    float ai = 1.0f - i.a;

    PredictionPoint res;
    res.t = predictions[i.i_prev].t * ai + predictions[i.i_next].t * i.a;
    res.x = predictions[i.i_prev].x * ai + predictions[i.i_next].x * i.a;
    res.y = predictions[i.i_prev].y * ai + predictions[i.i_next].y * i.a;
    res.heading = predictions[i.i_prev].heading 
        + shortAngleDist(predictions[i.i_prev].heading,
                         predictions[i.i_next].heading) * i.a;
    res.v = predictions[i.i_prev].v * ai + predictions[i.i_next].v * i.a;

    return res;
}

PredictionPoint DynObj::lerpStation(double s) {

    InterpVars i(dists, s);

    float ai = 1.0f - i.a;

    PredictionPoint res;
    res.t = predictions[i.i_prev].t * ai + predictions[i.i_next].t * i.a;
    res.x = predictions[i.i_prev].x * ai + predictions[i.i_next].x * i.a;
    res.y = predictions[i.i_prev].y * ai + predictions[i.i_next].y * i.a;
    res.heading = predictions[i.i_prev].heading 
        + shortAngleDist(predictions[i.i_prev].heading,
                         predictions[i.i_next].heading) * i.a;
    res.v = predictions[i.i_prev].v * ai + predictions[i.i_next].v * i.a;

    return res;
}

vec<4> DynObj::lerpHullProj(double t) {

    InterpVars i(predictions, &PredictionPoint::t, t);

    vec<4> res = hullProjs[i.i_prev] * (1.0f - i.a)
               + hullProjs[i.i_next] * i.a;

    return res;
}

std::vector<vec<2>> DynObj::lerpHullPred(double t) {

    InterpVars i(predictions, &PredictionPoint::t, t);

    float ia = 1.0f - i.a;

    std::vector<vec<2>> res(hull.size());
    for (size_t j = 0; j < hull.size(); ++j) {
        res[j] = hullPreds[i.i_prev][j] * ia
                 + hullPreds[i.i_next][j] * i.a;
    }

    return res;
}

void DynObj::updatePredGeometry(std::vector<vec<2>>& linestrip) {

    // transform hulls to predicted points

    hullPreds.resize(predictions.size());

    for (size_t i = 0; i < predictions.size(); ++i) {

        PredictionPoint pp = predictions[i];

        mat<2, 2> R;
        R(0, 0) = cos(pp.heading);
        R(0, 1) = -sin(pp.heading);
        R(1, 0) = -R(0, 1);
        R(1, 1) = R(0, 0);

        std::vector<vec<2>>& h = hullPreds[i];
        h.resize(hull.size());

        for (size_t j = 0; j < hull.size(); ++j) {
            h[j] = R * hull[j];
            h[j][0] += pp.x;
            h[j][1] += pp.y;
        }
    }
    
    // extract path of prediction

    path.resize(predictions.size());
    for (size_t k = 0; k < predictions.size(); ++k) {
        path[k] = vec<2>(predictions[k].x, predictions[k].y);
    }
    dists.resize(predictions.size());
    for (size_t k = 1; k < predictions.size(); ++k) {
        dists[k] = dists[k-1] + (path[k] - path[k-1]).norm();
    }

    // combine hull with next hull
    
    for (size_t idx_t = 0; idx_t < hullPreds.size()-1; ++idx_t) {

        std::vector<vec<2>> h = hullPreds[idx_t];

        h.reserve(h.size() + hullPreds[idx_t+1].size());
        h.insert(h.end(),
                 hullPreds[idx_t+1].begin(),
                 hullPreds[idx_t+1].end());

        hullPreds[idx_t] = convexHull(h);
    }

    // project to linestrip
    
    hullProjs.resize(predictions.size());

    for (int i = 0; i < hullProjs.size(); ++i) {
        vec<4>& proj = hullProjs[i];
        proj[0] = INFINITY;
        proj[1] = -INFINITY;
        proj[2] = INFINITY;
        proj[3] = -INFINITY;
        bool in_bounds = false;
        for (int j = 0; j < hullPreds[i].size(); ++j) { 
            Projection p = project(linestrip, hullPreds[i][j], false);
            proj[0] = std::min(proj[0], p.arc_len);
            proj[1] = std::max(proj[1], p.arc_len);
            if (p.in_bounds) {
                in_bounds = true;
                proj[2] = std::min(proj[2], p.distance);
                proj[3] = std::max(proj[3], p.distance);
            }
        }
        if (!in_bounds) {
            proj[0] = -1000.0;
            proj[1] = -1000.0;
            proj[2] = -1000.0;
            proj[3] = -1000.0;
        }
    }
}

double IdmSamplingPlanner::getNextStopPoint(const IdmSamplingRefState& state,
                                            const RefLine& refLine,
                                            double l_trg,
                                            IdmSamplingParams& params) {

    double d_min = INFINITY;

    for (size_t i = 0; i < refLine.points.size(); ++i) {
        double s = i * refLine.step_size;
        if (state.s > s) {
            continue;
        }
        double d = s - state.s;
        if (refLine.points[i].v_max == 0.0) {
            d_min = std::min(d_min, d);
        }
        if (state.l < -refLine.points[i].d_right 
                || state.l > refLine.points[i].d_left) {
            d_min = std::min(d_min, d - params.d_safe_min);
        }
    }

    return d_min;
}

vec<4> IdmSamplingPlanner::getLeader(const IdmSamplingRefState& state,
                                     IdmSamplingTraj& traj,
                                     double l_trg,
                                     IdmSamplingParams& params) {

    vec<4> res(1e6, 0.0, 100.0, 100.0);

    double l = 200.0;

    std::vector<vec<2>> path = {
        {state.x - std::cos(state.heading) * params.dist_back_veh,
         state.y - std::sin(state.heading) * params.dist_back_veh},
        {state.x + std::cos(state.heading) * l,
         state.y + std::sin(state.heading) * l}
    };

    for (int oi = 0; oi < (int)dynObjs.size(); ++oi) {
        DynObj& d = dynObjs[oi];
        PredictionPoint pp = d.lerp(state.t);
        std::vector<vec<2>> hull = d.lerpHullPred(state.t);
        std::vector<Projection> projs(hull.size());

        if (!d.on_local_map) {
            continue;
        }

        bool on_left = false;
        bool on_right = false;

        for (size_t i = 0; i < hull.size(); ++i) {
            projs[i] = project(
                    std::span<vec<2>>(path.data(), path.size()),
                    hull[i],
                    false);
            on_right |= projs[i].distance < 0.0;
            on_left |= projs[i].distance > 0.0;
        }

        for (size_t i = 0; i < hull.size(); ++i) {
            if (!projs[i].in_bounds) {
                continue;
            }
            if ((on_left && on_right) 
                    || std::abs(projs[i].distance) < params.width_veh/2.0 + params.d_safe_lat) {
                double dist = projs[i].arc_len - params.dist_back_veh;
                if (dist < res[0]) {
                    res[0] = dist;
                    res[1] = pp.v * std::cos(pp.heading - state.heading);
                }
            }
            if (projs[i].arc_len < params.dist_front_veh) {
                if (on_right && on_left) {
                    res[2] = 0.0;
                    res[3] = 0.0;
                } else if (on_right) {
                    res[2] = std::min(res[2], std::abs(projs[i].distance));
                } else if (on_left) {
                    res[3] = std::min(res[3], std::abs(projs[i].distance));
                }
            }
        }

        // additional hull check on target lange

        vec<4> hp = d.lerpHullProj(state.t);

        if (traj.l_trg > hp[2] - params.width_veh/2.0 - params.d_safe_lat
                && traj.l_trg < hp[3] + params.width_veh/2.0 + params.d_safe_lat) {
            if (state.s < hp[1]) {
                double dist = hp[0] - state.s;
                if (dist < res[0]) {
                    res[0] = dist;
                    res[1] = pp.v; //* std::cos(pp.heading - proj.angle);
                }
            }
        }
    }

    // stronger reaction to vehicles driving in opposite direction

    if (res[1] < 0.0) {
        res[0] -= 10.0;
        res[1] *= 2.0;
    }

    return res;
}

void IdmSamplingPlanner::rolloutTrajectory(IdmSamplingTraj& traj,
                                           const RefLine& refLine,
                                           double dt_replan,
                                           std::vector<vec<2>>& linestrip,
                                           IdmSamplingParams& params) {

    traj.refStates.resize(params.steps_t);
    traj.states.resize(params.steps_t);

    double l_local_trg = traj.l_trg;

    for (int idx_t = 0; idx_t < params.steps_t-1; ++idx_t) {

        // compute reference
        {
            IdmSamplingRefState& ps_ref = traj.refStates[idx_t];
            IdmSamplingRefState& ns_ref = traj.refStates[idx_t+1];

            // compute idm acceleration
            double a_idm = 0.0;
            {
                double v_trg = 100.0;
                double v_trg_dist = ps_ref.v * params.t_vel_lookahead;
                int v_trg_steps = 25;
                double v_trg_step_size = v_trg_dist / (double)(v_trg_steps);
                for (int i = 0; i < v_trg_steps; ++i) {
                    v_trg = std::min(v_trg, refLine.lerp(ps_ref.s + i * v_trg_step_size).v_max);
                }
                v_trg = std::max(0.001, v_trg);

                vec<4> leader = getLeader(ps_ref, traj, l_local_trg, params);
                double d_stop = getNextStopPoint(ps_ref, refLine, l_local_trg, params);
                double d_next_inters_point = traj.d_stop - ps_ref.s;
                d_stop = std::min(d_next_inters_point, d_stop);

                ps_ref.d_right = leader[2];
                ps_ref.d_left = leader[3];

                double d_lead = leader[0];
                double v_lead = leader[1];

                double t_headway = params.t_headway_desired;
                // this weighting factor causes the t_headway to be reduced
                // if the vehicle is not close to the targeted reference offset
                // enables more aggressive merging
                t_headway *= 1.0 - sq(std::tanh((ps_ref.l - l_local_trg) * 0.5));
                t_headway = std::max(t_headway, 0.5);

                double s_net_stop = d_stop - params.dist_front_veh + 1.0;
                double s_star_stop = 1.0
                                   + ps_ref.v * t_headway
                                   + ps_ref.v * ps_ref.v / (
                                           2 * std::sqrt(params.a_max * params.a_break_comf));

                double inter_term = s_star_stop / s_net_stop;

                if (d_lead < d_stop) {
                    double s_net = d_lead - params.dist_front_veh;
                    double s_star = params.d_safe_min
                                  + ps_ref.v * t_headway
                                  + ps_ref.v * (ps_ref.v - v_lead) / (
                                          2 * std::sqrt(params.a_max * params.a_break_comf));

                    inter_term = std::max(s_star / s_net, inter_term);
                }

                double v_rel = ps_ref.v / v_trg;
                a_idm = params.a_max * (1.0
                        - std::pow(v_rel, v_rel < 1.0 ? params.idm_exp_acc : params.idm_exp_dcc)
                        - std::pow(inter_term, 2.0));
            }

            RefLinePoint rp = refLine.lerp(ps_ref.s);

            ns_ref.t = ps_ref.t + params.dt;

            double l_change = std::max(-1.5, std::min(1.5, l_local_trg - ps_ref.l));
            ns_ref.l = ps_ref.l + l_change * params.dt;

            double s_rate = ps_ref.v * std::cos(ps_ref.heading - rp.heading) / (1.0 - ps_ref.l * rp.k);
            ns_ref.s = ps_ref.s + s_rate * params.dt;

            RefLinePoint nrp = refLine.lerp(ns_ref.s);

            double heading_rel = shortAngleDist(ps_ref.heading, rp.heading);
            heading_rel = heading_rel + s_rate * rp.k * params.dt;

            ns_ref.heading = nrp.heading + heading_rel;
            ns_ref.x = nrp.x - ns_ref.l * std::sin(nrp.heading);
            ns_ref.y = nrp.y + ns_ref.l * std::cos(nrp.heading);

            double dt_control = (idx_t == 0) ? dt_replan : params.dt;

            // prevents acceleration during lane change at low velocities
            if (std::abs(ps_ref.l - l_local_trg) > 0.5
                    && ps_ref.v > 1.0 && ps_ref.v < 5.0) {
                a_idm = std::min(0.0, a_idm);
            }

            // compute and limit jerk
            double j = (a_idm - ps_ref.a) / dt_control;
            
            if (ps_ref.v == 0.0 && ps_ref.a < 0.0) { 
                // allows fast opening of the brakes in standstill
                j = std::max(params.j_min, std::min(-ps_ref.a / dt_control, j));
            } else {
                j = std::max(params.j_min, std::min(params.j_max, j));
            }

            ps_ref.a = ps_ref.a + j * dt_control; 
            ps_ref.a = std::max(params.a_min, std::min(params.a_max, ps_ref.a));
            ns_ref.v = max(0.0, ps_ref.v + ps_ref.a * params.dt);
            ns_ref.a = ps_ref.a;
        }

        // compute following controller
        {
            IdmSamplingState& ps = traj.states[idx_t];
            IdmSamplingState& ns = traj.states[idx_t+1];

            const IdmSamplingRefState rs = traj.refStates[idx_t];

            double dt_control = (idx_t == 0) ? dt_replan : params.dt;
            
            // compute steering angle
            {
                const RefLinePoint rp = refLine.lerp(ps.s);
                double k_adj = rp.k;
                if (std::abs(rp.k) > 1.0e-4) {
                    k_adj = 1.0/(1.0 / rp.k + ps.l);
                }
                double steer_angle_ref = std::atan(k_adj * params.wheel_base);

                double angle_diff = shortAngleDist(ps.heading, rs.heading);
                double lat_diff = rs.l - ps.l;

                double steer_angle = steer_angle_ref + angle_diff + std::atan(
                    params.k_stanley * lat_diff
                    / (params.v_offset_stanley + ps.v));
                steer_angle = std::max(-params.steer_angle_max, 
                        std::min(params.steer_angle_max, steer_angle));

                double steer_rate = std::max(-params.steer_rate_max,
                        std::min(params.steer_rate_max,
                            (steer_angle - ps.steer_angle) / dt_control));

                if (ps.v > 1.0 || ps.a > 0.5 || std::abs(lat_diff) > 0.1) {
                    ps.steer_angle += steer_rate * dt_control;
                }
            }

            // compute acceleration controller
            {
                double err_s = rs.s - ps.s;
                double err_v = rs.v - ps.v;
                ps.a = rs.a + err_s * params.k_p_s
                            + err_v * params.k_p_v;
            }

            // integrate vehicle model with semi-implicit euler integration
            {
                ns.t = ps.t + params.dt;
                ns.a = ps.a;
                ns.steer_angle = ps.steer_angle;
                ns.v = std::max(0.0, ps.v + params.dt * ps.a);
                ns.heading = ps.heading + params.dt * ns.v * std::tan(ps.steer_angle) / params.wheel_base;
                ns.x = ps.x + params.dt * ns.v * std::cos(ns.heading);
                ns.y = ps.y + params.dt * ns.v * std::sin(ns.heading);

                Projection proj = project(linestrip, vec<2>(ns.x, ns.y), false);
                ns.s = proj.arc_len;
                ns.l = proj.distance;
            }
        }
    }
}

void IdmSamplingPlanner::evalTrajectory(IdmSamplingTraj& traj,
                                        const RefLine& refLine,
                                        IdmSamplingParams& params) {

    for (size_t i = 0; i < traj.states.size(); ++i) {
        IdmSamplingState& s = traj.states[i];

        // compute geometric vehicle center
        double l = params.length_veh/2.0 - params.dist_back_veh;
        vec<2> center_veh(s.x + l * std::cos(s.heading),
                          s.y + l * std::sin(s.heading));

        std::vector<vec<2>> hull_veh = getVehicleHull(s, params);

        // collision checks

        for (DynObj& d : dynObjs) {
            PredictionPoint pp = d.lerp(s.t);
            double dist = (vec<2>(pp.x, pp.y) - center_veh).norm();

            // fast conservative approximative check
            if (dist > params.radius_veh + d.radiusHull + pp.v) {
                continue;
            }
            
            // slower precise check
            std::vector<vec<2>> hull_obj = d.lerpHullPred(s.t);
            if (intersectPolygons(hull_veh, hull_obj)) {
                if (s.t < 3.0) { 
                    traj.invalid = true;
                }
                traj.cost_collision += params.steps_t * params.dt - s.t;
                return;
            }
        }

        // check if blocking obstacles traveling in opposite direction

        for (DynObj& d : traj.dynObjs) {

            Projection proj = project(d.path, vec<2>(s.x, s.y), false);

            if (!proj.in_bounds) {
                continue;
            }

            if (std::abs(proj.distance) > params.radius_veh + d.radiusHull) {
                continue;
            }

            const PredictionPoint pp = d.lerpStation(proj.arc_len);

            std::vector<vec<2>> hullPred = d.lerpHullPred(pp.t);

            if (!intersectPolygons(hull_veh, hullPred)) {
                continue;
            }

            if (std::cos(s.heading - proj.angle) < 0.0) {
                traj.cost_interaction += 1.0/(1.0 + std::abs(proj.distance));
            }
        }
    }

    // total traveled distance

    double distance = 0.0;
    for (size_t i = 1; i < traj.states.size(); ++i) {
        IdmSamplingState& prev = traj.states[i-1];
        IdmSamplingState& next = traj.states[i];
        distance += std::sqrt(sq(prev.x - next.x) + sq(prev.y - next.y));
    }
    traj.cost_distance = 1000.0 - distance;

    // comfort costs

    traj.cost = 0.0;
    traj.cost += params.w_l * sq(params.l_trg - traj.l_trg);

    double minDistLeft = 100.0;
    double minDistRight = 100.0;
    for (size_t i = 0; i < traj.refStates.size()-1; ++i) {
        minDistLeft = std::min(minDistLeft, traj.refStates[i].d_left);
        minDistRight = std::min(minDistRight, traj.refStates[i].d_right);
    }
    if (minDistLeft < params.d_comf_lat) { 
        traj.cost += params.w_lat_dist * (params.d_comf_lat - minDistLeft) / params.d_comf_lat;
    }
    if (minDistRight < params.d_comf_lat) { 
        traj.cost += params.w_lat_dist * (params.d_comf_lat - minDistRight) / params.d_comf_lat;
    }

    for (IdmSamplingState& s : traj.states) {
        traj.cost += params.w_a * sq(std::min(0.0, s.a));
    }

    for (size_t i = 0; i < traj.states.size(); ++i) {

        RefLinePoint rp = refLine.lerp(traj.states[i].s);

        if (traj.states[i].l > rp.d_left - params.width_veh / 2.0 * std::sqrt(2.0)) {
            traj.cost_collision += 1.0;
        }
        if (traj.states[i].l < -rp.d_right + params.width_veh / 2.0 * std::sqrt(2.0)) {
            traj.cost_collision += 1.0;
        }
    }
}

std::vector<vec<2>> IdmSamplingPlanner::getVehicleHull(IdmSamplingState& s,
                                                       IdmSamplingParams& params) {

    std::vector<vec<2>> hull_veh(4);
    hull_veh[0][0] = params.dist_back_veh;
    hull_veh[0][1] = -params.width_veh/2.0;
    hull_veh[1][0] = params.dist_front_veh;
    hull_veh[1][1] = -params.width_veh/2.0;
    hull_veh[2][0] = params.dist_front_veh;
    hull_veh[2][1] = params.width_veh/2.0;
    hull_veh[3][0] = params.dist_back_veh;
    hull_veh[3][1] = params.width_veh/2.0;

    for (vec<2>& v : hull_veh) {
        vec<2> res;
        res[0] = v[0] * std::cos(s.heading) - v[1] * std::sin(s.heading);
        res[1] = v[0] * std::sin(s.heading) + v[1] * std::cos(s.heading);
        res[0] += s.x;
        res[1] += s.y;
        v[0] = res[0];
        v[1] = res[1];
    }

    return hull_veh;
}

IdmSamplingTraj IdmSamplingPlanner::update(IdmSamplingState initState,
                                           IdmSamplingRefState initRefState,
                                           double dt_replan,
                                           const RefLine& refLine,
                                           IdmSamplingParams& params) {

    // preprocessing

    std::vector<vec<2>> linestrip(refLine.points.size());
    for (size_t i = 0; i < refLine.points.size(); ++i) {
        linestrip[i] = vec<2>(refLine.points[i].x,
                              refLine.points[i].y);
    }

    for (DynObj& d : dynObjs) {
        d.updatePredGeometry(linestrip);
    }

    // determine where the left and right steps should be

    double d_left = -100.0;
    double d_right = 100.0;

    for (const RefLinePoint& p : refLine.points) {
        d_left = max(d_left, p.d_left);
        d_right = min(d_right, -p.d_right);
    }

    // the maximum lane range we need to cover minus safety distance
    d_left -= params.d_safe_lat_path + params.width_veh / 2.0 * std::sqrt(2.0);
    d_right += params.d_safe_lat_path + params.width_veh / 2.0 * std::sqrt(2.0);

    double d_left_step_size = d_left / params.lat_steps;
    double d_right_step_size = d_right / params.lat_steps;

    std::vector<double> ls;
    for (int i = params.lat_steps-1; i >= 0; --i) {
        double l = d_right_step_size * (i + 1);
        ls.push_back(l);
    }
    ls.push_back(0.0);
    for (int i = 0; i < params.lat_steps; ++i) {
        double l = d_left_step_size * (i + 1);
        ls.push_back(l);
    }

    for (IdmSamplingState& s : prevTraj.states) {
        s.t -= dt_replan;
    }
    for (IdmSamplingRefState& s : prevTraj.refStates) {
        s.t -= dt_replan;
    }

    // shift update and integrate dead time traj
    IdmSamplingState initConState;
    std::vector<IdmSamplingState> intStates;
    if (params.dead_time > 0.0) {
        int steps_dead_int = 11;
        double dt = params.dead_time / (double)(steps_dead_int-1);

        // shift and drop old states
        for (IdmSamplingState& s : deadTimeTraj.states) {
            s.t -= dt_replan;
        }
        std::vector<IdmSamplingState> states;
        for (IdmSamplingState& s : deadTimeTraj.states) {
            if (s.t >= 0.0) {
                states.push_back(s);
            }
        }
        deadTimeTraj.states = states;

        if (deadTimeTraj.states.size() == 0) {
            // init dead time traj
            for (int i = 0; i < steps_dead_int; ++i) {
                IdmSamplingState s;
                s.t = i * dt;
                s.a = initState.a;
                s.steer_angle = initState.steer_angle;
                deadTimeTraj.states.push_back(s);
            }
        }

        for (int i = 0; i < steps_dead_int; ++i) {
            intStates.push_back(deadTimeTraj.lerp(dt * i));
        }

        double steerAngleTmp = intStates[0].steer_angle;
        double aTmp = intStates[0].a;
        intStates[0] = initState;
        intStates[0].steer_angle = steerAngleTmp;
        intStates[0].a = aTmp;

        for (int i = 0; i < steps_dead_int-1; ++i) {
            IdmSamplingState& ps = intStates[i];
            IdmSamplingState& ns = intStates[i+1];
            ns.t = ps.t + dt;
            ns.v = ps.v + dt * ps.a;
            ns.heading = ps.heading + dt * ns.v * std::tan(ps.steer_angle) / params.wheel_base;
            ns.x = ps.x + dt * ns.v * std::cos(ns.heading);
            ns.y = ps.y + dt * ns.v * std::sin(ns.heading);

            Projection proj = project(linestrip, vec<2>(ns.x, ns.y), false);
            ns.s = proj.arc_len;
            ns.l = proj.distance;
        }

        initConState = intStates.back();
    } else {
        initConState = initState;
    }

    trajs.clear();
    {
        for (const double& l : ls) {
            IdmSamplingTraj traj;
            traj.refStates.push_back(initRefState);
            traj.states.push_back(initConState);
            traj.dynObjs = dynObjs;
            traj.l_trg = l;
            trajs.push_back(traj);
        }

        if (params.enable_reverse) {
            for (const double& l : ls) {
                IdmSamplingTraj traj;
                traj.refStates.push_back(initRefState);
                traj.states.push_back(initConState);
                traj.dynObjs = dynObjs;
                traj.l_trg = l;
                traj.reverse = true;
                trajs.push_back(traj);
            }
        }

        // trajectory option to stop a next stop point
        IdmSamplingTraj stopTraj;
        stopTraj.states.push_back(initConState);
        stopTraj.refStates.push_back(initRefState);
        stopTraj.dynObjs = dynObjs;
        stopTraj.l_trg = 0.0;
        stopTraj.d_stop = params.d_next_inters_point;
        trajs.push_back(stopTraj);
    }

    for (IdmSamplingTraj& traj : trajs) {
        rolloutTrajectory(traj, refLine, dt_replan, linestrip, params);
    }
    for (IdmSamplingTraj& traj : trajs) {
        evalTrajectory(traj, refLine, params);
    }

    double cost_collision_min = INFINITY;
    for (size_t i = 0; i < trajs.size(); ++i) {
        cost_collision_min = std::min(
                cost_collision_min, trajs[i].cost_collision);
    }
    double cost_interaction_min = INFINITY;
    for (size_t i = 0; i < trajs.size(); ++i) {
        if (trajs[i].cost_collision > cost_collision_min) { 
            continue;
        }
        cost_interaction_min = std::min(
                cost_interaction_min, trajs[i].cost_interaction);
    }
    double cost_distance_min = INFINITY;
    for (size_t i = 0; i < trajs.size(); ++i) {
        if (trajs[i].cost_collision > cost_collision_min) { 
            continue;
        }
        if (trajs[i].cost_interaction > cost_interaction_min) { 
            continue;
        }
        if (trajs[i].cost_distance < cost_distance_min) {
            cost_distance_min = trajs[i].cost_distance;
        }
    }
    int i_min = 0;
    double cost_min = INFINITY;
    for (size_t i = 0; i < trajs.size(); ++i) {
        if (trajs[i].cost_collision > cost_collision_min) { 
            continue;
        }
        if (trajs[i].cost_interaction > cost_interaction_min) { 
            continue;
        }
        if (std::abs(trajs[i].cost_distance - cost_distance_min) > 5.0) {
            continue;
        }
        if (trajs[i].cost < cost_min) {
            cost_min = trajs[i].cost;
            i_min = i;
        }
    }

    int i_select = i_prev;
    if (i_select < 0) {
        i_select = i_min;
    }
    if (trajs[i_select].invalid) {
        i_select = i_min;
    }
    if (i_min != i_select) {
        if (i_min_prev == i_min) {
            t_decision += dt_replan;
            if (t_decision > params.dt_decision) {
                i_select = i_min;
                t_decision = 0;
            }
        } else {
            t_decision = 0;
        }
        i_min_prev = i_min;
    } 

    dynObjs.clear();

    prevTraj = trajs[i_select];
    i_prev = i_select;

    std::vector<IdmSamplingState> combinedStates;
    for (IdmSamplingState& s : intStates) {
        combinedStates.push_back(s);
    }
    if (combinedStates.size() > 0) {
        combinedStates.pop_back();
    }
    for (IdmSamplingState& s : prevTraj.states) {
        combinedStates.push_back(s);
    }
    prevTraj.states = combinedStates;

    // append to dead time traj
    {
        IdmSamplingState s = prevTraj.lerp(params.dead_time);
        deadTimeTraj.states.push_back(s);
    }

    return prevTraj;
}
