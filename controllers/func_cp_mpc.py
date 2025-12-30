from __future__ import annotations

import numpy as np

from cp.functional_cp import CPStepParameters
from typing import List, Optional


class FunctionalCPMPC:
    """
    PCA-GMM 파라미터를 사용하여 실시간으로 안전 거리를 계산하는 MPC 컨트롤러.
    그리드를 생성하지 않고 파라미터 수식을 직접 평가하여 매우 빠릅니다.
    """

    def __init__(
            self,
            cp_params: List[CPStepParameters],
            box: float = 10.0,
            world_center: np.ndarray = np.array([0, 0]),
            n_steps: int = 20,
            dt: float = 0.1,
            n_skip: int = 4,
            robot_rad: float = 0.4,
            obstacle_rad: float = 0.707,
            min_linear_x: float = -0.8,
            max_linear_x: float = 0.8,
            min_angular_z: float = -0.7,
            max_angular_z: float = 0.7,
            n_paths: int = 1000,
            seed: int = 42
    ):
        # 환경 및 로봇 설정
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.params = {p.t_idx: p for p in cp_params} # 타임스텝별 파라미터 맵
        
        self.n_steps = n_steps
        self.dt = dt
        self.n_skip = n_skip
        self.robot_rad = robot_rad
        self.obstacle_rad = obstacle_rad
        self.safe_rad = robot_rad + obstacle_rad

        # 제어 제한
        self.min_v, self.max_v = min_linear_x, max_linear_x
        self.min_w, self.max_w = min_angular_z, max_angular_z
        
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def _get_grid_idx(self, pos_world: np.ndarray, grid_res: int) -> Optional[int]:
        """월드 좌표를 SDF 그리드의 1차원 인덱스로 변환"""
        rel = pos_world - self.world_center
        u = (rel[0] + self.box / 2.0) / self.box * (grid_res - 1)
        v = (rel[1] + self.box / 2.0) / self.box * (grid_res - 1)
        
        if not (0 <= u < grid_res and 0 <= v < grid_res):
            return None
        return int(round(v)) * grid_res + int(round(u))

    def evaluate_u_parametric(self, pos_world: np.ndarray, t_idx: int) -> float:
        """
        논문 Section 1.4 (24)식: 파라미터화된 상한선 U_i(x) 평가.
        U_i(x) = mean(idx) + epsilon + max_k { mu_k^T @ phi(idx) + r_k * sqrt(phi^T @ Sigma_k @ phi) }
        """
        p = self.params.get(t_idx)
        if p is None: 
            return 0.5 * self.dt * (t_idx + 1) # 파라미터가 없으면 보수적 에러 반환

        # 1. 위치에 대응하는 PCA 기저 및 평균 참조
        D = p.phi_basis.shape[1]
        grid_res = int(np.sqrt(D))
        idx = self._get_grid_idx(pos_world, grid_res)
        
        if idx is None: return 1.5 # 맵 밖은 매우 불확실한 지역으로 간주

        phi_x = p.phi_basis[:, idx] # (p_eff,)
        m_x = p.mean_field[idx]     # (1,)

        # 2. GMM 각 성분별 타원체 엔벨로프 계산
        k_vals = []
        for k in range(p.K):
            # mu_k^T * phi(x)
            term_center = p.mus[k] @ phi_x
            # r_k * sqrt(phi(x)^T * Sigma_k * phi(x))
            term_radius = p.rks[k] * np.sqrt(max(phi_x.T @ p.sigmas[k] @ phi_x, 0.0))
            k_vals.append(term_center + term_radius)

        # 3. 최종 상한선 (식 24)
        return float(m_x + p.epsilon + max(k_vals))

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):
        """MPC 메인 실행 함수"""
        # 1. 경로 후보 생성 (Rollout)
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z)

        # 2. 정적 장애물 및 CP 기반 동적 안전성 필터링
        safe_paths, vels_safe = self.filter_paths(paths, vels, boxes, predictions)

        if safe_paths is None:
            return None, {'feasible': False, 'reason': 'no safe path'}

        # 3. 최적 경로 선택 (목적지 거리 + 제어 비용)
        best_idx, cost = self.score_paths(safe_paths, vels_safe, goal)

        info = {
            'feasible': True,
            'final_path': safe_paths[best_idx],
            'cost': cost,
            'n_safe': len(safe_paths)
        }
        
        # 반환값 형식: [v_best, w_best], info
        return vels_safe[best_idx, 0], info

    def filter_paths(self, paths, vels, boxes, predictions):
        P, T_p1, _ = paths.shape
        T = T_p1 - 1
        
        # 1) 정적 장애물 필터링
        mask_static_unsafe = np.zeros(P, dtype=bool)
        for box in boxes:
            center, sz, th = box.pos, np.array([box.w, box.h]), box.rad
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            lb, ub = -0.5 * sz - self.robot_rad, 0.5 * sz + self.robot_rad
            transformed = (paths[:, 1:, :] - center) @ R
            coll = np.any(np.all((transformed >= lb) & (transformed <= ub), axis=-1), axis=-1)
            mask_static_unsafe |= coll

        # 2) CP 기반 동적 장애물 필터링 (Section 3.3: D_lower >= r_safe)
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        if predictions:
            # predictions {node_id: (T, 2)} -> (T, M, 2) 배열로 변환
            pred_list = list(predictions.values())
            pred_array = np.array(pred_list).transpose(1, 0, 2) # (T, M, 2)
            T_use = min(T, pred_array.shape[0], len(self.params))

            for i in range(P):
                for t in range(T_use):
                    pos = paths[i, t+1]
                    # 명목 거리 (Nominal Distance to nearest obstacle)
                    d_nom = np.min(np.linalg.norm(pos - pred_array[t], axis=1))
                    
                    # 온라인 CP 상한선 계산 (수식 평가)
                    u_val = self.evaluate_u_parametric(pos, t)
                    
                    # 논문 Section 3.3 제약 조건: D_lower = max(D_nom - U, 0)
                    d_lower = max(d_nom - u_val, 0.0)
                    
                    if d_lower < self.safe_rad:
                        mask_dynamic_unsafe[i] = True
                        break

        mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)
        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe]
        return None, None

    def generate_paths_random(self, px, py, th):
        """무작위 샘플링 기반 경로 생성 (EgocentricCPMPC 스타일)"""
        n_epochs = self.n_steps // self.n_skip
        # 선형/각속도 세트 무작위 선택
        v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs))
        w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs))
        
        v = np.repeat(v_epoch, self.n_skip, axis=1)[:, :self.n_steps]
        w = np.repeat(w_epoch, self.n_skip, axis=1)[:, :self.n_steps]
        
        paths = np.zeros((self.n_paths, self.n_steps + 1, 2))
        paths[:, 0, 0], paths[:, 0, 1] = px, py
        
        curr_th = np.full(self.n_paths, th)
        for t in range(self.n_steps):
            paths[:, t+1, 0] = paths[:, t, 0] + self.dt * v[:, t] * np.cos(curr_th)
            paths[:, t+1, 1] = paths[:, t, 1] + self.dt * v[:, t] * np.sin(curr_th)
            curr_th += self.dt * w[:, t]
            
        return paths.astype(np.float32), np.stack([v, w], axis=-1).astype(np.float32)

    def score_paths(self, paths, vels, goal):
        """최종 경로 점수화"""
        # 목적지 거리 비용 (Terminal + Intermediate)
        dist_to_goal = np.linalg.norm(paths[:, -1, :] - goal, axis=1)
        # 제어 비용
        ctrl_cost = 0.01 * np.sum(vels**2, axis=(1, 2))
        
        total_cost = dist_to_goal + ctrl_cost
        best_idx = np.argmin(total_cost)
        return best_idx, total_cost[best_idx]