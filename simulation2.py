import os
import sys
import traci
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# SUMO Configuration
sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.path.append(os.path.join(sumo_home, 'tools'))

class OvertakingPhase(Enum):
    LANE_KEEPING = 0
    OVERTAKING_PHASE_1 = 1  # Lane change to opposite lane
    OVERTAKING_PHASE_2 = 2  # Driving in opposite lane
    OVERTAKING_PHASE_3 = 3  # Return to original lane
    WAITING = 4
    RETURN_TO_ORIGINAL = 5

class GameAction(Enum):
    YIELD = 0
    NOT_YIELD = 1

@dataclass
class VehicleState:
    id: str
    position: float
    speed: float
    lane_id: str
    lane_position: float
    lane_index: int
    
@dataclass
class GamePayoff:
    speed_payoff: float
    safety_payoff: float
    total_payoff: float

class PotentialConflictArea:
    def __init__(self, vehicle_f_pos: float, vehicle_f_speed: float):
        self.L_car = 4.5  # Vehicle length
        self.D_f_safe = self.calculate_safe_distance(vehicle_f_speed)
        self.D_stop = 2.0  # Stopping distance
        self.D_c_safe = self.L_car + self.D_f_safe + self.D_stop
        self.position = vehicle_f_pos + self.D_c_safe

        
    def calculate_safe_distance(self, speed: float) -> float:
        """Calculate safe distance (Formula 2)"""
        t_delay = 1.5  # Reaction time
        a_pd = 3.0  # Braking acceleration
        if speed <= 0:
            return 0
        return speed * t_delay + (speed * speed) / (2 * a_pd)
    
    def update_position(self, vehicle_f_pos: float, vehicle_f_speed: float):
        """Update potential conflict area position"""
        self.D_f_safe = self.calculate_safe_distance(vehicle_f_speed)
        self.D_c_safe = self.L_car + self.D_f_safe + self.D_stop
        self.position = vehicle_f_pos + self.D_c_safe

class OvertakingDecisionMaker:
    def __init__(self):
        self.TTC_threshold = 8.0  # seconds (更保守，給更多緩衝時間)
        self.D_threshold = 150.0  # Distance threshold for triggering overtaking 
        self.V_expect = 18.0  # Expected speed (m/s)
        self.TLC = 4.0  # Lane change time
        self.lane_width = 3.5  # Lane width
        
        # Game theory parameters
        self.alpha_1 = 0.9  # Ego vehicle speed weight (更重視速度)
        self.beta_1 = 0.1   # Ego vehicle safety weight
        self.alpha_2 = 0.5  # Oncoming vehicle speed weight
        self.beta_2 = 0.5   # Oncoming vehicle safety weight
        
        self.U_traffic = 25.0  # Road speed limit
        

    def compute_aggressiveness_rho(self, ego_state: VehicleState, oncoming_state: VehicleState,
                                conflict_area: PotentialConflictArea) -> float:
        """
        根據論文公式 (27)–(30) 計算 oncoming vehicle 的 aggressiveness ρ
        """
        tego = self.calculate_time_to_conflict(ego_state.position, ego_state.speed, conflict_area.position, 0)
        VF = 0  # 前車速度不影響這裡的計算
        aF = 0  # 假設前車加速度為 0
        aC1 = 0  # 假設對向車也維持等速（可加速強化）
        VC1 = oncoming_state.speed

        # 計算未來位置
        S_conflict_future = conflict_area.position + VF * tego + 0.5 * aF * tego ** 2
        S_C1_future = oncoming_state.position - VC1 * tego - 0.5 * aC1 * tego ** 2

        DP_C1 = S_C1_future - S_conflict_future

        # 預設最大加減速度
        amax = 2.0
        amin = -6.0  # 限制對向車最大減速能力

        Dmax = oncoming_state.position - VC1 * tego - 0.5 * amax * tego ** 2 - S_conflict_future
        Dmin = oncoming_state.position - VC1 * tego - 0.5 * amin * tego ** 2 - S_conflict_future

        if Dmin <= 0:
            k = 1.0
        elif Dmax >= 0:
            k = 0.0
        else:
            k = abs(Dmax) / (Dmin - Dmax)

        # 公式 (30)：計算 ρ
        if DP_C1 <= 0:
            rho = (k / abs(DP_C1)) * abs(Dmax)
        else:
            rho = k * (DP_C1 / Dmin)

        return rho

    def calculate_lane_change_distance(self, ego_speed: float) -> float:
        """Calculate lane change distance (Formula 3)"""
        if ego_speed >= 3.0:
            return ego_speed * self.TLC
        else:
            return 3.0 * self.TLC
    
    def check_overtaking_trigger(self, ego_pos: float, ego_speed: float, 
                                vehicle_f_pos: float, vehicle_f_speed: float) -> bool:
        """Check if overtaking intention is triggered - 更寬鬆的觸發條件"""
        distance_to_front = vehicle_f_pos - ego_pos
        
        # 非常寬鬆的觸發條件
        condition1 = (distance_to_front > 10 and distance_to_front < self.D_threshold)
        condition2 = (vehicle_f_speed < self.V_expect - 6.0)  # 前車較慢即可
        condition3 = (ego_speed > vehicle_f_speed + 3.0)  # 速度差異要求降低
        
        return condition1 and condition2 and condition3
    
    def calculate_time_to_conflict(self, ego_pos: float, ego_speed: float, 
                                 conflict_pos: float, vehicle_f_speed: float) -> float:
        """Calculate time to reach potential conflict area"""
        if ego_speed <= 0:
            return float('inf')
        return max((conflict_pos - ego_pos) / ego_speed, 0.1)
    
    def calculate_speed_payoff_ego_yield(self, ego_speed: float, vehicle_f_speed: float) -> float:
        """Calculate ego vehicle speed payoff when yielding"""
        v_wait = min(vehicle_f_speed * 1.2, ego_speed * 0.4)  # 降低讓行時的速度獎勵
        return max(v_wait / self.U_traffic, 0.05)
    
    def calculate_speed_payoff_ego_not_yield(self) -> float:
        """Calculate ego vehicle speed payoff when not yielding"""
        v_overtake = min(self.V_expect + 6.0, 25.0)  # 大幅增加超車獎勵
        return v_overtake / self.U_traffic
    
    def calculate_speed_payoff_oncoming_yield(self, oncoming_pos: float, conflict_pos: float, 
                                            ego_time_to_conflict: float) -> float:
        """Calculate oncoming vehicle speed payoff when yielding"""
        distance_to_conflict = abs(conflict_pos - oncoming_pos)
        if ego_time_to_conflict <= 0:
            return 0.1 / self.U_traffic
        
        v_yield = min(distance_to_conflict / ego_time_to_conflict, 12.0)
        return max(v_yield / self.U_traffic, 0.1)
    
    def calculate_speed_payoff_oncoming_not_yield(self) -> float:
        """Calculate oncoming vehicle speed payoff when not yielding"""
        v_oncoming = 14.0
        return v_oncoming / self.U_traffic
    
    def calculate_safety_payoff(self, ego_time: float, oncoming_time: float) -> float:
        """Calculate safety payoff"""
        TTC = abs(ego_time - oncoming_time)
        if TTC >= self.TTC_threshold:
            return 1.0
        else:
            return max(TTC / self.TTC_threshold, 0.2)  # 提高最低安全獎勵
    
    def calculate_game_payoffs(self, ego_state: VehicleState, oncoming_state: VehicleState,
                           conflict_area: PotentialConflictArea) -> dict:
        """Calculate game payoff matrix with aggressiveness-aware weights"""

        # Time to conflict
        ego_time = self.calculate_time_to_conflict(ego_state.position, ego_state.speed, conflict_area.position, 0)
        oncoming_distance = abs(conflict_area.position - oncoming_state.position)
        oncoming_time = oncoming_distance / max(oncoming_state.speed, 0.1)

        # Speed payoff
        ego_speed_yield = self.calculate_speed_payoff_ego_yield(ego_state.speed, ego_state.speed * 0.4)
        ego_speed_not_yield = self.calculate_speed_payoff_ego_not_yield()
        oncoming_speed_yield = self.calculate_speed_payoff_oncoming_yield(oncoming_state.position, conflict_area.position, ego_time)
        oncoming_speed_not_yield = self.calculate_speed_payoff_oncoming_not_yield()

        # Safety payoff
        safety_payoff_conflict = self.calculate_safety_payoff(ego_time, oncoming_time)
        safety_payoff_safe = 1.0

        # --- 這裡是新加的部分：計算 aggressiveness rho 並調整權重 ---
        rho = self.compute_aggressiveness_rho(ego_state, oncoming_state, conflict_area)
        omega_1 = 0.3
        omega_2 = 0.3
        beta1_adj = max(self.beta_1 - omega_1 * rho, 0.0)
        beta2_adj = min(self.beta_2 + omega_2 * rho, 1.0)
        alpha1_adj = 1.0 - beta1_adj
        alpha2_adj = 1.0 - beta2_adj

        # --- payoff matrix ---
        ego_payoff_yy = GamePayoff(
            ego_speed_yield, safety_payoff_safe,
            alpha1_adj * ego_speed_yield + beta1_adj * safety_payoff_safe
        )

        oncoming_payoff_yy = GamePayoff(
            oncoming_speed_yield, safety_payoff_safe,
            alpha2_adj * oncoming_speed_yield + beta2_adj * safety_payoff_safe
        )

        ego_payoff_yn = GamePayoff(
            ego_speed_yield, safety_payoff_safe,
            alpha1_adj * ego_speed_yield + beta1_adj * safety_payoff_safe
        )

        oncoming_payoff_yn = GamePayoff(
            oncoming_speed_not_yield, safety_payoff_safe,
            alpha2_adj * oncoming_speed_not_yield + beta2_adj * safety_payoff_safe
        )

        ego_payoff_ny = GamePayoff(
            ego_speed_not_yield * 1.5, safety_payoff_safe,
            alpha1_adj * ego_speed_not_yield * 1.5 + beta1_adj * safety_payoff_safe
        )

        oncoming_payoff_ny = GamePayoff(
            oncoming_speed_yield, safety_payoff_safe,
            alpha2_adj * oncoming_speed_yield + beta2_adj * safety_payoff_safe
        )

        ego_payoff_nn = GamePayoff(
            ego_speed_not_yield, safety_payoff_conflict * 0.9,
            alpha1_adj * ego_speed_not_yield + beta1_adj * safety_payoff_conflict * 0.9
        )

        oncoming_payoff_nn = GamePayoff(
            oncoming_speed_not_yield, safety_payoff_conflict,
            alpha2_adj * oncoming_speed_not_yield + beta2_adj * safety_payoff_conflict
        )

        return {
            'YY': (ego_payoff_yy, oncoming_payoff_yy),
            'YN': (ego_payoff_yn, oncoming_payoff_yn),
            'NY': (ego_payoff_ny, oncoming_payoff_ny),
            'NN': (ego_payoff_nn, oncoming_payoff_nn)
        }

    def solve_nash_equilibrium(self, payoff_matrix) -> Tuple[GameAction, GameAction]:
        """Solve Nash equilibrium with strong bias towards overtaking"""
        # Extract payoff values
        ego_yy, oncoming_yy = payoff_matrix['YY'][0].total_payoff, payoff_matrix['YY'][1].total_payoff
        ego_yn, oncoming_yn = payoff_matrix['YN'][0].total_payoff, payoff_matrix['YN'][1].total_payoff  
        ego_ny, oncoming_ny = payoff_matrix['NY'][0].total_payoff, payoff_matrix['NY'][1].total_payoff
        ego_nn, oncoming_nn = payoff_matrix['NN'][0].total_payoff, payoff_matrix['NN'][1].total_payoff
        
        # 強烈增加超車偏好
        ego_ny += 0.25  # 大幅獎勵
        ego_nn += 0.15
        
        # 強制選擇超車，除非安全性極差
        if ego_ny >= ego_yy * 0.8:  # 只要超車收益不是太差
            return GameAction.NOT_YIELD, GameAction.YIELD
        elif ego_nn >= ego_yn * 0.8:
            return GameAction.NOT_YIELD, GameAction.NOT_YIELD
        else:
            return GameAction.NOT_YIELD, GameAction.YIELD  # 默認超車
        

    
    def make_overtaking_decision(self, ego_state: VehicleState, 
                               vehicle_f_state: Optional[VehicleState],
                               oncoming_vehicles: List[VehicleState]) -> GameAction:
        """Main overtaking decision logic"""
        if not vehicle_f_state:
            return GameAction.YIELD
        
        # Check if overtaking intention is triggered
        if not self.check_overtaking_trigger(ego_state.position, ego_state.speed,
                                           vehicle_f_state.position, vehicle_f_state.speed):
            return GameAction.YIELD
        
        # Calculate potential conflict area
        conflict_area = PotentialConflictArea(vehicle_f_state.position, vehicle_f_speed=vehicle_f_state.speed)
        
        # If no oncoming vehicles, definitely overtake
        if not oncoming_vehicles:
            return GameAction.NOT_YIELD
        
        # Find closest oncoming vehicle
        closest_oncoming = min(oncoming_vehicles, 
                             key=lambda v: abs(v.position - conflict_area.position))
        
        # Calculate TTC
        ego_time = self.calculate_time_to_conflict(ego_state.position, ego_state.speed,
                                                 conflict_area.position, 0)
        oncoming_time = abs(conflict_area.position - closest_oncoming.position) / max(closest_oncoming.speed, 0.1)
        
        TTC = abs(ego_time - oncoming_time)
        
        # 非常激進的決策邏輯
        if TTC > self.TTC_threshold:
            return GameAction.NOT_YIELD
        elif oncoming_time > ego_time + 4.0:  # 對向車還很遠
            return GameAction.NOT_YIELD
        elif TTC > 3.0:  # 基本安全時間
            return GameAction.NOT_YIELD
        
        # Game theory decision
        payoff_matrix = self.calculate_game_payoffs(ego_state, closest_oncoming, conflict_area)
        ego_action, _ = self.solve_nash_equilibrium(payoff_matrix)
        
        return ego_action

class SUMOOvertakingSimulation:
    def __init__(self):
        self.decision_maker = OvertakingDecisionMaker()
        self.ego_vehicle_id = "ego"
        self.front_vehicle_id = "front"
        self.oncoming_vehicle_ids = ["oncoming1", "oncoming2"]
        
        self.current_phase = OvertakingPhase.LANE_KEEPING
        self.simulation_data = []
        self.lane_change_duration = 0
        self.overtaking_commitment = False  # 超車承諾標記
        
    def create_sumo_config(self):
        """Create SUMO configuration files with improved setup"""
        # Create network file
        net_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,1500.00,10.50" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>

    <!-- Eastbound road (main direction) -->
    <edge id="eastbound" from="west" to="east" priority="1">
        <lane id="eastbound_0" index="0" speed="25.00" length="1500.00" shape="0.00,1.75 1500.00,1.75"/>
        <lane id="eastbound_1" index="1" speed="25.00" length="1500.00" shape="0.00,5.25 1500.00,5.25"/>
    </edge>
    
    <!-- Westbound road (opposite direction) -->
    <edge id="westbound" from="east" to="west" priority="1">
        <lane id="westbound_0" index="0" speed="25.00" length="1500.00" shape="1500.00,8.75 0.00,8.75"/>
    </edge>

    <junction id="west" type="dead_end" x="0.00" y="5.25" incLanes="westbound_0" intLanes="" shape="0.00,10.50 0.00,0.00"/>
    <junction id="east" type="dead_end" x="1500.00" y="5.25" incLanes="eastbound_0 eastbound_1" intLanes="" shape="1500.00,0.00 1500.00,10.50"/>

</net>"""
        
        with open("network.net.xml", "w") as f:
            f.write(net_xml)
        
        # Create route file with improved vehicle setup
        route_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Eastbound route (west to east) -->
    <route id="route_eastbound" edges="eastbound"/>
    
    <!-- Westbound route (east to west) -->
    <route id="route_westbound" edges="westbound"/>
    
    <!-- Vehicle types -->
    <vType id="ego_type" maxSpeed="30" accel="3.0" decel="4.5" sigma="0.0" length="4.5" minGap="2.0"/>
    <vType id="slow_type" maxSpeed="6" accel="1.0" decel="3.0" sigma="0.0" length="4.5" minGap="2.0"/>
    <vType id="oncoming_type" maxSpeed="20" accel="2.0" decel="4.0" sigma="0.0" length="4.5" minGap="2.0"/>
    
    <!-- Ego vehicle: fast, wants to overtake -->
    <vehicle id="ego" type="ego_type" route="route_eastbound" depart="0" departLane="0" departPos="100" departSpeed="20">
        
        <param key="lcKeepRight" value="1.0"/>
        <param key="lcAssertive" value="1.0"/>
    </vehicle>
    
    <!-- Front vehicle: very slow, persistent -->
    <vehicle id="front" type="slow_type" route="route_eastbound" depart="0" departLane="0" departPos="250" departSpeed="4">
        <param key="lcStrategic" value="0.0"/>
        <param key="lcKeepRight" value="1.0"/>
    </vehicle>
    
    <!-- Oncoming vehicles with staggered timing -->
    <vehicle id="oncoming1" type="oncoming_type" route="route_westbound" depart="30" departPos="400" departSpeed="16">
        <param key="lcStrategic" value="0.0"/>
    </vehicle>
    
    <vehicle id="oncoming2" type="oncoming_type" route="route_westbound" depart="60" departPos="300" departSpeed="15">
        <param key="lcStrategic" value="0.0"/>
    </vehicle>
    
</routes>"""
        
        with open("routes.rou.xml", "w") as f:
            f.write(route_xml)
        
        # Create configuration file
        config_xml = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="600"/>
        <step-length value="0.1"/>
    </time>
    
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <lateral-resolution value="0.8"/>
        <ignore-route-errors value="true"/>
    </processing>
    
</configuration>"""
        
        with open("simulation.sumocfg", "w") as f:
            f.write(config_xml)
    
    def get_vehicle_state(self, vehicle_id: str) -> Optional[VehicleState]:
        """Get vehicle state with lane information"""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return None
                
            position = traci.vehicle.getPosition(vehicle_id)[0]  # x coordinate
            speed = traci.vehicle.getSpeed(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_position = traci.vehicle.getLanePosition(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            
            return VehicleState(vehicle_id, position, speed, lane_id, lane_position, lane_index)
        except:
            return None
    

        # === Helper functions updated to lane‑local coordinates ===
    def compute_return_position(self,
                                conflict_lane_pos: float,
                                vehicle_f_speed: float,
                                ego_speed: float,
                                TLC: float = 3.0) -> float:
        """Return S_switch (論文式 6) **沿車道方向的縱向座標**"""
        s_conflict_future = conflict_lane_pos + vehicle_f_speed * (TLC / 2.0)
        d_switch = ego_speed * (TLC / 2.0)
        return max(s_conflict_future - d_switch, 0.0)

    def should_start_overtake(self, ego_state: VehicleState, vehicle_f_state: VehicleState) -> bool:
        """判斷是否觸發超車意圖（使用 lane_position 取值）"""
        DLC = 5.0
        Dsafe = 5.0
        V_expect = 18.0

        D_threshold = vehicle_f_state.lane_position - (2 * DLC + Dsafe)
        return (ego_state.lane_position < D_threshold) and (vehicle_f_state.speed < V_expect)

    def control_ego_vehicle(self, action: GameAction, ego_state: VehicleState, vehicle_f_state: Optional[VehicleState]):
        """Control ego vehicle with improved overtaking logic (with phase transition logs)"""

        if action == GameAction.NOT_YIELD and vehicle_f_state and "eastbound" in ego_state.lane_id:

            if self.current_phase == OvertakingPhase.LANE_KEEPING and ego_state.lane_index == 0:
                if self.should_start_overtake(ego_state, vehicle_f_state):
                    self.return_position = self.compute_return_position(vehicle_f_state.lane_position, vehicle_f_state.speed, ego_state.speed)
                    self.current_phase = OvertakingPhase.OVERTAKING_PHASE_1
                    self.lane_change_duration = 0
                    traci.vehicle.changeLane(self.ego_vehicle_id, 1, 3.0)
                    print(f"[PHASE] LK → P1  (trigger)")

            elif self.current_phase == OvertakingPhase.OVERTAKING_PHASE_1:
                self.lane_change_duration += 1
                print(f"[DBG P1] lanePos e={ego_state.lane_position:.2f}  f={vehicle_f_state.lane_position:.2f}")
                if ego_state.lane_index == 1 and (
                    ego_state.lane_position >= vehicle_f_state.lane_position - 2.0 or
                    self.lane_change_duration >= 30
                ):
                    self.current_phase = OvertakingPhase.OVERTAKING_PHASE_2
                    print(f"[PHASE] P1 → P2  (oncoming lane driving)")

            elif self.current_phase == OvertakingPhase.OVERTAKING_PHASE_2:
                if ego_state.lane_position >= self.return_position:
                    print(f"[DBG P2] ego_pos={ego_state.lane_position:.2f}, return_pos={self.return_position:.2f}")
                    traci.vehicle.changeLane(self.ego_vehicle_id, 0, 3.0)
                    self.current_phase = OvertakingPhase.OVERTAKING_PHASE_3
                    print(f"[PHASE] P2 → P3  (prepare to return)")

            elif self.current_phase == OvertakingPhase.OVERTAKING_PHASE_3:
                ego_state = self.get_vehicle_state(self.ego_vehicle_id)  # 🔁 強制更新
                lane_index = ego_state.lane_index
                lane_pos = ego_state.lane_position

                print(f"[DBG P3] phase_check, lane_index={lane_index}, ego_pos={lane_pos:.2f}, return_pos={self.return_position:.2f}")

                if lane_index == 0:
                    print("[DBG P3] lane_index == 0 判斷成立")
                    self.current_phase = OvertakingPhase.LANE_KEEPING
                    print(f"[PHASE] P3 → LK  (return complete)")
                else:
                    print("[DBG P3] 尚未切回原道")




        else:
            traci.vehicle.changeLane(self.ego_vehicle_id, 0, 3.0)




    def run_simulation(self):
        """Run simulation"""
        self.create_sumo_config()
        
        # Start SUMO
        sumo_cmd = [os.path.join(sumo_home, "bin", "sumo-gui.exe"), 
                   "-c", "simulation.sumocfg",]# "--start"]
        
        traci.start(sumo_cmd)
        
        step = 0
        try:
            while traci.simulation.getMinExpectedNumber() > 0 and step < 6000:
                traci.simulationStep()
                
                # Get vehicle states
                ego_state = self.get_vehicle_state(self.ego_vehicle_id)
                vehicle_f_state = self.get_vehicle_state(self.front_vehicle_id)
                
                # 獲取對向車輛
                oncoming_vehicles = []
                for vehicle_id in self.oncoming_vehicle_ids:
                    state = self.get_vehicle_state(vehicle_id)
                    if state and "westbound" in state.lane_id:
                        oncoming_vehicles.append(state)
                
                if ego_state:
                    # Make overtaking decision
                    decision = self.decision_maker.make_overtaking_decision(
                        ego_state, vehicle_f_state, oncoming_vehicles)
                    
                    # Control vehicle
                    self.control_ego_vehicle(decision, ego_state, vehicle_f_state)
                    
                    ego_state = self.get_vehicle_state(self.ego_vehicle_id)
                    vehicle_f_state = self.get_vehicle_state(self.front_vehicle_id)
                    # Record data
                    oncoming_positions = [v.position for v in oncoming_vehicles] if oncoming_vehicles else []
                    front_speed = vehicle_f_state.speed if vehicle_f_state else 0
                    front_pos = vehicle_f_state.position if vehicle_f_state else None
                    
                    self.simulation_data.append({
                        'time': step * 0.1,
                        'ego_pos': ego_state.position,
                        'ego_speed': ego_state.speed,
                        'ego_lane': ego_state.lane_index,
                        'decision': decision.name,
                        'phase': self.current_phase.name,
                        'front_pos': front_pos,
                        'front_speed': front_speed,
                        'oncoming_count': len(oncoming_vehicles),
                        'oncoming_positions': oncoming_positions,
                        'speed_difference': ego_state.speed - front_speed if vehicle_f_state else 0,
                        'commitment': self.overtaking_commitment
                    })
                    # print(f"Time={step * 0.1:.1f}s Phase={self.current_phase.name}")

                
                step += 1
                
        finally:
            traci.close()
            
        self.plot_results()
    
    def plot_results(self):
        """Plot results showing real overtaking behavior"""
        if not self.simulation_data:
            return
            
        import pandas as pd
        
        df = pd.DataFrame(self.simulation_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Position plot with lane information
        colors = ['blue' if lane == 0 else 'red' for lane in df['ego_lane']]
        ax1.scatter(df['time'], df['ego_pos'], c=colors, s=15, alpha=0.8, label='Ego Vehicle')
        
        if df['front_pos'].notna().any():
            ax1.plot(df['time'], df['front_pos'], label='Front Vehicle (Slow)', linewidth=2, color='green')
        
        # Plot oncoming vehicles
        for i, row in df.iterrows():
            if row['oncoming_positions']:
                for j, pos in enumerate(row['oncoming_positions']):
                    if i == 0:
                        ax1.plot(row['time'], pos, 'orange', marker='.', markersize=3, label=f'Oncoming {j+1}' if j < 2 else "")
                    else:
                        ax1.plot(row['time'], pos, 'orange', marker='.', markersize=3)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Position (meters)')
        ax1.set_title('Vehicle Positions (Blue=Right Lane, Red=Left Lane/Overtaking)')
        ax1.legend()
        ax1.grid(True)
        
        # Lane usage and commitment
        ax2.plot(df['time'], df['ego_lane'], 'b-', linewidth=2, label='Ego Vehicle Lane')
        
        # Highlight commitment periods
        commitment_mask = df['commitment']
        if commitment_mask.any():
            ax2.fill_between(df['time'], 0, df['ego_lane'], 
                           where=commitment_mask, alpha=0.3, color='red', 
                           label='Committed to Overtaking')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Lane (0=Right, 1=Left)')
        ax2.set_title('Lane Usage and Overtaking Commitment')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True)
        
        # Speed comparison
        ax3.plot(df['time'], df['ego_speed'], label='Ego Vehicle Speed', linewidth=2, color='blue')
        if df['front_speed'].notna().any():
            ax3.plot(df['time'], df['front_speed'], label='Front Vehicle Speed', linewidth=2, color='green')
        ax3.axhline(y=18, color='orange', linestyle='--', alpha=0.7, label='Expected Speed')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed During Overtaking Maneuver')
        ax3.legend()
        ax3.grid(True)
        
        # Overtaking phase evolution
        phase_mapping = {
            'LANE_KEEPING': 0,
            'OVERTAKING_PHASE_1': 1,
            'OVERTAKING_PHASE_2': 2, 
            'OVERTAKING_PHASE_3': 3,
            'WAITING': 4,
            'RETURN_TO_ORIGINAL': 5
        }
        phases = [phase_mapping.get(p, 0) for p in df['phase']]
        ax4.plot(df['time'], phases, 's-', linewidth=2, markersize=4, color='purple')
        
        # Highlight active overtaking
        overtaking_mask = [p > 0 for p in phases]
        if any(overtaking_mask):
            ax4.fill_between(df['time'], 0, [p if p > 0 else 0 for p in phases], 
                           alpha=0.3, color='red', label='Overtaking Active')
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Overtaking Phase')
        ax4.set_title('Detailed Overtaking Phase Evolution')
        ax4.set_yticks(list(phase_mapping.values()))
        ax4.set_yticklabels(list(phase_mapping.keys()), rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_overtaking_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Enhanced statistics
        print("\n=== 改進版超車行為模擬結果 ===")
        print(f"總模擬時間: {df['time'].max():.1f} 秒")
        print(f"主車最終位置: {df['ego_pos'].iloc[-1]:.1f} 米")
        
        if df['front_pos'].notna().any():
            final_front_pos = df['front_pos'].dropna().iloc[-1] if not df['front_pos'].dropna().empty else None
            if final_front_pos is not None:
                print(f"前車最終位置: {final_front_pos:.1f} 米")
                distance_gained = df['ego_pos'].iloc[-1] - final_front_pos
                print(f"最終超越距離: {distance_gained:.1f} 米")
            else:
                print("前車已離開模擬範圍")
        
        # 分析車道使用
        lane_0_time = sum(1 for lane in df['ego_lane'] if lane == 0) * 0.1
        lane_1_time = sum(1 for lane in df['ego_lane'] if lane == 1) * 0.1
        print(f"右車道行駛時間: {lane_0_time:.1f} 秒")
        print(f"左車道行駛時間: {lane_1_time:.1f} 秒")
        
        # 分析超車階段
        overtaking_phases = [p for p in df['phase'] if 'OVERTAKING' in p]
        commitment_time = sum(df['commitment']) * 0.1
        
        if overtaking_phases:
            print("✓ 執行了超車行為")
            print(f"超車階段總時間: {len(overtaking_phases) * 0.1:.1f} 秒")
            print(f"承諾超車時間: {commitment_time:.1f} 秒")
            
            if df['front_pos'].notna().any() and final_front_pos is not None:
                if distance_gained > 50:
                    print("✓ 成功超越前車")
                else:
                    print("✗ 超車效果有限")
        else:
            print("✗ 未執行超車行為")
            
        decisions = [1 if d == 'NOT_YIELD' else 0 for d in df['decision']]
        overtaking_decisions = sum(decisions)
        total_decisions = len(decisions)
        print(f"超車決策次數: {overtaking_decisions}/{total_decisions}")
        print(f"超車決策比例: {overtaking_decisions/total_decisions*100:.1f}%")
        print(f"平均速度: {df['ego_speed'].mean():.1f} m/s")
        print(f"最高速度: {df['ego_speed'].max():.1f} m/s")

if __name__ == "__main__":
    print("啟動改進版超車行為模擬...")
    print("改進內容:")
    print("- 增加道路長度到1500米")
    print("- 定義專用車輛類型")
    print("- 增加超車承諾機制")
    print("- 更激進的決策參數")
    print("- 延後對向車輛出現時間")
    print("- 更寬鬆的觸發條件")
    
    simulation = SUMOOvertakingSimulation()
    simulation.run_simulation()
    
    print("模擬完成!")
