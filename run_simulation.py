import os
import sys
import traci
import sumolib
import time
import random

# 設定 SUMO_HOME 路徑（請修改為您的實際安裝路徑）
sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
os.environ['SUMO_HOME'] = sumo_home

# 新增 SUMO tools 到 Python 路徑
tools = os.path.join(sumo_home, 'tools')
if tools not in sys.path:
    sys.path.append(tools)

# 新增 SUMO bin 到系統 PATH
sumo_bin = os.path.join(sumo_home, 'bin')
if sumo_bin not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin

# 確認SUMO_HOME環境變數
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("請設定環境變數 'SUMO_HOME'")

class SUMOSimulation:
    def __init__(self, config_file="simulation.sumocfg"):
        self.config_file = config_file
        self.step = 0
        
    def start_gui_simulation(self):
        """啟動SUMO GUI仿真"""
        # 使用SUMO GUI啟動仿真
        sumo_cmd = [sumolib.checkBinary('sumo-gui'), "-c", self.config_file]
        traci.start(sumo_cmd)
        print("SUMO GUI 仿真已啟動")
        
    def start_cmd_simulation(self):
        """啟動SUMO命令列仿真"""
        # 使用SUMO命令列啟動仿真
        sumo_cmd = [sumolib.checkBinary('sumo'), "-c", self.config_file]
        traci.start(sumo_cmd)
        print("SUMO 命令列仿真已啟動")
        
    def add_vehicles(self):
        """動態新增車輛"""
        # 新增前導車
        traci.vehicle.add("dynamic_leader", "route1", typeID="car", depart="now", 
                         departLane=1, departPos=0, departSpeed=10)
        
        # 新增跟車
        for i in range(3):
            vehicle_id = f"follower_{i+10}"
            depart_pos = random.uniform(0, 50)
            traci.vehicle.add(vehicle_id, "route1", typeID="car", depart="now",
                             departLane=1, departPos=depart_pos, departSpeed=8)
    
    def control_car_following(self):
        """控制車輛跟車行為"""
        vehicles = traci.vehicle.getIDList()
        
        for veh_id in vehicles:
            if "follower" in veh_id:
                # 取得車輛資訊
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getPosition(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                
                # 取得前方車輛
                leader_info = traci.vehicle.getLeader(veh_id)
                
                if leader_info:
                    leader_id, distance = leader_info
                    leader_speed = traci.vehicle.getSpeed(leader_id)
                    
                    # 實施自適應跟車控制
                    if distance < 20:  # 距離太近，減速
                        new_speed = max(0, speed - 2)
                        traci.vehicle.setSpeed(veh_id, new_speed)
                    elif distance > 50:  # 距離太遠，加速
                        new_speed = min(leader_speed + 5, 15)
                        traci.vehicle.setSpeed(veh_id, new_speed)
    
    def monitor_simulation(self):
        """監控仿真狀態"""
        vehicles = traci.vehicle.getIDList()
        if vehicles:
            print(f"時間步: {self.step}, 車輛數量: {len(vehicles)}")
            for veh_id in vehicles[:5]:  # 顯示前5輛車的資訊
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getPosition(veh_id)
                print(f"  車輛 {veh_id}: 速度={speed:.2f} m/s, 位置=({position[0]:.2f}, {position[1]:.2f})")
    
    def run_simulation(self, use_gui=True, duration=1000):
        """執行仿真"""
        try:
            # 啟動仿真
            if use_gui:
                self.start_gui_simulation()
            else:
                self.start_cmd_simulation()
            
            # 等待幾步後新增車輛
            while self.step < 20:
                traci.simulationStep()
                self.step += 1
            
            # 新增動態車輛
            self.add_vehicles()
            
            # 主要仿真迴圈
            while self.step < duration and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.step += 1
                
                # 控制跟車行為
                self.control_car_following()
                
                # 監控仿真（每10步顯示一次）
                if self.step % 10 == 0:
                    self.monitor_simulation()
                
                # 在GUI模式下暫停一下以便觀察
                if use_gui:
                    time.sleep(0.1)
                    
        except traci.exceptions.FatalTraCIError:
            print("仿真已結束或使用者關閉GUI")
        finally:
            traci.close()
            print("仿真完成")

def main():
    """主程式"""
    print("=== SUMO 交通仿真系統 ===")
    print("1. GUI模式仿真")
    print("2. 命令列模式仿真") 
    
    choice = input("請選擇模式 (1/2): ").strip()
    
    simulation = SUMOSimulation()
    
    if choice == "1":
        print("啟動GUI模式...")
        simulation.run_simulation(use_gui=True, duration=1000)
    elif choice == "2":
        print("啟動命令列模式...")
        simulation.run_simulation(use_gui=False, duration=1000)
    else:
        print("無效選擇，預設使用GUI模式")
        simulation.run_simulation(use_gui=True, duration=1000)

if __name__ == "__main__":
    main()
