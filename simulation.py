import os
import sys
import traci
import sumolib
import time

# 設定 SUMO_HOME 路徑
sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
os.environ['SUMO_HOME'] = sumo_home

# 加入 tools 路徑
tools = os.path.join(sumo_home, 'tools')
if tools not in sys.path:
    sys.path.append(tools)

# 加入 bin 路徑
sumo_bin = os.path.join(sumo_home, 'bin')
if sumo_bin not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin

# 再次確認 SUMO_HOME 是否存在
if 'SUMO_HOME' not in os.environ:
    sys.exit("請設定環境變數 'SUMO_HOME'")

class SUMOSimulation:
    def __init__(self, config_file="simulation.sumocfg"):
        self.config_file = config_file
        self.step = 0

    def start_gui_simulation(self):
        sumo_cmd = [sumolib.checkBinary('sumo-gui'), "-c", self.config_file]
        traci.start(sumo_cmd)
        print("✅ SUMO GUI 仿真啟動")

    def start_cmd_simulation(self):
        sumo_cmd = [sumolib.checkBinary('sumo'), "-c", self.config_file]
        traci.start(sumo_cmd)
        print("✅ SUMO 命令列仿真啟動")

    def monitor_simulation(self):
        vehicles = traci.vehicle.getIDList()
        if vehicles:
            print(f"時間步: {self.step}, 車輛數量: {len(vehicles)}")
            for veh_id in vehicles:
                speed = traci.vehicle.getSpeed(veh_id)
                pos = traci.vehicle.getPosition(veh_id)
                print(f"  車輛 {veh_id}: 速度={speed:.2f}, 位置={pos}")

    def run_simulation(self, use_gui=True, duration=100):
        try:
            if use_gui:
                self.start_gui_simulation()
            else:
                self.start_cmd_simulation()

            while self.step < duration and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.step += 1

                if self.step % 10 == 0:
                    self.monitor_simulation()

                if use_gui:
                    time.sleep(0.1)
        except traci.exceptions.FatalTraCIError:
            print("仿真已結束或 GUI 關閉")
        finally:
            traci.close()
            print("✅ 仿真完成")

def main():
    print("=== SUMO 交通仿真系統 ===")
    print("1. GUI模式仿真")
    print("2. 命令列模式仿真")

    choice = input("請選擇模式 (1/2): ").strip()
    simulation = SUMOSimulation()

    if choice == "1":
        simulation.run_simulation(use_gui=True, duration=1000)
    elif choice == "2":
        simulation.run_simulation(use_gui=False, duration=1000)
    else:
        print("無效選擇，預設使用 GUI 模式")
        simulation.run_simulation(use_gui=True, duration=1000)

if __name__ == "__main__":
    main()
