import Rechenstudie_MGG_BA as rs
import data_management_BA as dm

rs.run_instance_set(2,100,1,50,"set_1_runtime.csv","set_1_results_instance_size_")
dm.load_results_and_plot("")
dm.read_runtimes_and_plot("set_1_runtime.csv")

rs.run_instance_set(110,400,10,10,"set_2_runtime.csv","set_2_results_instance_size_")
dm.load_results_and_plot("")
dm.read_runtimes_and_plot("set_2_runtime.csv")
