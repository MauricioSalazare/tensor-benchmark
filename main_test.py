from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter


#%% Example 3: Generate a random radial network of 100 nodes and a maximum of 1 to 3 branches per node.
NODES = 1000
network_rnd = GridTensor.generate_from_graph(nodes=NODES, child=3, plot_graph=False)
active_ns = np.random.normal(50,  # Power in kW
                             scale=10,
                             size=(1_000, NODES-1)).round(3)  # Assume 1 slack variable
reactive_ns = (active_ns * .1).round(3)  # Constant PF of 0.1

start_time = perf_counter()
solution_rnd = network_rnd.run_pf(active_power=active_ns, reactive_power=reactive_ns)
end_time = perf_counter()

print(solution_rnd["v"])
print(f"Time: {end_time - start_time} sec.")
print(f"Iterations: {solution_rnd['iterations']}")