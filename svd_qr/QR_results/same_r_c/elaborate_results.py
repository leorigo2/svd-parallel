import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df_serial = pd.read_csv('results_serial.txt', sep=r'\s+')
dict_serial = df_serial.to_dict(orient='list')
serial_times = dict_serial['time'] 

plt.style.use('bmh') 
ticks_x = [2, 4, 8, 16, 20, 50, 75, 96, 150]
ticks_y = ticks_x

files = {
    'serial': 'results_serial.txt',
    '1x2': 'results_parallel_1x2.txt',
    '2x2': 'results_parallel_2x2.txt',
    '2x4': 'results_parallel_2x4.txt',
    '2x8': 'results_parallel_2x8.txt',
    '4x5': 'results_parallel_4x5.txt',
    '2x25': 'results_parallel_2x25.txt',
    '3x25': 'results_parallel_3x25.txt',
    '4x24': 'results_parallel_4x24.txt',
    '5x30': 'results_parallel_5x30.txt',
    '2x8_pack_excl': "nodes_placing/results_parallel_2x8_pack_excl.txt",
    '2x8_pack': "nodes_placing/results_parallel_2x8_pack.txt",
    '2x8_scatter': "nodes_placing/results_parallel_2x8_scatter.txt",
    '2x8_scatter_excl': "nodes_placing/results_parallel_2x8_scatter_excl.txt"
}

results = {}

for name, filename in files.items():
    df = pd.read_csv(filename, sep=r'\s+')
    data = df.to_dict(orient='list')
    if name == 'serial': 
        num_p = 1
    else: 
        cores, cpus = map(int, name.split('_')[0].split('x'))
        num_p = cores * cpus
    data['num_p'] = num_p
    data['speedup'] = [t_s / t_p for t_s, t_p in zip(serial_times, data['time'])]
    data['efficiency'] = [s / num_p for s in data['speedup']]
    
    results[name] = data


mean_times = {name: np.mean(data['time']) for name, data in results.items()}
best_config = min(mean_times, key=mean_times.get)
print(f"Best configuration is: {best_config}")

for name, data in results.items():
    sort_indices = np.argsort(data['elements'])
    
    results[name]['elements'] = np.array(data['elements'])[sort_indices].tolist()
    results[name]['efficiency'] = np.array(data['efficiency'])[sort_indices].tolist()
    results[name]['time'] = np.array(data['time'])[sort_indices].tolist()
    results[name]['speedup'] = np.array(data['speedup'])[sort_indices].tolist()


data_1 = results['2x8']
data_2 = results['2x8_pack_excl']
data_3 = results['2x8_pack']
data_4 = results['2x8_scatter_excl']
data_5 = results['2x8_scatter']

num_p = data['num_p']

plt.figure(figsize=(10, 8))
plt.plot(data_1['elements'], data_1['efficiency'], 
         marker='o', linestyle='-', color='blue', label='2x8 Default')
plt.plot(data_2['elements'], data_2['efficiency'], 
         marker='s', linestyle='-', color='red', label='2x8 Pack Excl')
plt.plot(data_3['elements'], data_3['efficiency'], 
         marker='^', linestyle='-', color='green', label='2x8 Pack')
plt.plot(data_4['elements'], data_4['efficiency'], 
         marker='d', linestyle='-', color='purple', label='2x8 Scatter Excl')
plt.plot(data_5['elements'], data_5['efficiency'], 
         marker='d', linestyle='-', color='orange', label='2x8 Scatter')
plt.title(f'Efficiency vs Number of Elements (2x8)')
plt.xlabel('Number of Elements')
plt.ylabel(f'Efficiency (Speedup / {num_p})')
plt.grid(True)
plt.legend()


plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
plt.gca().get_yaxis().set_major_formatter(ScalarFormatter())

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('efficiency_2x8.png', dpi=300)
plt.close()


p_groups = {}

for name, data in results.items():
    p = data['num_p']
    mean_t = np.mean(data['time'])
    if p not in p_groups:
        p_groups[p] = []
    p_groups[p].append((name, mean_t))


filtered_results = {}

for p, candidates in p_groups.items():
    best_name, best_mean = min(candidates, key=lambda x: x[1])
    
    filtered_results[best_name] = results[best_name]

results = filtered_results


any_key = list(results.keys())[0]
n_elements = len(results[any_key]['elements'])

idx_min = 0                  # Index for Smallest
idx_mid = n_elements // 2    # Index for Mid
idx_max = -1                 # Index for Largest

label_min = f"Smallest ({results[any_key]['elements'][idx_min]})"
label_mid = f"Median ({results[any_key]['elements'][idx_mid]})"
label_max = f"Largest ({results[any_key]['elements'][idx_max]})"


plot_data = []
seen_p = {}

for name, data in results.items():
    p = data['num_p']

    s_min = data['speedup'][idx_min]
    s_mid = data['speedup'][idx_mid]
    s_max = data['speedup'][idx_max]

    e_min = data['efficiency'][idx_min]
    e_mid = data['efficiency'][idx_mid]
    e_max = data['efficiency'][idx_max]
    
    plot_data.append((p, s_min, s_mid, s_max, e_min, e_mid, e_max))

plot_data.sort(key=lambda x: x[0])

x_vals = [x[0] for x in plot_data]
y_s_min = [x[1] for x in plot_data]
y_s_mid = [x[2] for x in plot_data]
y_s_max = [x[3] for x in plot_data]

y_e_min = [x[4] for x in plot_data]
y_e_mid = [x[5] for x in plot_data]
y_e_max = [x[6] for x in plot_data]



plt.figure(figsize=(10, 8))
plt.plot(x_vals, y_s_min, marker='o', label=label_min)
plt.plot(x_vals, y_s_mid, marker='s', label=label_mid)
plt.plot(x_vals, y_s_max, marker='^', label=label_max)
plt.plot(x_vals, x_vals, '--', color='gray', alpha=0.5, label='Ideal Linear Speedup')
plt.title('Scaling Speedup', fontsize=16)
plt.xlabel('Number of CPUs', fontsize=12)
plt.ylabel('Speedup (T_serial / T_parallel)', fontsize=12)

plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(ticks_x)
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
plt.yticks(ticks_y)
plt.gca().get_yaxis().set_major_formatter(ScalarFormatter())

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('svd_speedup.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(x_vals, y_e_min, marker='o', label=label_min)
plt.plot(x_vals, y_e_mid, marker='s', label=label_mid)
plt.plot(x_vals, y_e_max, marker='^', label=label_max)
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
plt.title('Parallel Efficiency', fontsize=16)
plt.xlabel('Number of CPUs', fontsize=12)
plt.ylabel('Efficiency (Speedup / NCpus)', fontsize=12)

plt.xscale('log', base=2)
plt.xticks(ticks_x)
plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())

max_eff = max(y_e_max)
plt.ylim(0, max(1.1, max_eff * 1.15))
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('svd_efficiency.png', dpi=300)
plt.close()