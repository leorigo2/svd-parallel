import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Serial Data
df_serial = pd.read_csv('results_serial.txt', sep=r'\s+')
dict_serial = df_serial.to_dict(orient='list')
serial_times = dict_serial['time'] 

files = {
    '2x2':   'results_parallel_2x2.txt',
    '4x2':  'results_parallel_4x2.txt',
    '2x4': 'results_parallel_2x4.txt',
    '2x8':  'results_parallel_2x8.txt',
    '8x2':   'results_parallel_8x2.txt',
    '2x16':  'results_parallel_2x16.txt',
    '16x2':   'results_parallel_16x2.txt',
}

results = {}

for name, filename in files.items():
    df = pd.read_csv(filename, sep=r'\s+')
    data = df.to_dict(orient='list')
    rows, cols = map(int, name.split('x'))
    num_p = rows * cols
    data['num_p'] = num_p
    data['speedup'] = [t_s / t_p for t_s, t_p in zip(serial_times, data['time'])]
    data['efficiency'] = [s / num_p for s in data['speedup']]
    
    results[name] = data


mean_times = {name: np.mean(data['time']) for name, data in results.items()}
best_config = min(mean_times, key=mean_times.get)
print(f"Best Configuration: {best_config}")

for name, data in results.items():
    sort_indices = np.argsort(data['elements'])
    
    results[name]['elements'] = np.array(data['elements'])[sort_indices].tolist()
    results[name]['efficiency'] = np.array(data['efficiency'])[sort_indices].tolist()
    results[name]['time'] = np.array(data['time'])[sort_indices].tolist()
    results[name]['speedup'] = np.array(data['speedup'])[sort_indices].tolist()


data = results[best_config]
num_p = data['num_p']

plt.figure(figsize=(10, 6))
plt.plot(data['elements'], data['efficiency'], marker='o', linestyle='-', color='blue')

plt.title(f'Efficiency vs Number of Elements ({best_config})')
plt.xlabel('Number of Elements')
plt.ylabel(f'Efficiency (Speedup / {num_p})')
plt.grid(True)

plt.savefig(f'efficiency_{best_config}.png')
plt.show()

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

plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_s_min, marker='o', label=label_min)
plt.plot(x_vals, y_s_mid, marker='s', label=label_mid)
plt.plot(x_vals, y_s_max, marker='^', label=label_max)

plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()
plt.savefig('speedup_vs_processes.png')
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_e_min, marker='o', label=label_min)
plt.plot(x_vals, y_e_mid, marker='s', label=label_mid)
plt.plot(x_vals, y_e_max, marker='^', label=label_max)

plt.title('Efficiency vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency')
plt.grid(True)
plt.legend()
plt.savefig('efficiency_vs_processes.png')
plt.show()