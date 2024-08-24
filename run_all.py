# -*- coding: utf-8 -*-
import subprocess

combinations = [
	{"benchmark": "isic", "datapath": "./datasets/isic2018-classwise"},
	{"benchmark": "isic", "datapath": "./datasets/isic2018-classwise"}
]
seeds = range(0, 2)
for seed in seeds:
	for combo in combinations:
		command = f"python main.py --benchmark {combo['benchmark']} --datapath {combo['datapath']} --adapt-to every-episode --postprocessing dynamic --nshot 1 --seed 0"
		print(f"Running: {command}")
		result = subprocess.run(command, shell=True, capture_output=True, text=True)
		# 使用 benchmark 和 seed 创建一个独立的日志文件
		log_filename = f"log_{combo['benchmark']}_time{seed}_nshot1.txt"
		with open(log_filename, 'w') as log_file:
			log_file.write(f"Command: {command}\n")
			log_file.write("Output:\n")
			log_file.write(result.stdout)
			log_file.write("\nErrors:\n")
			log_file.write(result.stderr)
		
		print(f"Finished: {command}\nLogs are saved in {log_filename}")
