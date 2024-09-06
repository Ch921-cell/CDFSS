# -*- coding: utf-8 -*-
import subprocess

combinations = [
	{"benchmark": "deepglobe", "datapath": "../ABCDFSS/datasets/deepglobe/", "seed": "0"},
	{"benchmark": "fss", "datapath": "../ABCDFSS/datasets/fss1000-a-1000-class-fewshot-segmentation", "seed": "0"},
	{"benchmark": "isic", "datapath": "../ABCDFSS/datasets/isic2018-classwise", "seed": "0"},

]
shot_n = [1, 5]
for shot in shot_n:
	for combo in combinations:
		command = f"python main.py --benchmark {combo['benchmark']} --datapath {combo['datapath']} --adapt-to every-episode --postprocessing dynamic --nshot {shot} --seed {combo['seed']}"
		print(f"Running: {command}")
		result = subprocess.run(command, shell=True, capture_output=True, text=True)
		# 使用 benchmark 和 seed 创建一个独立的日志文件
		log_filename = f"log_{combo['benchmark']}_seed{combo['seed']}_nshot{shot}.txt"
		with open(log_filename, 'w') as log_file:
			log_file.write(f"Command: {command}\n")
			log_file.write("Output:\n")
			log_file.write(result.stdout)
			log_file.write("\nErrors:\n")
			log_file.write(result.stderr)
		
		print(f"Finished: {command}\nLogs are saved in {log_filename}")
