from project_simulation_full import simulation_main
import pdb



f = open('simulation_input.txt', 'r')
lines = f.readlines()
lines = [line for line in lines if line[0]!= '#']
lines = [line for line in lines if line[0]!= '\n']

simulation_info_dict = {}
for line in lines:
	if ':' in line:
		trans = line.split()[1::]
		if len(trans) == 1:
			simulation_info_dict[line.split()[0][0:-1]] = trans[0]
		else:
			simulation_info_dict[line.split()[0][0:-1]] = trans

simulation_main(simulation_info_dict)
