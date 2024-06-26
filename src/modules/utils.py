from typing import Dict, List
import matplotlib.pyplot as plt


def plot_lines(ret: Dict[str, List], emphasize_lines, name, x, range_=None, axes=None):
	if axes is None:
		fig, axes = plt.subplots(figsize=(10, 8))

	# x = list(range(len(ret[list(ret.keys())[0]])))
	for k, v in ret.items():
		if k in emphasize_lines:
			axes.plot(x, v, label=k, linewidth='3', alpha=1, marker='o', markersize=4)
		else:
			axes.plot(x, v, label=k, linewidth='1', alpha=0.7, marker='o', markersize=2)

	axes.set_xticks(x)
	axes.set_title(name)
	if range_:
		axes.set_ylim(*range_)

	if axes is None:
		#plt.savefig(name + '.png')
		plt.show()
	else:
		return axes
