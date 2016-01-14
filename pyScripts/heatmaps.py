import heatmap
import random
import os

for fileName in os.listdir("."):
	fileNameParts = fileName.split(" ")
	if fileNameParts[0]=='Confusion':
		allPoints = []
		fileNameParts[2] = fileNameParts[2][:fileNameParts[2].index('.')]
		with open(fileName,'r') as heatMap:
			first = True	
			count = 0
			for line in heatMap:
				if first:
					first = False
					continue
				nums = [int(float(i)) for i in line.split(",")[1:]]
				for i in range(len(nums)):
					for k in range(nums[i]):
						allPoints.append((1+i,11-count))
		#				allPoints.append((i+random.random()-0.5,11-count+random.random()-0.5))
				count+=1
		hm = heatmap.Heatmap()
		img = hm.heatmap(allPoints,dotsize=100,scheme='pgaitch',opacity=1,area=((0,0),(12,12)))
		img.save('HeatMap'+fileNameParts[2]+'.jpg')
