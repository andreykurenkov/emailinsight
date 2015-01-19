import os

frames = 80
cvsLine = []
cvsLines = [cvsLine]
for f in range(frames): 
    cvsLine+=("%s Hand 1 Frame %d"%(s,f) for s in ['x','y','z','yaw','pitch','roll','thumb','forefinger','middle','ring','little'])
    cvsLine+= ("%s Hand 2 Frame %d"%(s,f) for s in ['x','y','z','yaw','pitch','roll','thumb','forefinger','middle','ring','little'])
cvsLine.append("sign\n")
folders = os.listdir(".")
for folder in folders:
    if not os.path.isdir(folder):
        continue
    signFiles = os.listdir("./"+folder)
    personName = folder[:-1]
    maxCount = 0 
    for signFile in signFiles:
        signName = signFile[:signFile.index(".")-2]
        count = 0
        cvsLine = []
        cvsLines.append(cvsLine)
        with open("./"+folder+"/"+signFile,'r') as sFile:
            for line in sFile:
                cvsLine+=line.strip().split("\t")
                count+=1
                if count>=frames:
                    break
            for x in range(frames-count):
                cvsLine+=['-1']*22
            cvsLine.append('%s\n'%signName)

remove = []
for i in range(len(cvsLines[0])-1):
    if all([abs(float(line[i]))<0.0001 for line in cvsLines[1:]]):
        remove.append(i)

for index in range(len(cvsLines)):
    count = 0
    for r in remove:
        cvsLines[index].pop(r-count)
        count+=1

with open("signs.csv",'w') as outputCVS:
    for line in cvsLines:
        outputCVS.write(','.join(line))
