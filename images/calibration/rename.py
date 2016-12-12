import glob, os
os.chdir(".")
i = 1
for file in glob.glob("*.JPG"):
	os.system('mv ' + file + ' im%d.jpg' % i)
	i = i + 1
