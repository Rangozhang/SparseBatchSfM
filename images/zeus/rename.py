import glob, os
os.chdir(".")
i = 1
for file in glob.glob("*.JPG"):
	os.system('mv ' + file + ' %04d.JPG' % i)
	i = i + 1
