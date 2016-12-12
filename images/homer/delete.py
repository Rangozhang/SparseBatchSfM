import glob, os
os.chdir(".")
for i in range(20, 26):
	os.system('mv %04d.JPG' % i + ' %04d.JPG' % (i-1))
	print ('mv %04d.JPG' % i + ' %04d.JPG' % (i-1))
