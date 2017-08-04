import sys
nums = []
for l in open(sys.argv[1]).readlines():
	nums.append(float(l))

average = sum(nums)/len(nums)
sd = (sum((n - average)**2 for n in nums)/len(nums)) ** 0.5
sd = round(sd, 2)
print("{:.2} {}".format(average, sd))
