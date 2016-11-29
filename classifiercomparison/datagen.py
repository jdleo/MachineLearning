import random

def classNumber(x):
	if x % 5 == 0:
		return 1
	elif x % 4 == 0:
		return 2
	elif x % 3 == 0:
		return 3
	elif x % 2 == 0:
		return 4
	else:
		return 5


with open('data.txt', 'w') as f:
	for _ in range(0,4800):
		s3t = []
		for __ in range(0,5):
			s3t.append(random.randint(0,5))

		classifier = classNumber(sum(s3t))

		s3t.append(classifier)

		#converting each entry to str for join
		s3t = [str(e) for e in s3t]
		joined = ','.join(s3t)
		f.write(joined + '\n')
