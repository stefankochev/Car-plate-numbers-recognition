import math
import operator 


class Knn:
	def __init__(self, samples, responses):
		self.samples = samples.tolist()
		responses = responses.tolist()
		self.responses = [item for subl in responses for item in subl]


	def distance(self, vec1, vec2):
		dist = sum(list(map(lambda (x, y): (x - y)**2, zip(list(vec1), list(vec2)))))

		return math.sqrt(dist)


	def get_neighbors(self, data, k):
		neighbors =  [(index, sample, self.distance(data, sample)) for index, sample in enumerate(self.samples)]
		neighbors.sort(key=operator.itemgetter(2))

		return neighbors[:k]


	def get_response(self, neighbors):
		count = {}

		for index, sample, distance in neighbors:
			response = self.responses[index]
			if response in count:
				count[response] += 1
			else:
				count[response] = 1


		sortedVotes = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)

		#print sortedVotes
		cond = len(sortedVotes) > 1 and (sortedVotes[0][1] == sortedVotes[1][1] or sortedVotes[0][1] == 1)

		return sortedVotes[0][0], cond

	def find_nearest(self, data, k):
		conflict = True
		while(conflict and k < 8):
			neighbors = self.get_neighbors(data.tolist()[0], k)
			res, conflict = self.get_response(neighbors)
			k+=1


		return res
