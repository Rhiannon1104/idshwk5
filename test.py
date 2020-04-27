from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

trainlist = []
testlist=[]
class Domain:
	def __init__(self,_name,_label,_length,_numbers,_entropy):
		self.name = _name
		self.label = _label
		self.length=_length
		self.numbers=_numbers
		self.entropy=_entropy

	def returnData(self):
		return [self.length,self.numbers, self.entropy]

	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1

	def returnName(self):
		return self.name
		
def count_num(name):
	num=0
	for i in name:
		if i.isdigit():
			num+=1
	return num

def cal_entropy(name):
	h = 0.0
	count = 0
	letter = [0] * 26
	name= name.lower()
	for i in range(len(name)):
		if name[i].isalpha():
			letter[ord(name[i]) - ord('a')] += 1
			count += 1
	for i in range(26):
		p = 1.0 * letter[i] / count
		if p > 0:
			h += -(p * math.log(p, 2))
	return h

def initTrain(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = tokens[1]
			length=len(name)
			numbers=count_num(name)
			entropy=cal_entropy(name)
			trainlist.append(Domain(name,label,length,numbers,entropy))

def initTest(filename):
	with open(filename) as f:
		for line in f:
			line=line.strip()
			if line.startswith("#") or line=="":
				continue
			length=len(line)
			numbers=count_num(line)
			entropy=cal_entropy(line)
			testlist.append(Domain(line," ",length,numbers,entropy))

def main():
	initTrain("train.txt")
	initTest("test.txt")
	featureMatrix = []
	labelList = []
	for item in trainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())
	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	f = open("result.txt", "w")
	for item in testlist:
		s = item.returnName() + ","
		if clf.predict([item.returnData()]) == 0:
			s = s + "notdga"
		else:
			s = s + "dga"
		s = s + '\n'
		f.write(s)
	f.close()
	

if __name__ == '__main__':
	main()
