import os, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


class BaseClassifier:

	def __init__(self):
		self.train_data = []
		self.train_labels = []
		self.test_data = []
		self.test_labels = []

	def load_data_from_dir(self, data_dir):
		data = []
		for curr_class in self.classes:
			dirname = os.path.join(data_dir, curr_class)
			for fname in os.listdir(dirname):
				with open(os.path.join(dirname, fname), 'r') as f:
					content = f.read()
					data.append((content, curr_class))

		random.shuffle(data)
		length = int(len(data)*0.9)

		self.train_data = [data[0] for data in data[:length]]
		self.train_labels = [data[1] for data in data[:length]]
		self.test_data = [data[0] for data in data[length:]]
		self.test_labels = [data[1] for data in data[length:]]

	def load_data(self, dataset):
		


class LinearSVCClassifier(BaseClassifier):
	# Linear Support Vector Classification
	# Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear 
	# rather than libsvm, so it has more flexibility in the choice of penalties and loss 
	# functions and should scale better to large numbers of samples.

	def __init__(self, classes=['pos', 'neg']):
		super().__init__()
		self.classes = classes
		self.classifier = None
		self.vectorizer = TfidfVectorizer(min_df=1,
										  max_df = 0.8,
										  sublinear_tf=True,
										  use_idf=True)

	def classification(self):
		self.classifier = svm.LinearSVC()
		train_vectors = self.vectorizer.fit_transform(self.train_data)
		self.classifier.fit(train_vectors, self.train_labels)
		return self.classifier

	def classification_report(self):
		test_vectors = self.vectorizer.transform(self.test_data)
		predicted = self.classifier.predict(test_vectors)
		print(classification_report(self.test_labels, predicted))

	def predict(self, data):
		# Transform data into np.array form
		test_vectors = self.vectorizer.transform(data)
		predicted = self.classifier.predict(test_vectors)

		for item, labels in zip(data, predicted):
			print('%s => %s' % (item, labels))


		
